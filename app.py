# ╔══════════════════════════════════════════════════════════════════╗
#  app.py — Fake Logo Detector API  (Secured Production Version)
#  Vercel entry point — this file must be named app.py
# ╚══════════════════════════════════════════════════════════════════╝

import os, io, pickle, logging, time
from datetime import datetime
from functools import wraps

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── Config from .env ──────────────────────────
API_KEY          = os.environ.get("API_KEY", "change-me-secret")
ALLOWED_ORIGINS  = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
EMBEDDINGS_FILE  = os.environ.get("EMBEDDINGS_FILE", "./logo_embeddings_clip.pkl")
MAX_UPLOAD_MB    = int(os.environ.get("MAX_UPLOAD_MB", "10"))
RATE_LIMIT       = os.environ.get("RATE_LIMIT", "10 per minute")
IS_VERCEL        = bool(os.environ.get("VERCEL", False))

SIMILARITY_THRESHOLD = 0.80
HIGH_CONF_THRESHOLD  = 0.88
LOW_CONF_THRESHOLD   = 0.60
ALLOWED_MIME_TYPES   = {"jpeg", "png", "webp", "bmp"}

# ── Flask app ─────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app, origins=ALLOWED_ORIGINS)
limiter = Limiter(get_remote_address, app=app,
                  default_limits=[RATE_LIMIT], storage_uri="memory://")

# ── Feature extractor ─────────────────────────
class LightExtractor:
    def extract_from_pil(self, img):
        img = img.convert("RGB").resize((64, 64))
        arr = np.array(img).astype(np.float32)
        hist = []
        for ch in range(3):
            h, _ = np.histogram(arr[:,:,ch], bins=16, range=(0,255))
            hist.extend(h.tolist())
        gx = np.abs(np.diff(np.mean(arr,axis=2),axis=1)).mean()
        gy = np.abs(np.diff(np.mean(arr,axis=2),axis=0)).mean()
        vec = np.array(hist+[gx,gy,arr.mean()/255.0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec/norm if norm>0 else vec

class CLIPExtractor:
    def __init__(self):
        import torch, open_clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model,_,self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai")
        self.model.eval().to(self.device)
        self.torch = torch
    def extract_from_pil(self, img):
        t = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            f = self.model.encode_image(t)
        vec = f.squeeze().float().cpu().numpy()
        norm = np.linalg.norm(vec)
        return vec/norm if norm>0 else vec

def load_extractor():
    if IS_VERCEL:
        logger.info("Vercel mode — lightweight extractor")
        return LightExtractor()
    try:
        e = CLIPExtractor()
        logger.info("CLIP extractor loaded")
        return e
    except Exception as ex:
        logger.warning(f"CLIP failed ({ex}) — lightweight extractor")
        return LightExtractor()

extractor = load_extractor()

# ── Database ──────────────────────────────────
def load_database():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE,"rb") as f:
            db = pickle.load(f)
        logger.info(f"Database loaded — {len(db)} brand(s)")
        return db
    logger.warning("No database file found")
    return {}

database = load_database()

def save_database():
    with open(EMBEDDINGS_FILE,"wb") as f:
        pickle.dump(database,f)

# ── Security helpers ──────────────────────────
def validate_image(raw):
    if not raw:
        raise ValueError("Empty file.")
    import imghdr
    detected = imghdr.what(None, h=raw)
    if detected not in ALLOWED_MIME_TYPES:
        raise ValueError(f"Invalid file type. Allowed: jpg, png, webp, bmp")
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise ValueError("Corrupted or invalid image file.")
    clean = Image.new(img.mode, img.size)
    clean.putdata(list(img.getdata()))
    return clean.convert("RGB")

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key","")
        if not key:
            logger.warning(f"[AUTH] Missing key — IP:{request.remote_addr}")
            return jsonify({"success":False,"error":"API key required. Send X-API-Key header."}),401
        if key != API_KEY:
            logger.warning(f"[AUTH] Wrong key — IP:{request.remote_addr}")
            return jsonify({"success":False,"error":"Invalid API key."}),401
        return f(*args, **kwargs)
    return decorated

# ── Request logging ───────────────────────────
@app.before_request
def before():
    g.start = time.time()
    logger.info(f"[REQ] {request.method} {request.path} IP:{request.remote_addr}")

@app.after_request
def after(response):
    ms = round((time.time()-g.start)*1000,1)
    logger.info(f"[RES] {request.path} → {response.status_code} ({ms}ms)")
    return response

# ── Error handlers ────────────────────────────
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"success":False,"error":"Bad request."}),400

@app.errorhandler(401)
def unauthorized(e):
    return jsonify({"success":False,"error":"Unauthorized."}),401

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success":False,"error":f"Endpoint not found: {request.path}"}),404

@app.errorhandler(413)
def too_large(e):
    return jsonify({"success":False,"error":f"File too large. Max {MAX_UPLOAD_MB}MB."}),413

@app.errorhandler(429)
def rate_limited(e):
    return jsonify({"success":False,"error":f"Too many requests. Limit: {RATE_LIMIT}."}),429

@app.errorhandler(500)
def server_error(e):
    logger.error(f"[500] {e}")
    return jsonify({"success":False,"error":"Internal server error."}),500

# ── Detection logic ───────────────────────────
def cosine_sim(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))

def run_detection(img):
    if not database:
        return {"verdict":"UNKNOWN","confidence":"LOW","similarity_score":0.0,
                "matched_brand":None,"message":"Database empty. Add logos first."}
    q = extractor.extract_from_pil(img)
    best_brand,best_score = None,-1.0
    for brand,stored in database.items():
        s = cosine_sim(q,stored)
        if s>best_score:
            best_score,best_brand=s,brand
    best_score = round(best_score,4)
    if best_score>=HIGH_CONF_THRESHOLD:
        v,c,m="AUTHENTIC","HIGH",f"Very close match to '{best_brand}'."
    elif best_score>=SIMILARITY_THRESHOLD:
        v,c,m="AUTHENTIC","MEDIUM",f"Matches '{best_brand}' with moderate confidence."
    elif best_score>=LOW_CONF_THRESHOLD:
        v,c,m="FAKE","MEDIUM",f"Resembles '{best_brand}' but too low similarity."
    else:
        v,c,m="FAKE","HIGH","No match found in database."
    return {"verdict":v,"confidence":c,"similarity_score":best_score,
            "matched_brand":best_brand,"message":m,
            "timestamp":datetime.utcnow().isoformat()+"Z"}

# ── Routes ────────────────────────────────────
@app.route("/")
@limiter.exempt
def index():
    return jsonify({
        "name":"Fake Logo Detector API",
        "version":"3.0",
        "endpoints":{
            "POST /detect":"Detect logo (public)",
            "GET  /brands":"List brands (public)",
            "POST /add-brand":"Add brand (API key required)",
            "DELETE /brand/<name>":"Remove brand (API key required)",
            "GET  /health":"Health check"
        }
    })

@app.route("/health")
@limiter.exempt
def health():
    return jsonify({
        "status":"ok",
        "brands_loaded":len(database),
        "mode":"vercel-light" if IS_VERCEL else "clip-full",
        "timestamp":datetime.utcnow().isoformat()+"Z"
    })

@app.route("/detect", methods=["POST"])
@limiter.limit("10 per minute")
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"success":False,
                "error":"No image. Send as multipart/form-data with field 'image'."}),400
        raw = request.files["image"].read()
        img = validate_image(raw)
        result = run_detection(img)
        logger.info(f"[DETECT] {result['verdict']} score={result['similarity_score']} brand={result['matched_brand']}")
        return jsonify({"success":True,"result":result})
    except ValueError as e:
        return jsonify({"success":False,"error":str(e)}),400
    except Exception as e:
        logger.error(f"[DETECT] {e}")
        return jsonify({"success":False,"error":"Detection failed."}),500

@app.route("/brands")
@limiter.limit("30 per minute")
def brands():
    return jsonify({"success":True,"count":len(database),"brands":sorted(database.keys())})

@app.route("/add-brand", methods=["POST"])
@require_api_key
@limiter.limit("20 per hour")
def add_brand():
    try:
        name = request.form.get("brand_name","").strip().lower()
        if not name:
            return jsonify({"success":False,"error":"brand_name field required."}),400
        if len(name)>50:
            return jsonify({"success":False,"error":"brand_name too long (max 50 chars)."}),400
        if "image" not in request.files:
            return jsonify({"success":False,"error":"image field required."}),400
        raw = request.files["image"].read()
        img = validate_image(raw)
        database[name] = extractor.extract_from_pil(img)
        save_database()
        logger.info(f"[ADD] '{name}' added. Total: {len(database)}")
        return jsonify({"success":True,"message":f"Brand '{name}' added.","total_brands":len(database)}),201
    except ValueError as e:
        return jsonify({"success":False,"error":str(e)}),400
    except Exception as e:
        logger.error(f"[ADD] {e}")
        return jsonify({"success":False,"error":"Failed to add brand."}),500

@app.route("/brand/<brand_name>", methods=["DELETE"])
@require_api_key
@limiter.limit("20 per hour")
def remove_brand(brand_name):
    key = brand_name.lower().strip()
    if key not in database:
        return jsonify({"success":False,"error":f"Brand '{key}' not found."}),404
    del database[key]
    save_database()
    logger.info(f"[DEL] '{key}' removed. Remaining: {len(database)}")
    return jsonify({"success":True,"message":f"Brand '{key}' removed.","remaining":len(database)})

if __name__ == "__main__":
    logger.info("Starting local dev server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
