# ╔══════════════════════════════════════════════════════════════════╗
#  app.py — Fake Logo Detector
#  Hugging Face Spaces (Gradio) + Full CLIP ViT-B/32
#  100% free, accurate, no size limits
# ╚══════════════════════════════════════════════════════════════════╝

import os
import io
import pickle
import numpy as np
from PIL import Image
import gradio as gr
import torch
import open_clip

# ── Config ────────────────────────────────────
EMBEDDINGS_FILE      = "logo_embeddings_clip.pkl"
CLIP_MODEL_NAME      = "ViT-B-32"
CLIP_PRETRAINED      = "openai"
SIMILARITY_THRESHOLD = 0.80
HIGH_CONF_THRESHOLD  = 0.88
LOW_CONF_THRESHOLD   = 0.60

# ── Load CLIP ─────────────────────────────────
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
)
model.eval().to(device)
print(f"CLIP ready on {device}")

# ── Load database ─────────────────────────────
def load_db():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            db = pickle.load(f)
        print(f"Database loaded — {len(db)} brands")
        return db
    print("No database found")
    return {}

database = load_db()

# ── Extract embedding ──────────────────────────
def extract(img: Image.Image) -> np.ndarray:
    tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
    vec = features.squeeze().float().cpu().numpy()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# ── Cosine similarity ──────────────────────────
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ── Detection logic ───────────────────────────
def detect(img: Image.Image):
    if not database:
        return (
            "❓ UNKNOWN",
            "Database is empty — no brands loaded.",
            0.0,
            "None",
            "LOW"
        )

    query = extract(img)
    best_brand, best_score = None, -1.0

    for brand, stored in database.items():
        s = cosine_sim(query, stored)
        if s > best_score:
            best_score, best_brand = s, brand

    best_score = round(best_score, 4)
    pct        = round(best_score * 100, 1)
    brand_name = best_brand.capitalize() if best_brand else "None"

    if best_score >= HIGH_CONF_THRESHOLD:
        verdict = "✅ AUTHENTIC"
        conf    = "HIGH"
        msg     = f"Very close CLIP match to '{brand_name}'. This logo is genuine."
    elif best_score >= SIMILARITY_THRESHOLD:
        verdict = "✅ AUTHENTIC"
        conf    = "MEDIUM"
        msg     = f"Matches '{brand_name}' with moderate confidence."
    elif best_score >= LOW_CONF_THRESHOLD:
        verdict = "❌ FAKE"
        conf    = "MEDIUM"
        msg     = f"Resembles '{brand_name}' but similarity is too low. Likely counterfeit."
    else:
        verdict = "❌ FAKE"
        conf    = "HIGH"
        msg     = f"No match found. Score {pct}% is below threshold. Definitely fake."

    return verdict, msg, pct, brand_name, conf

# ── Gradio UI ─────────────────────────────────
def run_detection(image):
    if image is None:
        return "Please upload a logo image.", "", "", "", ""

    verdict, msg, score, brand, conf = detect(image)

    score_display = f"{score}%"
    brands_in_db  = f"{len(database)} brands loaded"

    return verdict, msg, score_display, brand, brands_in_db

# ── Brand list ────────────────────────────────
brand_list = ", ".join(
    b.capitalize() for b in sorted(database.keys())
) if database else "No brands loaded"

# ── Build Gradio interface ─────────────────────
with gr.Blocks(
    title="Fake Logo Detector",
    theme=gr.themes.Base(
        primary_hue="orange",
        secondary_hue="red",
        neutral_hue="gray",
        font=gr.themes.GoogleFont("Nunito"),
    ),
    css="""
    .gradio-container{max-width:680px!important;margin:0 auto}
    #header{text-align:center;padding:20px 0 10px}
    #header h1{font-size:32px;font-weight:800;color:#ff6b00}
    #header p{color:#aaa;font-size:14px;margin-top:4px}
    .verdict-box textarea{font-size:28px!important;font-weight:800!important;text-align:center!important}
    """
) as demo:

    gr.HTML("""
    <div id="header">
      <h1>🐦 Fake Logo Detector</h1>
      <p>Upload any brand logo — CLIP AI will tell you if it is authentic or fake</p>
      <p style="color:#ff6b00;font-weight:700;font-size:13px">
        Powered by OpenAI CLIP ViT-B/32 &nbsp;|&nbsp; Full AI accuracy
      </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Logo Image",
                height=280,
            )
            detect_btn = gr.Button(
                "🎯 Detect Logo",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=1):
            verdict_out = gr.Textbox(
                label="Verdict",
                interactive=False,
                elem_classes=["verdict-box"],
            )
            message_out = gr.Textbox(
                label="Details",
                interactive=False,
                lines=3,
            )
            with gr.Row():
                score_out = gr.Textbox(
                    label="Similarity score",
                    interactive=False,
                )
                brand_out = gr.Textbox(
                    label="Matched brand",
                    interactive=False,
                )
            db_out = gr.Textbox(
                label="Database",
                value=f"{len(database)} brands loaded",
                interactive=False,
            )

    detect_btn.click(
        fn=run_detection,
        inputs=[image_input],
        outputs=[verdict_out, message_out, score_out, brand_out, db_out],
    )

    gr.HTML(f"""
    <div style="margin-top:20px;padding:16px;background:rgba(255,107,0,0.08);
         border-radius:12px;border:1px solid rgba(255,107,0,0.2)">
      <p style="font-size:12px;color:#888;font-weight:700;margin-bottom:8px">
        SUPPORTED BRANDS ({len(database)})
      </p>
      <p style="font-size:12px;color:#aaa;line-height:1.8">{brand_list}</p>
    </div>
    <div style="text-align:center;margin-top:16px;font-size:12px;color:#555">
      Built by <strong style="color:#ff6b00">Gunasekar</strong> &nbsp;|&nbsp;
      CLIP ViT-B/32 &nbsp;|&nbsp;
      <a href="https://github.com/gunasekar05042005-dot/Fake-Logo-Detector"
         style="color:#ff6b00" target="_blank">GitHub ↗</a>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
