import os, torch, clip, numpy as np
from PIL import Image
import pickle
from tqdm import tqdm

DATASET = "logo_dataset"
OUTPUT = "logo_db.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

db = {"brand_names": [], "embeddings": []}

for brand in sorted(os.listdir(DATASET)):
    path = os.path.join(DATASET, brand)
    if not os.path.isdir(path): continue
    
    vecs = []
    for img in os.listdir(path):
        if not img.lower().endswith(('.png','.jpg','.jpeg','.webp')): continue
        try:
            image = preprocess(Image.open(os.path.join(path,img)).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                v = model.encode_image(image)
                vecs.append((v / v.norm(dim=-1, keepdim=True)).cpu().numpy().flatten())
        except: pass
    
    if vecs:
        avg = np.mean(vecs, axis=0)
        avg = avg / np.linalg.norm(avg)
        db["brand_names"].append(brand)
        db["embeddings"].append(avg)
        print(f"✓ {brand}: {len(vecs)} images")

db["embeddings"] = np.array(db["embeddings"])
with open(OUTPUT, "wb") as f:
    pickle.dump(db, f)

print(f"\n✅ DONE: {OUTPUT} | {len(db['brand_names'])} brands | shape {db['embeddings'].shape}")
