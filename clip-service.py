

import torch
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import CLIPModel, CLIPProcessor
import base64

app = Flask(__name__)
CORS(app)

embeddings = []  # 每个元素为np.ndarray

device = "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
embedding_dim = 512

def embedding_to_base64(embedding: np.ndarray) -> str:
    return base64.b64encode(embedding.astype(np.float32).tobytes()).decode('ascii')

def base64_to_embedding(b64str: str) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(b64str), dtype=np.float32)
    return arr

def get_image_embedding(image_data):
    image = Image.open(BytesIO(image_data)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    embedding = embedding.cpu().numpy().flatten()
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    embedding = embedding.cpu().numpy().flatten()
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

@app.route("/addImage", methods=["POST"])
def add_image():
    if not request.data and not request.files.get('image'):
        return jsonify({"error": "No image data provided"}), 400
    if request.files.get('image'):
        image_data = request.files['image'].read()
    else:
        image_data = request.data
    try:
        embedding = get_image_embedding(image_data)
        embeddings.append(embedding)
        emb_b64 = embedding_to_base64(embedding)
        return jsonify({"index": len(embeddings) - 1, "embedding": emb_b64}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/queryByText", methods=["GET"])
def query_by_text():
    text = request.args.get("text")
    if not text:
        return jsonify({"error": "Missing 'text' parameter"}), 400
    top_k = int(request.args.get("top_k", 5))
    try:
        text_emb = get_text_embedding(text)
        if not embeddings:
            return jsonify({"results": []}), 200
        sims = [float(np.dot(text_emb, emb)) for emb in embeddings]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = [{"index": int(i), "score": float(sims[i])} for i in top_indices]
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/queryByImage", methods=["POST"])
def query_by_image():
    if not request.data and not request.files.get('image'):
        return jsonify({"error": "No image data provided"}), 400
    if request.files.get('image'):
        image_data = request.files['image'].read()
    else:
        image_data = request.data
    top_k = int(request.form.get("top_k", 5)) if request.form.get("top_k") else int(request.args.get("top_k", 5))
    try:
        query_emb = get_image_embedding(image_data)
        if not embeddings:
            return jsonify({"results": []}), 200
        sims = [float(np.dot(query_emb, emb)) for emb in embeddings]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = [{"index": int(i), "score": float(sims[i])} for i in top_indices]
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/loadEmbeddings", methods=["POST"])
def load_embeddings():
    if not request.json or "embeddings" not in request.json:
        return jsonify({"error": "No embeddings data provided"}), 400
    try:
        data = request.json["embeddings"]
        embeddings.clear()
        for emb_b64 in data:
            arr = base64_to_embedding(emb_b64)
            if len(arr) == embedding_dim:
                embeddings.append(arr)
        return jsonify({"count": len(embeddings)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete", methods=["GET"])
def delete_embedding():
    idx = request.args.get("index")
    if idx is None:
        return jsonify({"error": "Missing 'index' parameter"}), 400
    try:
        idx = int(idx)
        if idx < 0 or idx >= len(embeddings):
            return jsonify({"error": "Index out of range"}), 400
        embeddings.pop(idx)
        return jsonify({"message": f"Deleted embedding at index {idx}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"embeddings_count": len(embeddings)}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)