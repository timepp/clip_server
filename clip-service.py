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

def parse_top_k(raw_value, default=5):
    value = default if raw_value is None else int(raw_value)
    if value <= 0:
        raise ValueError("'top_k' must be > 0")
    return value

def embedding_to_base64(embedding: np.ndarray) -> str:
    return base64.b64encode(embedding.astype(np.float32).tobytes()).decode('ascii')

def base64_to_embedding(b64str: str) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(b64str), dtype=np.float32)
    return arr

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

def get_model_parameter_stats():
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total_params),
        "trainable": int(trainable_params),
        "frozen": int(total_params - trainable_params)
    }

def get_model_config_info():
    cfg = model.config
    vision_cfg = getattr(cfg, "vision_config", None)
    text_cfg = getattr(cfg, "text_config", None)
    return {
        "model_type": getattr(cfg, "model_type", None),
        "projection_dim": getattr(cfg, "projection_dim", None),
        "logit_scale_init_value": getattr(cfg, "logit_scale_init_value", None),
        "vision": {
            "hidden_size": getattr(vision_cfg, "hidden_size", None),
            "num_hidden_layers": getattr(vision_cfg, "num_hidden_layers", None),
            "num_attention_heads": getattr(vision_cfg, "num_attention_heads", None),
            "image_size": getattr(vision_cfg, "image_size", None),
            "patch_size": getattr(vision_cfg, "patch_size", None)
        },
        "text": {
            "hidden_size": getattr(text_cfg, "hidden_size", None),
            "num_hidden_layers": getattr(text_cfg, "num_hidden_layers", None),
            "num_attention_heads": getattr(text_cfg, "num_attention_heads", None),
            "vocab_size": getattr(text_cfg, "vocab_size", None),
            "max_position_embeddings": getattr(text_cfg, "max_position_embeddings", None)
        }
    }

def run_similarity_search(query_emb: np.ndarray, top_k: int):
    if not embeddings:
        return []
    top_k = max(1, min(top_k, len(embeddings)))
    sims = [float(np.dot(query_emb, emb)) for emb in embeddings]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [
        {
            "index": int(i),
            "score": float(sims[i]),
            "embedding": embedding_to_base64(embeddings[i])
        }
        for i in top_indices
    ]

def get_image_embedding(image_data):
    image = Image.open(BytesIO(image_data)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    embedding = embedding.cpu().numpy().flatten()
    return normalize_embedding(embedding)

def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    embedding = embedding.cpu().numpy().flatten()
    return normalize_embedding(embedding)

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

@app.route("/getImageEmbedding", methods=["POST"])
def get_image_embedding_api():
    if not request.data and not request.files.get('image'):
        return jsonify({"error": "No image data provided"}), 400
    if request.files.get('image'):
        image_data = request.files['image'].read()
    else:
        image_data = request.data
    try:
        embedding = get_image_embedding(image_data)
        emb_b64 = embedding_to_base64(embedding)
        return jsonify({"embedding": emb_b64}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/getTextEmbedding", methods=["POST"])
def get_text_embedding_api():
    if not request.json or "text" not in request.json:
        return jsonify({"error": "Missing 'text' in JSON body"}), 400
    text = request.json.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text' must be a non-empty string"}), 400
    try:
        embedding = get_text_embedding(text)
        emb_b64 = embedding_to_base64(embedding)
        return jsonify({"embedding": emb_b64}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/queryByText", methods=["GET"])
def query_by_text():
    text = request.args.get("text")
    if not text:
        return jsonify({"error": "Missing 'text' parameter"}), 400
    try:
        top_k = parse_top_k(request.args.get("top_k"), default=5)
        text_emb = get_text_embedding(text)
        results = run_similarity_search(text_emb, top_k)
        return jsonify({"results": results}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
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
    try:
        top_k_raw = request.form.get("top_k") if request.form.get("top_k") else request.args.get("top_k")
        top_k = parse_top_k(top_k_raw, default=5)
        query_emb = get_image_embedding(image_data)
        results = run_similarity_search(query_emb, top_k)
        return jsonify({"results": results}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/queryByEmbedding", methods=["POST"])
def query_by_embedding():
    if not request.json or "embedding" not in request.json:
        return jsonify({"error": "Missing 'embedding' in JSON body"}), 400

    raw_embedding = request.json["embedding"]

    try:
        top_k = parse_top_k(request.json.get("top_k", request.args.get("top_k")), default=5)
        if not isinstance(raw_embedding, str):
            return jsonify({"error": "'embedding' must be a base64 string"}), 400

        query_emb = base64_to_embedding(raw_embedding)

        if len(query_emb) != embedding_dim:
            return jsonify({"error": f"Invalid embedding dim: expected {embedding_dim}, got {len(query_emb)}"}), 400

        query_emb = normalize_embedding(query_emb)
        results = run_similarity_search(query_emb, top_k)
        return jsonify({"results": results}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/addEmbeddings", methods=["POST"])
def add_embeddings():
    if not request.json or "embeddings" not in request.json:
        return jsonify({"error": "No embeddings data provided"}), 400
    try:
        data = request.json["embeddings"]
        if not isinstance(data, list):
            return jsonify({"error": "'embeddings' must be a list of base64 strings"}), 400
        added = 0
        for emb_b64 in data:
            if not isinstance(emb_b64, str):
                continue
            arr = base64_to_embedding(emb_b64)
            if len(arr) == embedding_dim:
                embeddings.append(normalize_embedding(arr))
                added += 1
        return jsonify({"added": added, "count": len(embeddings)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/deleteEmbeddings", methods=["POST"])
def delete_embeddings():
    if not request.json:
        return jsonify({"error": "Missing JSON body"}), 400
    try:
        total = len(embeddings)
        start_index = int(request.json.get("startIndex", 0))
        end_index = int(request.json.get("endIndex", total))

        if start_index < 0 or end_index < 0:
            return jsonify({"error": "startIndex/endIndex must be >= 0"}), 400
        if start_index > end_index:
            return jsonify({"error": "startIndex must be <= endIndex"}), 400
        if end_index > total:
            return jsonify({"error": f"endIndex out of range: max {total}"}), 400

        deleted = end_index - start_index
        del embeddings[start_index:end_index]
        return jsonify({"deleted": deleted, "count": len(embeddings)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status", methods=["GET"])
def get_status():
    status = {
        "service": {
            "name": "clip_server",
            "endpoints": [
                "/addImage",
                "/getImageEmbedding",
                "/getTextEmbedding",
                "/queryByText",
                "/queryByImage",
                "/queryByEmbedding",
                "/addEmbeddings",
                "/deleteEmbeddings",
                "/status"
            ]
        },
        "runtime": {
            "device": device,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__
        },
        "embeddings": {
            "count": len(embeddings),
            "dimension": embedding_dim,
            "dtype": "float32",
            "normalized": True
        },
        "model": {
            "name": model_name,
            "class": model.__class__.__name__,
            "processor_class": processor.__class__.__name__,
            "parameters": get_model_parameter_stats(),
            "config": get_model_config_info()
        }
    }
    return jsonify(status), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)