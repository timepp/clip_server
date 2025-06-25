import torch
import faiss
import os
import json
import numpy as np
import logging
import time
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageSearchServer:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cpu"  # macOS x86_64, no MPS
        self.model_name = model_name
        self.model, self.processor = self.load_clip_model(model_name)
        self.embedding_dim = 512
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = []
        self.image_dir = "./db"
        self.metadata_file = self.image_dir + "/" + "clip_metadata.json"
        self.embeddings_file = self.image_dir + "/" + "clip_embeddings.npy"
        self.status = {"scanning": False, "image_count": 0, "last_scan_time": None}
        os.makedirs(self.image_dir, exist_ok=True)
        self.load_metadata_and_embeddings()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def load_clip_model(self, model_name):
        logger.info(f"Loading CLIP model: {model_name}")
        try:
            # Try to load from cache first (Docker environment)
            model = CLIPModel.from_pretrained(model_name, local_files_only=False).to(self.device)
            processor = CLIPProcessor.from_pretrained(model_name, local_files_only=False)
            logger.info("Model loaded successfully from cache/download")
        except Exception as e:
            logger.warning(f"Failed to load model with cache preference: {e}")
            logger.info("Downloading model from HuggingFace...")
            # Fallback to normal loading (will download if needed)
            model = CLIPModel.from_pretrained(model_name).to(self.device)
            processor = CLIPProcessor.from_pretrained(model_name)
            logger.info("Model downloaded and loaded successfully")
        return model, processor

    def get_image_embedding(self, image_data, image_type):
        try:
            image = Image.open(BytesIO(image_data)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
            embedding = embedding.cpu().numpy().flatten()
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
        embedding = embedding.cpu().numpy().flatten()
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def load_metadata_and_embeddings(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
            self.status["image_count"] = len(self.metadata)
        if os.path.exists(self.embeddings_file):
            embeddings = np.load(self.embeddings_file).astype(np.float32)
            if len(embeddings) == len(self.metadata):
                self.index.add(embeddings)
                logger.info(f"Loaded {len(embeddings)} embeddings from {self.embeddings_file}")
            else:
                logger.warning(f"Mismatch: {len(embeddings)} embeddings vs {len(self.metadata)} metadata entries")
                self.metadata = []
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.save_metadata()
        else:
            logger.info(f"No embeddings file found at {self.embeddings_file}")

    def save_metadata(self):
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=4)

    def save_embeddings(self, embeddings):
        np.save(self.embeddings_file, np.array(embeddings, dtype=np.float32))
        logger.info(f"Saved {len(embeddings)} embeddings to {self.embeddings_file}")

    def process_image(self, image_path, idx):
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            embedding = self.get_image_embedding(image_data, os.path.splitext(image_path)[1].lower())
            image_size = os.path.getsize(image_path)
            rel_path = os.path.relpath(image_path, self.image_dir)
            return idx, embedding, {"key": rel_path, "imageSize": image_size, "source": "db"}
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return idx, None, None

    def scan_directory(self):
        self.status["scanning"] = True
        self.status["last_scan_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info("Starting directory scan")
        valid_extensions = {".jpg", ".jpeg", ".png"}
        futures = []
        embeddings = list(np.load(self.embeddings_file).astype(np.float32) if os.path.exists(self.embeddings_file) else [])
        existing_keys = {item["key"] for item in self.metadata}

        # --- 新增：删除source为'db'但文件已不存在的条目 ---
        to_delete = []
        for i, meta in enumerate(self.metadata):
            if meta.get("source") == "db":
                file_path = os.path.join(self.image_dir, meta["key"])
                if not os.path.exists(file_path):
                    to_delete.append(i)
        # 倒序删除，避免索引错位
        for i in reversed(to_delete):
            logger.info(f"Removing metadata and embedding for missing file: {self.metadata[i]['key']}")
            del self.metadata[i]
            if i < len(embeddings):
                del embeddings[i]
        self.save_metadata()
        self.save_embeddings(embeddings)
        self.index.reset()
        if embeddings:
            self.index.add(np.array(embeddings, dtype=np.float32))
        self.status["image_count"] = len(self.metadata)

        # --- 继续原有扫描逻辑 ---
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    image_path = os.path.join(root, file)
                    rel_path = os.path.relpath(image_path, self.image_dir)
                    if rel_path not in existing_keys:
                        idx = len(self.metadata)
                        futures.append(self.executor.submit(self.process_image, image_path, idx))

        for future in tqdm(futures, desc="Processing images"):
            idx, embedding, meta = future.result()
            if embedding is not None:
                self.metadata.append(meta)
                embeddings.append(embedding)
                self.status["image_count"] = len(self.metadata)
                self.save_metadata()
                self.save_embeddings(embeddings)

        if embeddings:
            self.index.reset()
            self.index.add(np.array(embeddings, dtype=np.float32))

        self.status["scanning"] = False
        logger.info("Directory scan completed")

    def generate_sse_progress(self):
        # Placeholder for SSE progress (unchanged from original)
        yield f"data: {json.dumps({'progress': 0, 'status': 'starting'})}\n\n"
        time.sleep(1)
        yield f"data: {json.dumps({'progress': 100, 'status': 'completed'})}\n\n"

@app.route("/index.html", methods=["GET"])
def serve_index():
    try:
        return send_file("./index.html", mimetype='text/html')
    except FileNotFoundError:
        return jsonify({"error": "index.html not found"}), 404
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/db/<path:filename>", methods=["GET"])
def serve_db_file(filename):
    try:
        file_path = os.path.join("./db", filename)
        if not os.path.exists(file_path):
            return jsonify({"error": f"File {filename} not found"}), 404
        
        # Security check: ensure the file is within the db directory
        if not os.path.abspath(file_path).startswith(os.path.abspath("./db")):
            return jsonify({"error": "Access denied"}), 403
            
        return send_file(file_path)
    except Exception as e:
        logger.error(f"Error serving db file {filename}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/status", methods=["GET"])
def status():
    return jsonify(server.status), 200

@app.route("/scandir", methods=["GET"])
def scandir():
    if server.status["scanning"]:
        return jsonify({"error": "Scan already in progress"}), 400
    server.executor.submit(server.scan_directory)
    return Response(
        response=server.generate_sse_progress(),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.route("/add", methods=["POST"])
def add():
    if not request.data:
        return jsonify({"error": "No image data provided"}), 400
    image_data = request.data
    image_type = request.headers.get("Content-Type", "").split("/")[-1].lower()
    key = request.form.get("key")
    extra_metadata = request.form.get("metadata")

    if not key or not image_type in {"jpg", "jpeg", "png"}:
        return jsonify({"error": "Missing or invalid key or image_type"}), 400

    # 解析额外metadata
    extra = {}
    if extra_metadata:
        try:
            extra = json.loads(extra_metadata)
            if not isinstance(extra, dict):
                return jsonify({"error": "Invalid metadata format, must be a dict"}), 400
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid metadata JSON"}), 400

    try:
        embedding = server.get_image_embedding(image_data, image_type)
        embeddings = list(np.load(server.embeddings_file).astype(np.float32) if os.path.exists(server.embeddings_file) else [])
        idx = len(server.metadata)

        # 获取真实imageSize
        image_size = len(image_data)
        # 构造新的metadata
        new_metadata = {"key": key, "imageSize": image_size, "source": "db"}
        new_metadata.update(extra)

        # Check if key exists
        existing_idx = next((i for i, m in enumerate(server.metadata) if m["key"] == key), None)
        if existing_idx is not None:
            server.metadata[existing_idx] = new_metadata
            embeddings[existing_idx] = embedding
        else:
            server.metadata.append(new_metadata)
            embeddings.append(embedding)

        server.save_metadata()
        server.save_embeddings(embeddings)
        server.index.reset()
        server.index.add(np.array(embeddings, dtype=np.float32))
        server.status["image_count"] = len(server.metadata)
        logger.info(f"Added image with key: {key}")
        return jsonify({"message": f"Image with key {key} added", "id": idx}), 200
    except Exception as e:
        logger.error(f"Error adding image with key {key}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["GET"])
def query():
    text = request.args.get("text")
    top_k = int(request.args.get("top_k", 5))
    if not text:
        return jsonify({"error": "Missing text parameter"}), 400
    try:
        text_embedding = server.get_text_embedding(text).astype(np.float32)
        distances, indices = server.index.search(np.array([text_embedding]), top_k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(server.metadata):
                meta = server.metadata[idx]
                results.append({
                    "key": meta["key"],
                    "imageSize": meta["imageSize"],
                    "score": float(score * 100)
                })
        logger.info(f"Queried with text: {text}, found {len(results)} matches")
        return jsonify({"text": text, "results": results}), 200
    except Exception as e:
        logger.error(f"Error processing query for {text}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/similar", methods=["POST"])
def similar():
    # 新增：支持通过key直接查找
    key = request.form.get("key") or request.args.get("key")
    top_k = int(request.form.get("top_k", 5)) if request.form.get("top_k") else int(request.args.get("top_k", 5))

    if key:
        # 通过key查找embedding
        idx = next((i for i, m in enumerate(server.metadata) if m["key"] == key), None)
        if idx is None:
            return jsonify({"error": f"Key '{key}' not found"}), 404
        embeddings = np.load(server.embeddings_file).astype(np.float32)
        image_embedding = embeddings[idx].astype(np.float32)
        query_embedding = np.expand_dims(image_embedding, axis=0)
    else:
        if not request.data:
            return jsonify({"error": "No image data provided"}), 400
        image_data = request.data
        image_type = request.headers.get("Content-Type", "").split("/")[-1].lower()
        if not image_type in {"jpg", "jpeg", "png"}:
            return jsonify({"error": "Invalid image type. Supported: jpg, jpeg, png"}), 400
        image_embedding = server.get_image_embedding(image_data, image_type).astype(np.float32)
        query_embedding = np.array([image_embedding])

    try:
        distances, indices = server.index.search(query_embedding, top_k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(server.metadata):
                meta = server.metadata[idx]
                results.append({
                    "key": meta["key"],
                    "imageSize": meta["imageSize"],
                    "score": float(score * 100)
                })
        logger.info(f"Queried similar images with key={key if key else '[uploaded image]'}, found {len(results)} matches")
        return jsonify({"results": results}), 200
    except Exception as e:
        logger.error(f"Error processing similar query with key={key}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/delete", methods=["GET"])
def delete():
    key = request.args.get("key")
    if not key:
        return jsonify({"error": "Missing key parameter"}), 400
    try:
        idx = next((i for i, m in enumerate(server.metadata) if m["key"] == key), None)
        if idx is None:
            return jsonify({"error": "Key not found"}), 404
        embeddings = list(np.load(server.embeddings_file).astype(np.float32))
        del server.metadata[idx]
        del embeddings[idx]
        server.save_metadata()
        server.save_embeddings(embeddings)
        server.index.reset()
        if embeddings:
            server.index.add(np.array(embeddings, dtype=np.float32))
        server.status["image_count"] = len(server.metadata)
        logger.info(f"Deleted image with key: {key}")
        return jsonify({"message": f"Image with key {key} deleted"}), 200
    except Exception as e:
        logger.error(f"Error deleting image with key {key}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Image Search Server")
    server = ImageSearchServer()
    app.run(host="0.0.0.0", port=5000, debug=True)
