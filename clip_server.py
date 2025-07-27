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
import cv2

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

    def extract_video_frames(self, video_path, interval_sec=10, sim_threshold=0.95):
        """
        从视频每隔interval_sec抽取一帧，若与上一帧embedding相似度大于sim_threshold则跳过。
        返回：(key, embedding, metadata) 列表
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0
        results = []
        prev_emb = None
        prev_t = 0
        t = 0
        while t < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            # 转为JPEG编码
            _, buf = cv2.imencode('.jpg', frame)
            image_data = buf.tobytes()
            emb = self.get_image_embedding(image_data, 'jpg')
            if prev_emb is not None:
                sim = float(np.dot(emb, prev_emb) / (np.linalg.norm(emb) * np.linalg.norm(prev_emb) + 1e-8))
                logger.info(f"Similarity between {prev_t} - {t}: {sim:.4f}")
                if sim > sim_threshold:
                    t += interval_sec
                    continue
            prev_emb = emb
            prev_t = t
            # 生成key
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            key = f"{os.path.relpath(video_path, self.image_dir)}>{h:02}:{m:02}:{s:02}"
            meta = {
                "key": key,
                "imageSize": len(image_data),
                "source": "db",
                "video_file": os.path.relpath(video_path, self.image_dir),
                "timestamp": f"{h:02}:{m:02}:{s:02}"
            }
            results.append((key, emb, meta))
            t += interval_sec
        cap.release()
        return results

    def scan_directory(self, scan_videos=False):
        self.status["scanning"] = True
        self.status["last_scan_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info("Starting directory scan")
        valid_extensions = {".jpg", ".jpeg", ".png"}
        if scan_videos:
            valid_extensions.add(".mp4")
        futures = []
        embeddings = list(np.load(self.embeddings_file).astype(np.float32) if os.path.exists(self.embeddings_file) else [])
        existing_keys = {item["key"] for item in self.metadata}

        # 删除source为'db'但文件已不存在的条目
        to_delete = []
        for i, meta in enumerate(self.metadata):
            if meta.get("source") == "db":
                file_path = os.path.join(self.image_dir, meta["key"].split('>')[0])
                if not os.path.exists(file_path):
                    to_delete.append(i)
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

        # 新增：处理图片和（可选）视频
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                image_path = os.path.join(root, file)
                rel_path = os.path.relpath(image_path, self.image_dir)
                if ext in {".jpg", ".jpeg", ".png"}:
                    if rel_path not in existing_keys:
                        idx = len(self.metadata)
                        futures.append(self.executor.submit(self.process_image, image_path, idx))
                elif scan_videos and ext == ".mp4":
                    # 视频帧抽取
                    video_results = self.extract_video_frames(image_path, interval_sec=2, sim_threshold=0.87)
                    for key, emb, meta in video_results:
                        if key not in existing_keys:
                            self.metadata.append(meta)
                            embeddings.append(emb)
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
    scan_videos = request.args.get("scan_videos", "false").lower() == "true"
    if server.status["scanning"]:
        return jsonify({"error": "Scan already in progress"}), 400
    server.executor.submit(server.scan_directory, scan_videos)
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

@app.route("/files", methods=["GET"])
def list_files():
    """
    List files in the database with optional filtering by filename substring.
    Parameters:
    - filter: optional string to filter filenames (case-insensitive)
    - limit: optional limit on number of results (default: 100)
    - offset: optional offset for pagination (default: 0)
    """
    try:
        filter_str = request.args.get("filter", "").lower()
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
        
        # Validate parameters
        if limit < 1 or limit > 1000:
            return jsonify({"error": "Limit must be between 1 and 1000"}), 400
        if offset < 0:
            return jsonify({"error": "Offset must be non-negative"}), 400
        
        # Filter files based on the filter string
        filtered_files = []
        for meta in server.metadata:
            key = meta.get("key", "")
            if not filter_str or filter_str in key.lower():
                file_info = {
                    "key": key,
                    "imageSize": meta.get("imageSize", 0),
                    "source": meta.get("source", "unknown")
                }
                
                # Add additional metadata for video frames
                if "video_file" in meta:
                    file_info["video_file"] = meta["video_file"]
                    file_info["timestamp"] = meta.get("timestamp", "")
                    file_info["type"] = "video_frame"
                else:
                    file_info["type"] = "image"
                
                filtered_files.append(file_info)
        
        # Apply pagination
        total_count = len(filtered_files)
        paginated_files = filtered_files[offset:offset + limit]
        
        result = {
            "files": paginated_files,
            "total_count": total_count,
            "returned_count": len(paginated_files),
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < total_count
        }
        
        if filter_str:
            result["filter"] = filter_str
        
        logger.info(f"Listed files: filter='{filter_str}', total={total_count}, returned={len(paginated_files)}")
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/ve", methods=["GET"])
def ve_debug():
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    import math
    import io

    # Get parameters from request
    video_path = request.args.get("path")
    if not video_path:
        return jsonify({"error": "Missing 'path' parameter"}), 400
    
    # Security check: ensure the file is within allowed directories
    allowed_dirs = ["./db", "./pipe"]
    video_path_abs = os.path.abspath(video_path)
    if not any(video_path_abs.startswith(os.path.abspath(allowed_dir)) for allowed_dir in allowed_dirs):
        return jsonify({"error": "Access denied - path not in allowed directories"}), 403
    
    # Check if file exists and is an mp4
    if not os.path.exists(video_path):
        return jsonify({"error": f"Video file not found: {video_path}"}), 404
    
    if not video_path.lower().endswith('.mp4'):
        return jsonify({"error": "Only MP4 files are supported"}), 400

    interval = float(request.args.get("interval", 2))
    thumb_width = int(request.args.get("thumb_width", 200))
    thumbs_per_row = int(request.args.get("thumbs_per_row", 10))
    sim_threshold = float(request.args.get("sim_threshold", 0.87))
    
    # Generate output path based on input video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = "./pipe"
    out_path = os.path.join(out_dir, f"{video_name}_ve.jpg")
    font_path = None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": f"Cannot open video file: {video_path}"}), 400
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0

        if duration == 0:
            cap.release()
            return jsonify({"error": "Invalid video file or zero duration"}), 400

        thumbs = []
        times = []
        embs = []
        t = 0
        while t < duration:
            logger.info(f"Processing time {t:.2f}s")
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            w, h = pil_img.size
            scale = thumb_width / w
            thumb = pil_img.resize((thumb_width, int(h * scale)))
            thumbs.append(thumb)
            times.append(t)
            # embedding
            _, buf = cv2.imencode('.jpg', frame)
            image_data = buf.tobytes()
            emb = server.get_image_embedding(image_data, 'jpg')
            embs.append(emb)
            t += interval
        cap.release()

        if not thumbs:
            return jsonify({"error": "No frames extracted from video"}), 400

        # 突变帧检测，并记录每帧与上一个突变帧的相似度
        mutation_idx = [0]
        sim_to_last_mutation = [None]  # 第0帧无相似度
        for i in range(1, len(embs)):
            # 计算与所有突变帧的相似度
            sims = [float(np.dot(embs[i], embs[m]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[m]) + 1e-8)) for m in mutation_idx]
            sim_to_last_mutation.append(sims[-1])  # 记录与最近突变帧的相似度
            # 只有所有突变帧都不大于threshold才新增
            if all(sim <= sim_threshold for sim in sims):
                mutation_idx.append(i)

        # 字体
        try:
            font = ImageFont.truetype(font_path or "arial.ttf", 18)
            font_big = ImageFont.truetype(font_path or "arial.ttf", 26)
        except:
            font = ImageFont.load_default()
            font_big = ImageFont.load_default()

        thumbs_with_time = []
        for idx, (thumb, t) in enumerate(zip(thumbs, times)):
            w, h = thumb.size
            canvas = Image.new("RGB", (w, h + 48), (255, 255, 255))
            canvas.paste(thumb, (0, 0))
            draw = ImageDraw.Draw(canvas)
            time_str = "%02d:%02d:%02d" % (t // 3600, (t % 3600) // 60, t % 60)
            # 取与最近突变帧的相似度
            if idx == 0:
                sim_str = "(N/A)"
            else:
                sim = sim_to_last_mutation[idx]
                sim_str = f"({sim:.2f})"
            label = f"{time_str}{sim_str}"
            is_mutation = idx in mutation_idx
            # 画红框
            if is_mutation:
                draw.rectangle([0, 0, w-1, h-1], outline=(255, 0, 0), width=6)
            # 时间戳+相似度
            if is_mutation:
                bbox = draw.textbbox((0, 0), label, font=font_big)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(((w - text_w) // 2, h + 2), label, fill=(255, 0, 0), font=font_big)
            else:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(((w - text_w) // 2, h + 12), label, fill=(0, 0, 0), font=font)
            thumbs_with_time.append(canvas)

        # 拼接成矩阵
        rows = math.ceil(len(thumbs_with_time) / thumbs_per_row)
        thumb_h = thumbs_with_time[0].height if thumbs_with_time else 0
        grid = Image.new("RGB", (thumb_width * thumbs_per_row, thumb_h * rows), (255, 255, 255))
        for idx, thumb in enumerate(thumbs_with_time):
            x = (idx % thumbs_per_row) * thumb_width
            y = (idx // thumbs_per_row) * thumb_h
            grid.paste(thumb, (x, y))
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        grid.save(out_path)
        
        logger.info(f"Video extraction completed. Output saved to: {out_path}")
        # 直接返回图片
        return send_file(out_path, mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return jsonify({"error": f"Error processing video: {str(e)}"}), 500

if __name__ == "__main__":
    logger.info("Starting Image Search Server")
    server = ImageSearchServer()
    app.run(host="0.0.0.0", port=5000, debug=True)
