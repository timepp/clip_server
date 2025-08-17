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
        """
        从图片数据生成embedding
        返回: normalized embedding vector 或 None if failed
        """
        try:
            # 验证输入
            if not image_data or len(image_data) == 0:
                logger.error("Empty image data provided")
                return None
            
            # 打开并转换图片
            try:
                image = Image.open(BytesIO(image_data)).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to open/convert image: {e}")
                return None
            
            # 检查图片尺寸
            if image.size[0] == 0 or image.size[1] == 0:
                logger.error("Invalid image dimensions")
                return None
            
            # 处理图片
            try:
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            except Exception as e:
                logger.error(f"Failed to process image with CLIP processor: {e}")
                return None
            
            # 生成embedding
            try:
                with torch.no_grad():
                    embedding = self.model.get_image_features(**inputs)
            except Exception as e:
                logger.error(f"Failed to generate image features: {e}")
                return None
            
            # 转换为numpy并规范化
            try:
                embedding = embedding.cpu().numpy().flatten()
                
                # 检查embedding有效性
                if len(embedding) != self.embedding_dim:
                    logger.error(f"Embedding dimension mismatch: got {len(embedding)}, expected {self.embedding_dim}")
                    return None
                
                # 检查是否包含NaN或inf
                if not np.isfinite(embedding).all():
                    logger.error("Embedding contains NaN or inf values")
                    return None
                
                # 规范化
                norm = np.linalg.norm(embedding)
                if norm <= 1e-8:  # 避免除零
                    logger.error("Embedding norm too small for normalization")
                    return None
                
                return embedding / norm
                
            except Exception as e:
                logger.error(f"Failed to process embedding tensor: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            return None

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
        """
        处理单个图片文件，生成embedding和metadata
        返回: (idx, embedding, metadata) 或 (idx, None, None) 如果失败
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                return idx, None, None
            
            # 检查文件大小
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                logger.warning(f"Empty image file: {image_path}")
                return idx, None, None
            
            # 读取文件
            try:
                with open(image_path, "rb") as f:
                    image_data = f.read()
            except (IOError, OSError) as e:
                logger.error(f"Failed to read image file {image_path}: {e}")
                return idx, None, None
            
            # 检查读取的数据
            if not image_data:
                logger.warning(f"No data read from image file: {image_path}")
                return idx, None, None
            
            # 生成embedding
            try:
                embedding = self.get_image_embedding(image_data, os.path.splitext(image_path)[1].lower())
            except Exception as e:
                logger.error(f"Failed to generate embedding for {image_path}: {e}")
                return idx, None, None
            
            # 验证embedding
            if embedding is None or len(embedding) != self.embedding_dim:
                logger.error(f"Invalid embedding generated for {image_path}")
                return idx, None, None
            
            # 构建metadata
            rel_path = os.path.relpath(image_path, self.image_dir)
            metadata = {
                "key": rel_path, 
                "imageSize": file_size, 
                "source": "db"
            }
            
            return idx, embedding, metadata
            
        except Exception as e:
            logger.error(f"Unexpected error processing {image_path}: {e}")
            return idx, None, None

    def extract_video_frames(self, video_path, interval_sec=10, sim_threshold=0.95):
        """
        从视频每隔interval_sec抽取一帧，若与上一帧embedding相似度大于sim_threshold则跳过。
        返回：(key, embedding, metadata) 列表
        """
        results = []
        cap = None
        
        try:
            # 检查文件是否存在
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return results
            
            # 检查文件大小
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                logger.error(f"Empty video file: {video_path}")
                return results
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                return results
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0:
                logger.error(f"Invalid FPS ({fps}) for video: {video_path}")
                return results
            
            if total_frames <= 0:
                logger.error(f"Invalid frame count ({total_frames}) for video: {video_path}")
                return results
            
            duration = total_frames / fps
            logger.info(f"Processing video {video_path}: {duration:.1f}s, {fps:.1f}fps, {total_frames} frames")
            
            prev_emb = None
            prev_t = 0
            t = 0
            frame_count = 0
            error_count = 0
            
            while t < duration:
                try:
                    # 设置视频位置
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.warning(f"Failed to read frame at {t:.1f}s in {video_path}")
                        t += interval_sec
                        error_count += 1
                        if error_count > 10:  # 连续失败太多次就跳出
                            logger.error(f"Too many frame read errors in {video_path}, stopping extraction")
                            break
                        continue
                    
                    # 检查帧是否有效
                    if frame is None or frame.size == 0:
                        logger.warning(f"Invalid frame at {t:.1f}s in {video_path}")
                        t += interval_sec
                        continue
                    
                    # 转为JPEG编码
                    try:
                        _, buf = cv2.imencode('.jpg', frame)
                        if buf is None or len(buf) == 0:
                            logger.warning(f"Failed to encode frame at {t:.1f}s in {video_path}")
                            t += interval_sec
                            continue
                        image_data = buf.tobytes()
                    except Exception as e:
                        logger.error(f"Frame encoding error at {t:.1f}s in {video_path}: {e}")
                        t += interval_sec
                        continue
                    
                    # 生成embedding
                    try:
                        emb = self.get_image_embedding(image_data, 'jpg')
                        if emb is None or len(emb) != self.embedding_dim:
                            logger.warning(f"Invalid embedding at {t:.1f}s in {video_path}")
                            t += interval_sec
                            continue
                    except Exception as e:
                        logger.error(f"Embedding generation error at {t:.1f}s in {video_path}: {e}")
                        t += interval_sec
                        continue
                    
                    # 相似度检查
                    if prev_emb is not None:
                        try:
                            sim = float(np.dot(emb, prev_emb) / (np.linalg.norm(emb) * np.linalg.norm(prev_emb) + 1e-8))
                            logger.debug(f"Similarity between {prev_t:.1f}s - {t:.1f}s: {sim:.4f}")
                            if sim > sim_threshold:
                                t += interval_sec
                                continue
                        except Exception as e:
                            logger.warning(f"Similarity calculation error at {t:.1f}s in {video_path}: {e}")
                            # 继续处理，不跳过这一帧
                    
                    prev_emb = emb
                    prev_t = t
                    
                    # 生成key和metadata
                    try:
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
                        frame_count += 1
                        
                    except Exception as e:
                        logger.error(f"Metadata creation error at {t:.1f}s in {video_path}: {e}")
                    
                    t += interval_sec
                    
                except Exception as e:
                    logger.error(f"Frame processing error at {t:.1f}s in {video_path}: {e}")
                    t += interval_sec
                    error_count += 1
                    
                    if error_count > 20:  # 总错误太多就跳出
                        logger.error(f"Too many errors processing {video_path}, stopping extraction")
                        break
            
            logger.info(f"Extracted {frame_count} frames from {video_path} ({error_count} errors)")
            
        except Exception as e:
            logger.error(f"Critical error processing video {video_path}: {e}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except:
                    pass  # 忽略释放错误
        
        return results

    def scan_directory(self, scan_videos=False):
        self.status["scanning"] = True
        self.status["last_scan_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info("Starting directory scan")
        
        try:
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
            
            if to_delete:
                logger.info(f"Removing {len(to_delete)} entries for missing files")
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

            # 先扫描所有文件，统计总数量
            all_new_files = []
            all_new_videos = []
            
            for root, _, files in os.walk(self.image_dir):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    image_path = os.path.join(root, file)
                    rel_path = os.path.relpath(image_path, self.image_dir)
                    
                    if ext in {".jpg", ".jpeg", ".png"}:
                        if rel_path not in existing_keys:
                            all_new_files.append((image_path, rel_path))
                    elif scan_videos and ext == ".mp4":
                        all_new_videos.append((image_path, rel_path))

            total_new_files = len(all_new_files)
            total_new_videos = len(all_new_videos)
            
            logger.info(f"Found {total_new_files} new image files and {total_new_videos} new video files to process")
            
            # 存储扫描统计信息
            self.status["scan_stats"] = {
                "total_new_images": total_new_files,
                "total_new_videos": total_new_videos,
                "processed_images": 0,
                "processed_videos": 0,
                "failed_images": 0,
                "failed_videos": 0,
                "deleted_files": len(to_delete),  # 记录删除的文件数量
                "added_files": 0  # 记录实际添加的文件数量
            }

            processed_count = 0
            failed_count = 0

            # 处理图片文件
            for image_path, rel_path in all_new_files:
                try:
                    idx = len(self.metadata)
                    futures.append(self.executor.submit(self.process_image, image_path, idx))
                except Exception as e:
                    logger.error(f"Failed to submit processing task for {rel_path}: {e}")
                    failed_count += 1
                    self.status["scan_stats"]["failed_images"] += 1

            # 处理视频文件
            for image_path, rel_path in all_new_videos:
                try:
                    logger.info(f"Processing video: {rel_path}")
                    video_results = self.extract_video_frames(image_path, interval_sec=2, sim_threshold=0.87)
                    
                    video_frame_count = 0
                    for key, emb, meta in video_results:
                        if key not in existing_keys:
                            try:
                                self.metadata.append(meta)
                                embeddings.append(emb)
                                self.status["image_count"] = len(self.metadata)
                                processed_count += 1
                                video_frame_count += 1
                                self.status["scan_stats"]["added_files"] += 1  # 记录添加的文件
                                
                                # Save every 5 frames to avoid data loss
                                if processed_count % 5 == 0:
                                    self.save_metadata()
                                    self.save_embeddings(embeddings)
                            except Exception as e:
                                logger.error(f"Failed to add video frame {key}: {e}")
                                failed_count += 1
                    
                    logger.info(f"Extracted {video_frame_count} frames from {rel_path}")
                    self.status["scan_stats"]["processed_videos"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process video {rel_path}: {e}")
                    failed_count += 1
                    self.status["scan_stats"]["failed_videos"] += 1

            # 处理图片任务的结果
            successful_images = 0
            for i, future in enumerate(tqdm(futures, desc="Processing images")):
                try:
                    idx, embedding, meta = future.result(timeout=30)  # 30秒超时
                    if embedding is not None and meta is not None:
                        self.metadata.append(meta)
                        embeddings.append(embedding)
                        self.status["image_count"] = len(self.metadata)
                        processed_count += 1
                        successful_images += 1
                        self.status["scan_stats"]["processed_images"] += 1
                        self.status["scan_stats"]["added_files"] += 1  # 记录添加的文件
                        
                        # Save periodically during processing
                        if processed_count % 10 == 0:
                            self.save_metadata()
                            self.save_embeddings(embeddings)
                    else:
                        logger.warning(f"Image processing returned null result for task {i}")
                        failed_count += 1
                        self.status["scan_stats"]["failed_images"] += 1
                        
                except Exception as e:
                    logger.error(f"Image processing task {i} failed: {e}")
                    failed_count += 1
                    self.status["scan_stats"]["failed_images"] += 1

            # Final save
            try:
                if embeddings:
                    self.index.reset()
                    self.index.add(np.array(embeddings, dtype=np.float32))
                    self.save_metadata()
                    self.save_embeddings(embeddings)
            except Exception as e:
                logger.error(f"Failed to save final results: {e}")
            
            # 统计报告
            total_attempted = total_new_files + sum(len(self.extract_video_frames(path, interval_sec=2, sim_threshold=0.87)) for path, _ in all_new_videos)
            success_rate = (processed_count / total_attempted * 100) if total_attempted > 0 else 100
            
            # 获取统计数据
            added_files = self.status["scan_stats"]["added_files"] if "scan_stats" in self.status else processed_count
            deleted_files = self.status["scan_stats"]["deleted_files"] if "scan_stats" in self.status else len(to_delete) if 'to_delete' in locals() else 0
            
            logger.info(f"Directory scan completed:")
            logger.info(f"  - Added {added_files} new items")
            logger.info(f"  - Deleted {deleted_files} missing items")
            logger.info(f"  - Failed to process {failed_count} items")
            logger.info(f"  - Success rate: {success_rate:.1f}%")
            logger.info(f"  - Total items in database: {len(self.metadata)}")
            
        except Exception as e:
            logger.error(f"Critical error during directory scan: {e}")
            raise
        finally:
            self.status["scanning"] = False
            # 清理扫描统计信息
            if "scan_stats" in self.status:
                del self.status["scan_stats"]

    def generate_sse_progress(self):
        """Generate real-time SSE progress updates for directory scanning"""
        yield f"data: {json.dumps({'progress': 0, 'status': 'starting'})}\n\n"
        
        # Wait for scan to actually start
        initial_wait = 0
        while not self.status["scanning"] and initial_wait < 5:
            time.sleep(0.1)
            initial_wait += 0.1
        
        if not self.status["scanning"]:
            yield f"data: {json.dumps({'progress': 100, 'status': 'failed to start'})}\n\n"
            return
        
        yield f"data: {json.dumps({'progress': 5, 'status': 'scanning directory structure'})}\n\n"
        
        # Wait for scan stats to be available
        stats_wait = 0
        while self.status["scanning"] and "scan_stats" not in self.status and stats_wait < 10:
            time.sleep(0.2)
            stats_wait += 0.2
        
        if "scan_stats" in self.status:
            stats = self.status["scan_stats"]
            total_images = stats["total_new_images"]
            total_videos = stats["total_new_videos"]
            deleted_files = stats["deleted_files"]
            
            status_parts = []
            if total_images > 0 or total_videos > 0:
                status_parts.append(f"found {total_images} images, {total_videos} videos to process")
            if deleted_files > 0:
                status_parts.append(f"removed {deleted_files} missing files")
            
            if status_parts:
                yield f"data: {json.dumps({'progress': 10, 'status': '; '.join(status_parts)})}\n\n"
            else:
                yield f"data: {json.dumps({'progress': 100, 'status': 'no new files to process'})}\n\n"
                return
        
        # Monitor scanning progress
        last_processed_images = 0
        last_processed_videos = 0
        scan_duration = 0
        
        while self.status["scanning"]:
            time.sleep(0.8)
            scan_duration += 0.8
            
            if "scan_stats" in self.status:
                stats = self.status["scan_stats"]
                processed_images = stats["processed_images"]
                processed_videos = stats["processed_videos"]
                failed_images = stats["failed_images"]
                failed_videos = stats["failed_videos"]
                total_images = stats["total_new_images"]
                total_videos = stats["total_new_videos"]
                added_files = stats["added_files"]
                deleted_files = stats["deleted_files"]
                
                total_items = total_images + total_videos
                processed_items = processed_images + processed_videos
                failed_items = failed_images + failed_videos
                
                if total_items > 0:
                    # Calculate progress based on actual completion
                    progress = min(95, 10 + (processed_items + failed_items) / total_items * 85)
                    
                    # Create detailed status message
                    status_parts = []
                    if total_images > 0:
                        status_parts.append(f"images: {processed_images}/{total_images}")
                    if total_videos > 0:
                        status_parts.append(f"videos: {processed_videos}/{total_videos}")
                    if added_files > 0:
                        status_parts.append(f"added: {added_files}")
                    if deleted_files > 0:
                        status_parts.append(f"deleted: {deleted_files}")
                    if failed_items > 0:
                        status_parts.append(f"failed: {failed_items}")
                    
                    status_msg = f"processing {', '.join(status_parts)}"
                    
                    yield f"data: {json.dumps({'progress': int(progress), 'status': status_msg})}\n\n"
                    
                    last_processed_images = processed_images
                    last_processed_videos = processed_videos
                else:
                    # Fallback to time-based progress
                    progress = min(80, 10 + (scan_duration * 1))
                    yield f"data: {json.dumps({'progress': int(progress), 'status': 'processing files'})}\n\n"
            else:
                # No stats available, use time-based progress
                progress = min(80, 10 + (scan_duration * 1))
                yield f"data: {json.dumps({'progress': int(progress), 'status': 'processing files'})}\n\n"
            
            # Safety timeout after 120 seconds
            if scan_duration > 120:
                yield f"data: {json.dumps({'progress': 95, 'status': 'scan taking longer than expected'})}\n\n"
                break
        
        # Final status
        final_count = self.status["image_count"]
        if "scan_stats" in self.status:
            stats = self.status["scan_stats"]
            added_files = stats["added_files"]
            deleted_files = stats["deleted_files"]
            failed_total = stats["failed_images"] + stats["failed_videos"]
            
            # 构建完成消息
            status_parts = []
            if added_files > 0:
                status_parts.append(f"added {added_files}")
            if deleted_files > 0:
                status_parts.append(f"deleted {deleted_files}")
            if failed_total > 0:
                status_parts.append(f"failed {failed_total}")
            
            if status_parts:
                change_summary = ", ".join(status_parts)
                yield f"data: {json.dumps({'progress': 100, 'status': f'completed - {change_summary}, total: {final_count} images'})}\n\n"
            else:
                yield f"data: {json.dumps({'progress': 100, 'status': f'completed - no changes, total: {final_count} images'})}\n\n"
        else:
            yield f"data: {json.dumps({'progress': 100, 'status': f'completed - {final_count} images total'})}\n\n"

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
