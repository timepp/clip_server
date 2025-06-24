FROM python:3.12-slim

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
# Install PyTorch and dependencies
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# Set environment variables for model caching
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV HF_DATASETS_CACHE=/app/model_cache

WORKDIR /app

# Create model cache directory
RUN mkdir -p /app/model_cache

# Copy model download script
COPY download_model.py .

# Download the CLIP model during build time
RUN python download_model.py

# Copy application files
COPY clip_server.py .
COPY index.html .

# Create db directory for images
RUN mkdir -p /app/db

# Clean up download script (optional)
RUN rm download_model.py

EXPOSE 5000
CMD ["python", "clip_server.py"]
