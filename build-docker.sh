#!/bin/bash

# CLIP Server Docker Build Script
set -e

IMAGE_NAME="clip-server"
TAG="latest"

echo "🐳 Building CLIP Server Docker Image..."
echo "📦 Image: ${IMAGE_NAME}:${TAG}"
echo ""

# Build the Docker image
echo "🔨 Starting Docker build..."
docker build -t ${IMAGE_NAME}:${TAG} .

echo ""
echo "✅ Docker image built successfully!"
echo ""
echo "🚀 To run the container:"
echo "   docker run -p 5000:5000 -v \$(pwd)/db:/app/db ${IMAGE_NAME}:${TAG}"
echo ""
echo "🔗 Access the web interface at: http://localhost:5000/index.html"
echo ""
echo "📝 Notes:"
echo "   - The CLIP model is pre-downloaded in the image"
echo "   - Mount ./db directory to persist images and embeddings"
echo "   - The container will start immediately without model download delay"
