#!/bin/bash

# CLIP Server Docker Run Script
set -e

IMAGE_NAME="clip-server"
TAG="latest"
CONTAINER_NAME="clip-server-instance"
HOST_PORT="5000"
CONTAINER_PORT="5000"

echo "🚀 Starting CLIP Server Docker Container..."
echo ""

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🛑 Stopping existing container..."
    docker stop ${CONTAINER_NAME} || true
    docker rm ${CONTAINER_NAME} || true
fi

# Create db directory if it doesn't exist
mkdir -p ./db

echo "🐳 Starting new container..."
docker run -d \
    --name ${CONTAINER_NAME} \
    -p ${HOST_PORT}:${CONTAINER_PORT} \
    -v "$(pwd)/db:/app/db" \
    ${IMAGE_NAME}:${TAG}

echo ""
echo "✅ Container started successfully!"
echo ""
echo "📋 Container Info:"
echo "   Name: ${CONTAINER_NAME}"
echo "   Port: ${HOST_PORT} -> ${CONTAINER_PORT}"
echo "   Volume: ./db -> /app/db"
echo ""
echo "🔗 Access the application:"
echo "   Web Interface: http://localhost:${HOST_PORT}/index.html"
echo "   API Status: http://localhost:${HOST_PORT}/status"
echo ""
echo "📊 Container logs:"
echo "   docker logs -f ${CONTAINER_NAME}"
echo ""
echo "🛑 To stop the container:"
echo "   docker stop ${CONTAINER_NAME}"
