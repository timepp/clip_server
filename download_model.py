#!/usr/bin/env python3
"""
Pre-download CLIP model for Docker container
This script downloads the CLIP model during Docker build time
to avoid downloading it on every container startup.
"""

import os
import logging
from transformers import CLIPModel, CLIPProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_clip_model(model_name="openai/clip-vit-base-patch32"):
    """Download CLIP model and processor"""
    try:
        logger.info(f"Starting download of CLIP model: {model_name}")
        
        # Download model
        logger.info("Downloading CLIP model...")
        model = CLIPModel.from_pretrained(model_name)
        logger.info("✅ CLIP model downloaded successfully")
        
        # Download processor
        logger.info("Downloading CLIP processor...")
        processor = CLIPProcessor.from_pretrained(model_name)
        logger.info("✅ CLIP processor downloaded successfully")
        
        # Test loading to ensure everything works
        logger.info("Testing model loading...")
        device = "cpu"
        model = model.to(device)
        logger.info("✅ Model test successful")
        
        logger.info("🎉 All CLIP components downloaded and verified successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading CLIP model: {e}")
        return False

if __name__ == "__main__":
    success = download_clip_model()
    if not success:
        exit(1)
    
    logger.info("Model download completed. Ready for production use!")
