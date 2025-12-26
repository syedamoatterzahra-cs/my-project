"""
Configuration file for Moattar Project Pipeline - Inference Only
"""

import os
import torch

# Base paths
BASE_DIR = r"C:\Users\Dell\Desktop\FYP"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Output subdirectories
PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image settings
IMAGE_SIZE = 512  # Standard size for processing
CLASSIFIER_SIZE = 352  # Size for classification model input

# Model settings (for potential pre-trained model loading)
SEGMENTATION_MODEL = "unet"
SEGMENTATION_ENCODER = "resnet34"
CLASSIFICATION_MODEL = "efficientnetv2_s"

# Processing settings
BATCH_SIZE = 4
NUM_WORKERS = 2

# Normalization settings (ImageNet defaults)
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

# Risk scoring thresholds
RISK_THRESHOLDS = {
    "high": 0.85,
    "moderate": 0.60,
    "moderate_low": 0.40
}

def create_directories():
    """Create all necessary output directories"""
    dirs = [
        OUTPUT_DIR,
        MODELS_DIR,
        PROCESSED_DIR,
        VISUALIZATIONS_DIR,
        REPORTS_DIR
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f"[OK] Created output directories")

def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Images Directory: {IMAGES_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("=" * 60)

if __name__ == "__main__":
    create_directories()
    print_config()
