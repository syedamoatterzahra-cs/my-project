"""
Simple Inference Pipeline - Process unlabeled images without training
This script provides basic image processing and visualization
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os
import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    IMAGES_DIR, PROCESSED_DIR, VISUALIZATIONS_DIR, REPORTS_DIR,
    IMAGE_SIZE, create_directories, print_config
)
from utils import read_image, normalize_image, denormalize_image, display_images

def preprocess_image(img, target_size=512):
    """
    Preprocess image for inference
    
    Args:
        img: input image (H, W, C)
        target_size: target size for resizing
    
    Returns:
        preprocessed image
    """
    # Resize to standard size
    img_resized = cv2.resize(img, (target_size, target_size))
    
    # Normalize
    img_normalized = normalize_image(img_resized)
    
    return img_resized, img_normalized

def enhance_ultrasound_image(img):
    """
    Apply basic enhancement to ultrasound image
    
    Args:
        img: input ultrasound image (H, W, C) in range [0, 255]
    
    Returns:
        enhanced image
    """
    # Convert to grayscale for processing
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return enhanced_rgb

def create_basic_mask(img, method='otsu'):
    """
    Create a basic segmentation mask using traditional CV methods
    (Placeholder for actual deep learning segmentation)
    
    Args:
        img: input image
        method: thresholding method ('otsu' or 'adaptive')
    
    Returns:
        binary mask
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if method == 'otsu':
        # Otsu's thresholding
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Adaptive thresholding
        mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    
    # Apply morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def extract_roi_from_mask(img, mask):
    """
    Extract ROI from image using mask
    
    Args:
        img: original image
        mask: binary mask
    
    Returns:
        ROI image
    """
    # Apply mask
    roi = cv2.bitwise_and(img, img, mask=mask)
    
    # Find bounding box of mask
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Crop to bounding box with small padding
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        roi_cropped = roi[y1:y2, x1:x2]
        return roi_cropped
    
    return roi

def process_single_image(image_path, save_outputs=True):
    """
    Process a single image through the pipeline
    
    Args:
        image_path: path to image
        save_outputs: whether to save processed outputs
    
    Returns:
        dict with processing results
    """
    filename = Path(image_path).stem
    
    # Load image
    img_original = read_image(image_path, as_rgb=True)
    
    # Preprocess
    img_resized, img_normalized = preprocess_image(img_original, IMAGE_SIZE)
    
    # Enhance
    img_enhanced = enhance_ultrasound_image(img_resized)
    
    # Create basic mask (placeholder for deep learning segmentation)
    mask = create_basic_mask(img_enhanced)
    
    # Extract ROI
    roi = extract_roi_from_mask(img_resized, mask)
    
    # Create visualization
    if save_outputs:
        # Save processed image
        processed_path = os.path.join(PROCESSED_DIR, f"{filename}_processed.png")
        cv2.imwrite(processed_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        
        # Save enhanced
        enhanced_path = os.path.join(PROCESSED_DIR, f"{filename}_enhanced.png")
        cv2.imwrite(enhanced_path, cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2BGR))
        
        # Save mask
        mask_path = os.path.join(PROCESSED_DIR, f"{filename}_mask.png")
        cv2.imwrite(mask_path, mask)
        
        # Save ROI
        if roi is not None and roi.size > 0:
            roi_path = os.path.join(PROCESSED_DIR, f"{filename}_roi.png")
            cv2.imwrite(roi_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(img_original)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img_resized)
        axes[0, 1].set_title(f'Resized ({IMAGE_SIZE}x{IMAGE_SIZE})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(img_enhanced)
        axes[0, 2].set_title('Enhanced (CLAHE)')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('Segmentation Mask')
        axes[1, 0].axis('off')
        
        if roi is not None and roi.size > 0:
            axes[1, 1].imshow(roi)
            axes[1, 1].set_title('Extracted ROI')
            axes[1, 1].axis('off')
        
        # Overlay mask on original
        overlay = img_resized.copy()
        overlay[mask > 0] = overlay[mask > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
        axes[1, 2].imshow(overlay.astype(np.uint8))
        axes[1, 2].set_title('Mask Overlay')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Processing Pipeline: {filename}', fontsize=16)
        plt.tight_layout()
        
        viz_path = os.path.join(VISUALIZATIONS_DIR, f"{filename}_pipeline.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        "filename": filename,
        "original_shape": img_original.shape,
        "processed_shape": img_resized.shape,
        "mask_coverage": np.sum(mask > 0) / mask.size,
        "roi_shape": roi.shape if roi is not None else None
    }

def process_all_images(max_images=None):
    """
    Process all images in the dataset
    
    Args:
        max_images: maximum number of images to process (None for all)
    """
    print("\n" + "="*60)
    print("PROCESSING UNLABELED IMAGES")
    print("="*60 + "\n")
    
    # Get all image paths
    image_paths = sorted(Path(IMAGES_DIR).glob("*.png"))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"Found {len(image_paths)} images to process\n")
    
    # Process each image
    results = []
    for img_path in tqdm(image_paths, desc="Processing images"):
        result = process_single_image(img_path, save_outputs=True)
        results.append(result)
    
    # Generate summary report
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(results)}")
    
    avg_coverage = np.mean([r['mask_coverage'] for r in results])
    print(f"Average mask coverage: {avg_coverage*100:.1f}%")
    
    print(f"\nOutputs saved to:")
    print(f"  - Processed images: {PROCESSED_DIR}")
    print(f"  - Visualizations: {VISUALIZATIONS_DIR}")
    print("="*60 + "\n")
    
    return results

def main():
    """Main function"""
    
    # Create directories
    create_directories()
    
    # Print configuration
    print_config()
    
    # Process images
    # Start with first 5 images as a test
    print("\n[PROCESSING] Processing first 5 images as a test...\n")
    results = process_all_images(max_images=5)
    
    print("\n[OK] Test processing complete!")
    print(f"\nTo process all images, run:")
    print(f"  python simple_inference.py --all")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process ultrasound images')
    parser.add_argument('--all', action='store_true', help='Process all images')
    parser.add_argument('--max', type=int, default=5, help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    if args.all:
        create_directories()
        print_config()
        process_all_images(max_images=None)
    else:
        main()
