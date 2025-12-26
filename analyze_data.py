"""
Data Analysis Script - Analyze unlabeled dataset
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import IMAGES_DIR, VISUALIZATIONS_DIR, create_directories, print_config
from utils import analyze_dataset, create_grid_visualization, get_image_info, read_image

def main():
    """Main function to analyze the dataset"""
    
    print("\n" + "="*60)
    print("MOATTAR PROJECT - DATA ANALYSIS")
    print("="*60 + "\n")
    
    # Create directories
    create_directories()
    
    # Print configuration
    print_config()
    
    # Analyze dataset
    dataset_info = analyze_dataset(IMAGES_DIR)
    
    if dataset_info is None:
        print("❌ No images found. Please check the images directory.")
        return
    
    image_paths = dataset_info['image_paths']
    
    # Display first image as example
    print("\n[IMAGE] Loading sample image...")
    first_image = read_image(image_paths[0], as_rgb=True)
    first_info = get_image_info(image_paths[0])
    
    print(f"\nFirst image info:")
    print(f"  Filename: {first_info['filename']}")
    print(f"  Dimensions: {first_info['width']} x {first_info['height']}")
    print(f"  Channels: {first_info['channels']}")
    print(f"  File size: {first_info['size_kb']:.1f} KB")
    
    # Create visualizations
    print(f"\n[VIZ] Creating visualizations...")
    
    # Show first few images
    print("  - Creating sample images grid (first 10 images)...")
    sample_paths = [str(p) for p in image_paths[:10]]
    grid_path = os.path.join(VISUALIZATIONS_DIR, "sample_grid_10.png")
    create_grid_visualization(
        sample_paths, 
        grid_size=(2, 5), 
        img_size=(150, 150),
        save_path=grid_path
    )
    
    # Create full dataset grid
    print("  - Creating full dataset grid (all images)...")
    all_paths = [str(p) for p in image_paths]
    full_grid_path = os.path.join(VISUALIZATIONS_DIR, "all_images_grid.png")
    create_grid_visualization(
        all_paths,
        grid_size=(5, 10),
        img_size=(100, 100),
        save_path=full_grid_path
    )
    
    # Summary
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"[OK] Analyzed {len(image_paths)} images")

    print(f"  - {Path(grid_path).name}")
    print(f"  - {Path(full_grid_path).name}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
