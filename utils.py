"""
Utility functions for image loading, preprocessing, and visualization
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - saves to files instead of showing
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def read_image(image_path, as_rgb=True, target_size=None):
    """
    Read an image from path
    
    Args:
        image_path: Path to image file
        as_rgb: Convert to RGB if True
        target_size: Resize to (width, height) if specified
    
    Returns:
        numpy array of image
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    if as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if target_size is not None:
        img = cv2.resize(img, target_size)
    
    return img

def normalize_image(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Normalize image with mean and std
    
    Args:
        img: numpy array (H, W, C) in range [0, 255]
        mean: tuple of means for each channel
        std: tuple of stds for each channel
    
    Returns:
        normalized image as float32
    """
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    return img

def denormalize_image(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Denormalize image back to [0, 1] range
    
    Args:
        img: normalized image
        mean: tuple of means used for normalization
        std: tuple of stds used for normalization
    
    Returns:
        denormalized image in [0, 1] range
    """
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    return img

def display_images(images, titles=None, figsize=(15, 5), save_path=None):
    """
    Display multiple images in a row
    
    Args:
        images: list of images to display
        titles: list of titles for each image
        figsize: figure size
        save_path: path to save figure (optional)
    """
    n_images = len(images)
    if titles is None:
        titles = [f"Image {i+1}" for i in range(n_images)]
    
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        axes[idx].imshow(img)
        axes[idx].set_title(title)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved visualization to {save_path}")
        plt.close()
    else:
        plt.close()

def create_grid_visualization(image_paths, grid_size=(5, 10), img_size=(100, 100), save_path=None):
    """
    Create a grid visualization of multiple images
    
    Args:
        image_paths: list of image paths
        grid_size: (rows, cols) for grid
        img_size: size to resize each image
        save_path: path to save the grid
    
    Returns:
        grid image as numpy array
    """
    rows, cols = grid_size
    n_images = min(len(image_paths), rows * cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()
    
    for idx in range(rows * cols):
        if idx < n_images:
            img = read_image(image_paths[idx], as_rgb=True, target_size=img_size)
            axes[idx].imshow(img)
            axes[idx].set_title(Path(image_paths[idx]).stem, fontsize=8)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved grid visualization to {save_path}")
    
    plt.close()

def get_image_info(image_path):
    """
    Get basic information about an image
    
    Args:
        image_path: path to image
    
    Returns:
        dict with image info
    """
    img = read_image(image_path, as_rgb=False)
    return {
        "filename": Path(image_path).name,
        "shape": img.shape,
        "dtype": img.dtype,
        "size_kb": Path(image_path).stat().st_size / 1024,
        "height": img.shape[0],
        "width": img.shape[1],
        "channels": img.shape[2] if len(img.shape) > 2 else 1
    }

def analyze_dataset(images_dir):
    """
    Analyze all images in a directory
    
    Args:
        images_dir: directory containing images
    
    Returns:
        dict with dataset statistics
    """
    image_paths = sorted(Path(images_dir).glob("*.png"))
    
    if len(image_paths) == 0:
        print(f"No PNG images found in {images_dir}")
        return None
    
    print(f"\n{'='*60}")
    print(f"DATASET ANALYSIS")
    print(f"{'='*60}")
    print(f"Total images: {len(image_paths)}")
    
    # Analyze first few images
    shapes = []
    sizes = []
    
    for img_path in image_paths[:10]:  # Sample first 10 images
        info = get_image_info(img_path)
        shapes.append(info['shape'])
        sizes.append(info['size_kb'])
    
    print(f"Image dimensions (sample of 10):")
    unique_shapes = list(set([str(s) for s in shapes]))
    for shape in unique_shapes:
        print(f"  - {shape}")
    
    print(f"File size range: {min(sizes):.1f} KB - {max(sizes):.1f} KB")
    print(f"Average size: {np.mean(sizes):.1f} KB")
    print(f"{'='*60}\n")
    
    return {
        "total_images": len(image_paths),
        "image_paths": image_paths,
        "shapes": shapes,
        "sizes": sizes
    }

if __name__ == "__main__":
    # Test utilities
    print("Utility functions loaded successfully!")
