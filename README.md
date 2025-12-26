# Moattar Project Pipeline - Placental Ultrasound Analysis

*Inference-only pipeline for processing unlabeled ultrasound images (no training required)*

## Quick Start

### 1. Analyze Dataset
```bash
python analyze_data.py
```
This will:
- Analyze all 50 images in the `images` folder
- Create visualization grids showing all images
- Save results to `outputs/visualizations/`

### 2. Process Images (Test Run - First 5 Images)
```bash
python simple_inference.py
```
This will process the first 5 images and generate:
- Resized images (512x512)
- Enhanced images (CLAHE)
- Segmentation masks
- ROI extractions
- Complete pipeline visualizations

### 3. Process All Images
```bash
python simple_inference.py --all
```

### 4. Process Specific Number
```bash
python simple_inference.py --max 10
```

## What This Pipeline Does

The pipeline implements the **Moattar Project** workflow from the Word document:

1. **Image Loading** - Loads ultrasound images from `images/` folder
2. **Preprocessing** - Resizes to standard 512x512 size
3. **Enhancement** - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
4. **Segmentation** - Uses Otsu thresholding + morphological operations
5. **ROI Extraction** - Extracts region of interest from segmentation mask
6. **Visualization** - Creates 6-panel view showing entire pipeline

## Project Structure

```
FYP/
├── images/                    # INPUT: Your 50 unlabeled images
├── outputs/
│   ├── processed/            # Processed versions of each image
│   └── visualizations/       # Pipeline visualizations
├── config.py                 # Configuration settings
├── utils.py                  # Utility functions
├── analyze_data.py           # Dataset analysis script
├── simple_inference.py       # Main processing pipeline
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Output Files

For each processed image, the pipeline generates:
- `{name}_processed.png` - Resized 512x512
- `{name}_enhanced.png` - CLAHE enhanced
- `{name}_mask.png` - Segmentation mask
- `{name}_roi.png` - Extracted ROI
- `{name}_pipeline.png` - 6-panel visualization

## Requirements

All required packages are listed in `requirements.txt`:
- numpy
- opencv-python
- matplotlib
- Pillow
- tqdm

Install with: `pip install -r requirements.txt`

## Dataset Information

- **Total Images**: 50 PNG files (001.png to 050.png)
- **Format**: RGB ultrasound images
- **Dimensions**: Variable (392-959px wide)
- **Average Size**: 178.8 KB per file

## Technical Notes

- Uses **traditional computer vision methods** (no deep learning)
- **Non-interactive mode** - All visualizations saved to disk
- **Windows compatible** - Fixed Unicode encoding issues
- **CPU-based** - No GPU required

## Future Extensions

The Word document contains additional steps for:
- Deep learning segmentation (U-Net)
- Classification (EfficientNetV2)
- Risk scoring with fuzzy logic
- Grad-CAM++ interpretability
- Federated learning (optional)

These can be added later if pre-trained models become available.

## Status

[OK] Dataset analyzed (50 images)
[OK] Processing pipeline working
[OK] Test run completed (5 images)
[OK] Ready for full processing

---

**Last Updated**: December 25, 2025
**Version**: 1.0 (Inference Only)
