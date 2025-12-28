# YOLOX Minimal Example

A minimal working example of YOLOX object detection in Python.

## Overview

This repository contains a simple setup for running YOLOX object detection on images. YOLOX is a high-performance anchor-free YOLO detector that supports real-time object detection.

## Setup

### 1. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download YOLOX source code

```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
```

### 4. Download pretrained model

```bash
mkdir -p models
wget -P models https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

## Usage

### Run inference on an image

```bash
python demo_image.py --image test_images/street_scene.png --output result.jpg
```

### Command-line options

- `--image`: Path to input image (required)
- `--model`: Path to model weights (default: `models/yolox_s.pth`)
- `--output`: Path to output image (default: `test_output.jpg`)
- `--conf`: Confidence threshold (default: 0.25)
- `--nms`: NMS threshold (default: 0.45)
- `--size`: Input size (default: 640)
- `--device`: Device to use - `cpu` or `cuda` (default: `cpu`)

### Example with custom parameters

```bash
python demo_image.py \
    --image test_images/street_scene.png \
    --output detections.jpg \
    --conf 0.3 \
    --nms 0.5 \
    --device cpu
```

## Model Information

**YOLOX-S** (Small variant):
- Size: ~69 MB
- Input: 640x640
- COCO mAP: ~40.5%
- Speed: ~30 FPS on GPU

Other available models (download from [YOLOX releases](https://github.com/Megvii-BaseDetection/YOLOX/releases)):
- `yolox_nano.pth` - Smallest, fastest
- `yolox_tiny.pth` - Tiny variant
- `yolox_s.pth` - Small (included)
- `yolox_m.pth` - Medium
- `yolox_l.pth` - Large
- `yolox_x.pth` - Extra large, most accurate

## Detected Object Classes

YOLOX is trained on COCO dataset and can detect 80 object classes including:
- People
- Vehicles (car, truck, bus, motorcycle, bicycle, etc.)
- Animals (dog, cat, bird, horse, etc.)
- Common objects (chair, bottle, laptop, phone, etc.)

See the full list in `demo_image.py`.

## Project Structure

```
yolox/
├── .venv/              # Virtual environment
├── YOLOX/              # YOLOX source code (cloned)
├── models/             # Model weights
│   └── yolox_s.pth
├── test_images/        # Test images
│   └── street_scene.png
├── demo_image.py       # Main inference script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Troubleshooting

### ImportError: No module named 'yolox'

The script adds YOLOX to the Python path automatically. Make sure the YOLOX directory exists in the project root.

### CUDA out of memory

Use CPU instead:
```bash
python demo_image.py --image your_image.jpg --device cpu
```

Or reduce the input size:
```bash
python demo_image.py --image your_image.jpg --size 416
```

## References

- [YOLOX GitHub Repository](https://github.com/Megvii-BaseDetection/YOLOX)
- [YOLOX Paper](https://arxiv.org/abs/2107.08430)
- [YOLOX Documentation](https://yolox.readthedocs.io/)

## License

This example follows the Apache 2.0 license of the YOLOX project.
