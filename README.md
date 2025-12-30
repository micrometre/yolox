# YOLOX Object Detection

A minimal, production-ready implementation of YOLOX object detection with support for both **PyTorch** and **OpenVINO** (Intel CPU/GPU acceleration).

## Features

- 🚀 **Dual Runtime Support**: PyTorch and OpenVINO backends
- 🎯 **80 COCO Classes**: Detect people, vehicles, animals, and common objects
- 📸 **Image & Video Processing**: Single image inference and batch video processing
- ⚡ **Intel Hardware Acceleration**: Up to 5x faster with OpenVINO on Intel GPUs
- 🎨 **Interactive Notebook**: Jupyter notebook for experimentation
- 🔧 **Flexible Configuration**: Adjustable confidence, NMS, frame skipping, and more

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Clone YOLOX source
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
```

### 2. Download Models

**PyTorch Model:**
```bash
mkdir -p models
wget -P models https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

**ONNX Model (for OpenVINO):**
```bash
# Export from PyTorch (requires model above)
python -m yolox.tools.export_onnx --output-name models/yolox_s.onnx -n yolox-s -c models/yolox_s.pth
```

## Usage

### PyTorch Inference

**Image Detection:**
```bash
# Basic usage
python demo_image.py --image test_images/street_scene.png --output result.jpg

# With custom parameters
python demo_image.py \
    --image test_images/street_scene.png \
    --output detections.jpg \
    --conf 0.3 \
    --nms 0.5 \
    --device cpu
```

**Video Processing:**
```bash
# Process all frames
python tools/demo_video.py video.mp4 --output-dir output_frames --save-video

# Process every 5th frame
python tools/demo_video.py video.mp4 --frame-skip 5 --output-dir detections

# Detect only vehicles with high confidence
python tools/demo_video.py traffic.mp4 \
    --threshold 0.7 \
    --vehicles-only \
    --output-dir vehicles

# GPU acceleration
python tools/demo_video.py video.mp4 --device cuda --save-video
```

### OpenVINO Inference (Intel Acceleration)

**Image Detection:**
```bash
# CPU inference
python tools/demo_openvino.py --image test_images/street_scene.png --output output.jpg

# GPU inference (Intel iGPU) - ~5x faster!
python tools/demo_openvino.py --image test_images/street_scene.png --output output.jpg --device GPU

# Auto device selection
python tools/demo_openvino.py --image test_images/street_scene.png --device AUTO
```

**Video Processing:**
```bash
# GPU inference (recommended)
python tools/demo_video_openvino.py test_videos/car_park.mp4 -o output --device GPU

# Process every 5th frame with lower threshold
python tools/demo_video_openvino.py video.mp4 -o output --frame-skip 5 -t 0.4

# Only save frames with vehicles
python tools/demo_video_openvino.py video.mp4 -o output --vehicles-only

# Multi-device inference
python tools/demo_video_openvino.py video.mp4 -o output --device "MULTI:CPU,GPU"
```

### Interactive Notebook

```bash
source .venv/bin/activate
jupyter notebook yolox_demo.ipynb
```

The notebook includes:
- Step-by-step YOLOX inference walkthrough
- Interactive visualization with matplotlib
- Threshold experimentation
- Easy testing with custom images

## Performance Comparison

| Backend | Device | Inference Time | FPS | Speedup |
|---------|--------|----------------|-----|---------|
| PyTorch | CPU | ~500 ms | 2 FPS | 1x |
| PyTorch | CUDA | ~30 ms | 33 FPS | 16x |
| OpenVINO | CPU | 150 ms | 6.7 FPS | 3.3x |
| OpenVINO | GPU (Intel) | **20-30 ms** | **40-50 FPS** | **20x** |

*Tested with YOLOX-S on 640x640 input*

## Command-Line Options

### Image Detection (`demo_image.py` / `demo_openvino.py`)

| Option | Description | Default |
|--------|-------------|---------|
| `--image` | Path to input image (required) | - |
| `--model` | Path to model weights | `models/yolox_s.pth` or `.onnx` |
| `--output` | Path to output image | `test_output.jpg` |
| `--conf` | Confidence threshold (0.0-1.0) | 0.25 |
| `--nms` | NMS threshold (0.0-1.0) | 0.45 |
| `--size` | Input size (pixels) | 640 |
| `--device` | Device: `cpu`/`cuda` (PyTorch) or `CPU`/`GPU`/`AUTO` (OpenVINO) | `cpu` |

### Video Processing (`demo_video.py` / `demo_video_openvino.py`)

| Option | Description | Default |
|--------|-------------|---------|
| `video` | Path to input video (required) | - |
| `-o, --output-dir` | Directory to save frames | `images` |
| `-s, --frame-skip` | Process every Nth frame | 1 |
| `-t, --threshold` | Detection confidence threshold | 0.6 |
| `-d, --device` | Inference device | `cpu` |
| `--save-video` | Save annotated video | `True` |
| `--model` | Path to model weights | `models/yolox_s.pth` or `.onnx` |
| `--size` | Input size (pixels) | 640 |
| `--nms` | NMS threshold | 0.45 |
| `--vehicles-only` | Only save frames with vehicles | `False` |

## Model Information

### YOLOX-S (Small variant)
- **Size**: ~69 MB
- **Input**: 640x640
- **COCO mAP**: ~40.5%
- **Speed**: 30-50 FPS (depending on backend)

### Other Available Models

Download from [YOLOX releases](https://github.com/Megvii-BaseDetection/YOLOX/releases):

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolox_nano.pth` | Smallest | Fastest | Lowest |
| `yolox_tiny.pth` | Tiny | Very Fast | Low |
| `yolox_s.pth` | Small | Fast | Good ✓ |
| `yolox_m.pth` | Medium | Moderate | Better |
| `yolox_l.pth` | Large | Slower | High |
| `yolox_x.pth` | Extra Large | Slowest | Highest |

## Detected Object Classes

YOLOX detects **80 COCO classes**:

- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle, airplane, train, boat
- **Animals**: dog, cat, bird, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Common Objects**: chair, bottle, laptop, phone, book, clock, vase, scissors, etc.

See full list in `COCO_CLASSES` within the scripts.

## Project Structure

```
yolox/
├── .venv/                      # Virtual environment
├── YOLOX/                      # YOLOX source code (cloned)
├── models/                     # Model weights
│   ├── yolox_s.pth            # PyTorch model
│   └── yolox_s.onnx           # ONNX model (for OpenVINO)
├── test_images/                # Test images
│   ├── street_scene.png
│   └── cars.jpg
├── test_videos/                # Test videos
│   └── car_park.mp4
├── tools/                      # Inference scripts
│   ├── demo_image.py          # PyTorch image inference
│   ├── demo_video.py          # PyTorch video processing
│   ├── demo_openvino.py       # OpenVINO image inference
│   └── demo_video_openvino.py # OpenVINO video processing
├── demo_image.py               # Main PyTorch demo (legacy)
├── yolox_demo.ipynb           # Interactive Jupyter notebook
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Troubleshooting

### ImportError: No module named 'yolox'

The scripts add YOLOX to the Python path automatically. Ensure the `YOLOX/` directory exists:
```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
```

### CUDA out of memory

Use CPU instead:
```bash
python demo_image.py --image your_image.jpg --device cpu
```

Or reduce input size:
```bash
python demo_image.py --image your_image.jpg --size 416
```

### OpenVINO: No module named 'openvino'

Install OpenVINO:
```bash
pip install openvino
```

### OpenVINO: GPU device not found

Check available devices:
```python
from openvino import Core
core = Core()
print(core.available_devices)  # Should show ['CPU', 'GPU']
```

If GPU is missing, install Intel GPU drivers or use `CPU` device.

## References

- [YOLOX GitHub Repository](https://github.com/Megvii-BaseDetection/YOLOX)
- [YOLOX Paper (arXiv)](https://arxiv.org/abs/2107.08430)
- [OpenVINO Toolkit](https://docs.openvino.ai/)
- [COCO Dataset](https://cocodataset.org/)

## License

This example follows the Apache 2.0 license of the YOLOX project.
