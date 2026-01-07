# YOLOX ONNX Export Tool

Export YOLOX PyTorch models to ONNX format for use with OpenVINO.

## Features

- ✅ **Simple interface** - No experiment file needed, just specify model name
- ✅ **Automatic verification** - Validates exported model and tests inference
- ✅ **Output validation** - Confirms correct coordinate format (pixel vs normalized)
- ✅ **OpenVINO testing** - Tests the model with OpenVINO runtime
- ✅ **Helpful errors** - Clear guidance when files are missing

## Usage

### Basic Export

```bash
# Export YOLOX-S model
python tools/export_onnx.py -c models/yolox_s.pth

# Export YOLOX-M model
python tools/export_onnx.py -n yolox-m -c models/yolox_m.pth

# Specify output path
python tools/export_onnx.py -c models/yolox_s.pth -o my_model.onnx
```

### Advanced Options

```bash
# Skip verification (faster, but not recommended)
python tools/export_onnx.py -c models/yolox_s.pth --no-verify

# Enable simplification (may cause issues)
python tools/export_onnx.py -c models/yolox_s.pth --simplify

# Specify opset version
python tools/export_onnx.py -c models/yolox_s.pth --opset 18
```

## Supported Models

- `yolox-nano` - Smallest, fastest
- `yolox-tiny` - Tiny model
- `yolox-s` - Small model (default)
- `yolox-m` - Medium model
- `yolox-l` - Large model
- `yolox-x` - Extra large model

## Download Pretrained Models

```bash
# YOLOX-S
wget -P models https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth

# YOLOX-M
wget -P models https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth

# YOLOX-L
wget -P models https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth
```

## Using Exported Models

After exporting, use the ONNX model with OpenVINO:

```bash
# Run video detection
python tools/demo_video_openvino.py test_videos/car_park.mp4 --model models/yolox_s.onnx --device GPU

# Run image detection
python tools/demo_openvino.py --image test.jpg --model models/yolox_s.onnx
```

## Verification Output

The script automatically verifies the exported model:

```
✅ Model verification passed!
✅ Test inference successful!
✅ Output format looks correct (pixel coordinates)
```

If you see warnings about normalized coordinates, the export may have issues.

## Troubleshooting

### "Output values are very small" Warning

This indicates the model is outputting normalized coordinates (0-1) instead of pixel coordinates (0-640). This will cause invisible bounding boxes. Re-export using this updated script to fix.

### Opset Version Warnings

The script may auto-upgrade from opset 11 to 18. This is normal and doesn't affect functionality.

### Simplification Issues

The `--simplify` flag is not recommended as it can cause export failures. Only use if you know what you're doing.
