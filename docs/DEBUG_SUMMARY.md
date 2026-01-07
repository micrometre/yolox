# YOLOX OpenVINO Video Detection - Debugging Summary

## Problem
The `demo_video_openvino.py` script was detecting objects correctly (showing detection statistics) but the bounding boxes were not visible in the output video or saved frames.

## Root Cause
The ONNX model file (`models/yolox_s.onnx`) was outputting bounding box coordinates in a **normalized format (0-1 range)** instead of **pixel coordinates (0-640 range)**. This caused the boxes to be rendered as tiny 2x3 pixel rectangles that were invisible.

### Evidence
- **Old ONNX model output**: `[0.855, 0.447, 0.711, 0.144]` (normalized)
- **PyTorch model output**: `[6.97, 3.53, 16.47, 9.33]` (pixel coordinates)
- **New ONNX model output**: `[12.3, 5.6, 23.4, 12.2]` (pixel coordinates) ✅

## Solution
Re-exported the ONNX model from the PyTorch checkpoint using the correct export settings.

### Export Script
Created `export_yolox_onnx.sh` which:
1. Backs up the existing ONNX model
2. Loads the PyTorch checkpoint (`models/yolox_s.pth`)
3. Exports to ONNX format with proper settings
4. Verifies the exported model
5. Tests inference to confirm correct output format

### Usage
```bash
./export_yolox_onnx.sh
```

## Results
✅ **Fixed!** The video detection now works correctly:
- Bounding boxes are visible and properly positioned
- Detection statistics: 275 cars, 125 persons, 33 parking meters, 9 traffic lights
- Inference speed: ~32 FPS on GPU

## Files Modified
1. **`export_yolox_onnx.sh`** - New export script for future use
2. **`tools/demo_video_openvino.py`** - Fixed to handle dynamic shapes in new ONNX model
3. **`models/yolox_s.onnx`** - Re-exported with correct output format

## Testing
```bash
# Run detection on video
python tools/demo_video_openvino.py test_videos/car_park.mp4 -o output --device GPU

# Check output
ls output/frame_*.jpg
ffplay output/car_park_detected_openvino.mp4
```

## Notes
- The old ONNX model is backed up as `models/yolox_s.onnx.backup.YYYYMMDD_HHMMSS`
- The new model uses opset version 18 (auto-upgraded from 11)
- The model has dynamic batch dimension for flexibility
