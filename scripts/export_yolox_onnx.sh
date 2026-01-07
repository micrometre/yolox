#!/bin/bash
# Export YOLOX model to ONNX format
# This script re-exports the YOLOX PyTorch model to ONNX to ensure correct output format

set -e  # Exit on error

echo "========================================="
echo "YOLOX ONNX Model Export Script"
echo "========================================="
echo ""

# Configuration
MODEL_NAME="yolox-s"
CHECKPOINT="models/yolox_s.pth"
OUTPUT_DIR="models"
OUTPUT_NAME="yolox_s.onnx"
OPSET_VERSION=11

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint file not found: $CHECKPOINT"
    echo "Please download it first:"
    echo "  wget -P models https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
    exit 1
fi

# Backup existing ONNX model if it exists
if [ -f "$OUTPUT_DIR/$OUTPUT_NAME" ]; then
    BACKUP_NAME="${OUTPUT_NAME}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "📦 Backing up existing model to: $OUTPUT_DIR/$BACKUP_NAME"
    mv "$OUTPUT_DIR/$OUTPUT_NAME" "$OUTPUT_DIR/$BACKUP_NAME"
    
    # Also backup .data file if it exists
    if [ -f "$OUTPUT_DIR/${OUTPUT_NAME}.data" ]; then
        mv "$OUTPUT_DIR/${OUTPUT_NAME}.data" "$OUTPUT_DIR/${BACKUP_NAME}.data"
    fi
fi

# Export model using Python
echo "🔄 Exporting $MODEL_NAME to ONNX..."
echo "   Checkpoint: $CHECKPOINT"
echo "   Output: $OUTPUT_DIR/$OUTPUT_NAME"
echo "   Opset version: $OPSET_VERSION"
echo ""

python3 << 'PYTHON_SCRIPT'
import sys
import os
sys.path.insert(0, 'YOLOX')

import torch
import torch.onnx
from yolox.exp import get_exp

# Configuration
MODEL_NAME = "yolox-s"
CHECKPOINT = "models/yolox_s.pth"
OUTPUT_PATH = "models/yolox_s.onnx"
OPSET_VERSION = 11
INPUT_SIZE = (640, 640)

print("Loading experiment configuration...")
exp = get_exp(None, MODEL_NAME)

print("Creating model...")
model = exp.get_model()
model.eval()

print(f"Loading checkpoint: {CHECKPOINT}")
checkpoint = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(checkpoint["model"])

print(f"Model loaded successfully!")
print(f"  Test size: {exp.test_size}")
print(f"  Number of classes: {exp.num_classes}")

# Create dummy input
dummy_input = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1])

print(f"\nExporting to ONNX (opset={OPSET_VERSION})...")
print(f"  Input shape: {dummy_input.shape}")

torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_PATH,
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={
        "images": {0: "batch"},
        "output": {0: "batch"}
    },
    opset_version=OPSET_VERSION,
    verbose=False
)

print(f"✅ ONNX model exported to: {OUTPUT_PATH}")

# Verify the export
import onnx
print("\nVerifying exported model...")
onnx_model = onnx.load(OUTPUT_PATH)
onnx.checker.check_model(onnx_model)
print("✅ Model verification passed!")

print(f"\nModel info:")
print(f"  Input: {onnx_model.graph.input[0].name} - {[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
print(f"  Output: {onnx_model.graph.output[0].name} - {[d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]}")
print(f"  Opset version: {onnx_model.opset_import[0].version}")

# Test the model with a quick inference
print("\nTesting exported model...")
import numpy as np
from openvino import Core

core = Core()
model_ov = core.read_model(OUTPUT_PATH)
compiled_model = core.compile_model(model_ov, "CPU")

test_input = np.random.randn(1, 3, 640, 640).astype(np.float32) * 114
results = compiled_model([test_input])
output = list(results.values())[0]

print(f"✅ Test inference successful!")
print(f"  Output shape: {output.shape}")
print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
print(f"  First detection bbox: {output[0, 0, :4]}")

PYTHON_SCRIPT

echo ""
echo "========================================="
echo "✅ Export complete!"
echo "========================================="
echo ""
echo "The new ONNX model is ready at: $OUTPUT_DIR/$OUTPUT_NAME"
echo ""
echo "To test the model, run:"
echo "  python tools/demo_video_openvino.py test_videos/car_park.mp4 -o output --device GPU"
echo ""
