#!/usr/bin/env python3
"""Test different input normalizations for ONNX model"""
import sys
sys.path.insert(0, 'YOLOX')

import cv2
import torch
import numpy as np
from openvino import Core
from yolox.utils import postprocess as yolox_postprocess

# Load test image
img = cv2.imread('output/frame_000000.jpg')

def preprocess(img, input_size=(640, 640), normalize=False):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img.transpose((2, 0, 1))
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    
    if normalize:
        padded_img = padded_img / 255.0
    
    return padded_img, r

# Test with normalization
input_normalized, ratio = preprocess(img, (640, 640), normalize=True)
print(f"Normalized input range: [{input_normalized.min():.3f}, {input_normalized.max():.3f}]")

core = Core()
model_onnx = core.read_model("models/yolox_s.onnx")
compiled_model = core.compile_model(model_onnx, "GPU")

input_onnx = np.expand_dims(input_normalized, axis=0)
results = compiled_model([input_onnx])
onnx_output = list(results.values())[0]

onnx_output_torch = torch.from_numpy(onnx_output)
onnx_postprocessed = yolox_postprocess(onnx_output_torch, 80, 0.6, 0.45)[0]
if onnx_postprocessed is not None:
    onnx_postprocessed = onnx_postprocessed.cpu().numpy()
    print(f"\nWith normalized input (0-1):")
    print(f"  Detections: {len(onnx_postprocessed)}")
    if len(onnx_postprocessed) > 0:
        print(f"  First detection (640x640 space): {onnx_postprocessed[0, :4]}")
        print(f"  First detection (original space): {onnx_postprocessed[0, :4] / ratio}")
else:
    print("\nWith normalized input: No detections")

# Test without normalization (0-255)
input_unnormalized, ratio = preprocess(img, (640, 640), normalize=False)
print(f"\nUnnormalized input range: [{input_unnormalized.min():.1f}, {input_unnormalized.max():.1f}]")

input_onnx = np.expand_dims(input_unnormalized, axis=0)
results = compiled_model([input_onnx])
onnx_output = list(results.values())[0]

onnx_output_torch = torch.from_numpy(onnx_output)
onnx_postprocessed = yolox_postprocess(onnx_output_torch, 80, 0.6, 0.45)[0]
if onnx_postprocessed is not None:
    onnx_postprocessed = onnx_postprocessed.cpu().numpy()
    print(f"\nWith unnormalized input (0-255):")
    print(f"  Detections: {len(onnx_postprocessed)}")
    if len(onnx_postprocessed) > 0:
        print(f"  First detection (640x640 space): {onnx_postprocessed[0, :4]}")
        print(f"  First detection (original space): {onnx_postprocessed[0, :4] / ratio}")
else:
    print("\nWith unnormalized input: No detections")
