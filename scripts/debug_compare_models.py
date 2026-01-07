#!/usr/bin/env python3
"""
Compare PyTorch and ONNX YOLOX outputs to debug coordinate space issues.
"""
import sys
sys.path.insert(0, 'YOLOX')

import cv2
import torch
import numpy as np
from openvino import Core
from yolox.exp import get_exp
from yolox.utils import postprocess as yolox_postprocess

# Load test image
img = cv2.imread('output/frame_000000.jpg')
print(f"Image shape: {img.shape}")

# Preprocess
def preprocess(img, input_size=(640, 640)):
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
    return padded_img, r

input_np, ratio = preprocess(img, (640, 640))
print(f"Ratio: {ratio}")
print(f"Input shape: {input_np.shape}")

# PyTorch inference
print("\n=== PyTorch Model ===")
exp = get_exp(None, "yolox-s")
model = exp.get_model()
model.eval()
checkpoint = torch.load("models/yolox_s.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"])

input_torch = torch.from_numpy(input_np).unsqueeze(0).float()
with torch.no_grad():
    pytorch_output = model(input_torch)
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"PyTorch output (first detection, raw): {pytorch_output[0, 0, :]}")
    
    pytorch_postprocessed = yolox_postprocess(pytorch_output, exp.num_classes, 0.6, 0.45)[0]
    if pytorch_postprocessed is not None:
        pytorch_postprocessed = pytorch_postprocessed.cpu().numpy()
        print(f"PyTorch postprocessed shape: {pytorch_postprocessed.shape}")
        print(f"PyTorch first detection (640x640 space): {pytorch_postprocessed[0, :4]}")
        print(f"PyTorch first detection (original space): {pytorch_postprocessed[0, :4] / ratio}")

# ONNX/OpenVINO inference
print("\n=== ONNX/OpenVINO Model ===")
core = Core()
model_onnx = core.read_model("models/yolox_s.onnx")
compiled_model = core.compile_model(model_onnx, "GPU")

input_onnx = np.expand_dims(input_np, axis=0)
results = compiled_model([input_onnx])
onnx_output = list(results.values())[0]
print(f"ONNX output shape: {onnx_output.shape}")
print(f"ONNX output (first detection, raw): {onnx_output[0, 0, :]}")

onnx_output_torch = torch.from_numpy(onnx_output)
onnx_postprocessed = yolox_postprocess(onnx_output_torch, exp.num_classes, 0.6, 0.45)[0]
if onnx_postprocessed is not None:
    onnx_postprocessed = onnx_postprocessed.cpu().numpy()
    print(f"ONNX postprocessed shape: {onnx_postprocessed.shape}")
    print(f"ONNX first detection (640x640 space): {onnx_postprocessed[0, :4]}")
    print(f"ONNX first detection (original space): {onnx_postprocessed[0, :4] / ratio}")

print("\n=== Comparison ===")
if pytorch_postprocessed is not None and onnx_postprocessed is not None:
    print(f"Difference in first box: {np.abs(pytorch_postprocessed[0, :4] - onnx_postprocessed[0, :4])}")
