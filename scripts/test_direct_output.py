#!/usr/bin/env python3
"""Test using ONNX output directly without YOLOX postprocess"""
import sys
sys.path.insert(0, 'YOLOX')

import cv2
import numpy as np
from openvino import Core

# Load test image
img = cv2.imread('output/frame_000000.jpg')

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

core = Core()
model_onnx = core.read_model("models/yolox_s.onnx")
compiled_model = core.compile_model(model_onnx, "GPU")

input_onnx = np.expand_dims(input_np, axis=0)
results = compiled_model([input_onnx])
onnx_output = list(results.values())[0]

print(f"ONNX output shape: {onnx_output.shape}")
print(f"First 10 detections (raw):")
for i in range(min(10, onnx_output.shape[1])):
    det = onnx_output[0, i, :]
    # Assuming format: [cx, cy, w, h, objectness, class_scores...]
    cx, cy, w, h = det[:4]
    obj_conf = det[4]
    class_scores = det[5:]
    class_id = np.argmax(class_scores)
    class_conf = class_scores[class_id]
    combined_score = obj_conf * class_conf
    
    if combined_score > 0.1:  # Low threshold to see what we get
        # Convert from normalized to pixel coords
        cx_px, cy_px, w_px, h_px = cx * 640, cy * 640, w * 640, h * 640
        # Convert from center format to corner format
        x1, y1 = cx_px - w_px/2, cy_px - h_px/2
        x2, y2 = cx_px + w_px/2, cy_px + h_px/2
        # Scale to original image
        x1, y1, x2, y2 = x1/ratio, y1/ratio, x2/ratio, y2/ratio
        
        print(f"  Det {i}: score={combined_score:.3f}, class={class_id}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
