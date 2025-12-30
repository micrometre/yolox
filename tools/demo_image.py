#!/usr/bin/env python3
"""
Minimal YOLOX inference example.
Runs object detection on an image using a pretrained YOLOX model.
"""
import sys
import os
import argparse
import cv2
import torch
import numpy as np

# Add YOLOX to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'YOLOX'))

from yolox.exp import get_exp
from yolox.utils import postprocess, vis

# COCO class names
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


def preprocess(img, input_size=(640, 640)):
    """Preprocess image for YOLOX inference."""
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


def main():
    parser = argparse.ArgumentParser(description="YOLOX Inference Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/yolox_s.pth", help="Path to model weights")
    parser.add_argument("--output", type=str, default="test_output.jpg", help="Path to output image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--size", type=int, default=640, help="Input size")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        return

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Please download the model first:")
        print("wget -P models https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth")
        return

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    exp = get_exp(None, "yolox-s")
    model = exp.get_model()
    model.eval()
    
    checkpoint = torch.load(args.model, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    print("Model loaded successfully!")

    # Load and preprocess image
    print(f"Loading image: {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read image '{args.image}'")
        return
    
    original_img = img.copy()
    img, ratio = preprocess(img, (args.size, args.size))
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(
            outputs, exp.num_classes, args.conf, args.nms
        )[0]

    if outputs is None:
        print("No detections found!")
        cv2.imwrite(args.output, original_img)
        print(f"Original image saved to: {args.output}")
        return

    # Process detections
    outputs = outputs.cpu().numpy()
    bboxes = outputs[:, 0:4] / ratio
    cls = outputs[:, 6]
    scores = outputs[:, 4] * outputs[:, 5]

    # Draw detections
    print(f"\nDetected {len(bboxes)} objects:")
    for i, (bbox, score, cls_id) in enumerate(zip(bboxes, scores, cls)):
        x1, y1, x2, y2 = bbox.astype(int)
        class_name = COCO_CLASSES[int(cls_id)]
        print(f"  {i+1}. {class_name}: {score:.2f} at [{x1}, {y1}, {x2}, {y2}]")
        
        # Draw bounding box
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(original_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(original_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save output
    cv2.imwrite(args.output, original_img)
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
