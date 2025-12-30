#!/usr/bin/env python3
"""
Minimal YOLOX OpenVINO inference example.
Runs object detection on an image using OpenVINO for Intel CPU/GPU acceleration.
"""
import sys
import os
import argparse
import cv2
import numpy as np

# OpenVINO Runtime API
from openvino import Core

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
    """Preprocess image for YOLOX inference (letterbox + normalize)."""
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    # Calculate resize ratio (maintain aspect ratio)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    
    # Place resized image on padded background
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    # Convert to NCHW format and float32
    padded_img = padded_img.transpose((2, 0, 1))  # HWC -> CHW
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    padded_img = np.expand_dims(padded_img, axis=0)  # Add batch dimension
    
    return padded_img, r


def nms(boxes, scores, nms_thresh):
    """Non-maximum suppression."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def postprocess(outputs, img_shape, input_size, conf_thresh, nms_thresh, ratio):
    """
    Postprocess YOLOX output.
    YOLOX outputs: [batch, num_anchors, 85] where 85 = 4 (bbox) + 1 (obj) + 80 (classes)
    """
    predictions = outputs[0]  # [num_anchors, 85]
    
    # Get box coordinates (center_x, center_y, width, height)
    box_corner = np.zeros_like(predictions[:, :4])
    box_corner[:, 0] = predictions[:, 0] - predictions[:, 2] / 2  # x1
    box_corner[:, 1] = predictions[:, 1] - predictions[:, 3] / 2  # y1
    box_corner[:, 2] = predictions[:, 0] + predictions[:, 2] / 2  # x2
    box_corner[:, 3] = predictions[:, 1] + predictions[:, 3] / 2  # y2
    predictions[:, :4] = box_corner
    
    # Get objectness and class scores
    obj_conf = predictions[:, 4]
    class_conf = predictions[:, 5:]
    class_pred = np.argmax(class_conf, axis=1)
    class_score = class_conf[np.arange(len(class_pred)), class_pred]
    
    # Combined score = objectness * class_confidence
    scores = obj_conf * class_score
    
    # Filter by confidence threshold
    mask = scores > conf_thresh
    boxes = predictions[mask, :4]
    scores = scores[mask]
    class_ids = class_pred[mask]
    
    if len(boxes) == 0:
        return None, None, None
    
    # Scale boxes back to original image size
    boxes = boxes / ratio
    
    # Apply NMS per class
    final_boxes = []
    final_scores = []
    final_class_ids = []
    
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        keep = nms(cls_boxes, cls_scores, nms_thresh)
        final_boxes.extend(cls_boxes[keep])
        final_scores.extend(cls_scores[keep])
        final_class_ids.extend([cls_id] * len(keep))
    
    return np.array(final_boxes), np.array(final_scores), np.array(final_class_ids)


def draw_detections(img, boxes, scores, class_ids):
    """Draw bounding boxes and labels on the image."""
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        class_name = COCO_CLASSES[int(cls_id)]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background and text
        label = f"{class_name}: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="YOLOX OpenVINO Inference Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/yolox_s.onnx", 
                        help="Path to ONNX model (OpenVINO will convert automatically)")
    parser.add_argument("--output", type=str, default="output_openvino.jpg", help="Path to output image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--size", type=int, default=640, help="Input size")
    parser.add_argument("--device", type=str, default="CPU", 
                        help="OpenVINO device: CPU, GPU, AUTO, or MULTI:CPU,GPU")
    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        return

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Please ensure you have the ONNX model. You can export it from PyTorch:")
        print("  python -m yolox.tools.export_onnx --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth")
        return

    # Initialize OpenVINO
    print(f"Initializing OpenVINO with device: {args.device}")
    core = Core()
    
    # Print available devices
    print(f"Available devices: {core.available_devices}")
    
    # Read and compile model
    print(f"Loading model: {args.model}")
    model = core.read_model(args.model)
    compiled_model = core.compile_model(model, args.device)
    
    # Get input/output info
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    print(f"Input shape: {input_layer.shape}")
    print(f"Output shape: {output_layer.shape}")

    # Load and preprocess image
    print(f"Loading image: {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read image '{args.image}'")
        return
    
    original_img = img.copy()
    input_tensor, ratio = preprocess(img, (args.size, args.size))
    print(f"Preprocessed input shape: {input_tensor.shape}, ratio: {ratio:.4f}")

    # Run inference
    print("Running OpenVINO inference...")
    import time
    start_time = time.time()
    
    results = compiled_model([input_tensor])
    outputs = results[output_layer]
    
    inference_time = (time.time() - start_time) * 1000
    print(f"Inference time: {inference_time:.2f} ms")

    # Postprocess
    boxes, scores, class_ids = postprocess(
        outputs, img.shape, args.size, args.conf, args.nms, ratio
    )

    if boxes is None or len(boxes) == 0:
        print("No detections found!")
        cv2.imwrite(args.output, original_img)
        print(f"Original image saved to: {args.output}")
        return

    # Print and draw detections
    print(f"\nDetected {len(boxes)} objects:")
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = box.astype(int)
        class_name = COCO_CLASSES[int(cls_id)]
        print(f"  {i+1}. {class_name}: {score:.2f} at [{x1}, {y1}, {x2}, {y2}]")

    output_img = draw_detections(original_img, boxes, scores, class_ids)
    
    # Save output
    cv2.imwrite(args.output, output_img)
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
