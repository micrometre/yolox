#!/usr/bin/env python3
"""
Process video files for object detection and save frames with detected objects to an output directory.
Uses YOLOX for real-time object detection.
"""

import sys
import os
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add YOLOX to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'YOLOX'))

from yolox.exp import get_exp
from yolox.utils import postprocess

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

# Vehicle classes
VEHICLE_CLASSES = {"car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "bicycle"}


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


def draw_detections(img, bboxes, scores, cls_ids, threshold=0.5):
    """Draw bounding boxes on image."""
    for bbox, score, cls_id in zip(bboxes, scores, cls_ids):
        if score < threshold:
            continue
            
        x1, y1, x2, y2 = bbox.astype(int)
        class_name = COCO_CLASSES[int(cls_id)]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img


def process_video(
    video_path,
    output_dir="images",
    frame_skip=1,
    threshold=0.6,
    model_path="models/yolox_s.pth",
    device="cpu",
    save_video=True,
    input_size=640,
    nms_threshold=0.45,
    filter_vehicles_only=False
):
    """
    Process video file for object detection.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save detected frames
        frame_skip: Process every Nth frame
        threshold: Detection confidence threshold
        model_path: Path to YOLOX model weights
        device: Device to use (cpu or cuda)
        save_video: Save output as video file
        input_size: Input size for model
        nms_threshold: NMS threshold
        filter_vehicles_only: Only save frames with vehicles detected
    """
    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found!")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found! "
            "Please download it first:\n"
            "wget -P models https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
        )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(device.lower())
    print(f"Using device: {device}")
    
    # Load model
    print("Loading YOLOX model...")
    exp = get_exp(None, "yolox-s")
    model = exp.get_model()
    model.eval()
    
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    print("Model loaded successfully!")
    
    # Open video
    print(f"\nOpening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
    print(f"Processing every {frame_skip} frame(s)")
    print(f"Detection threshold: {threshold}")
    
    # Setup video writer if needed
    video_writer = None
    if save_video:
        output_video_path = output_path / f"{Path(video_path).stem}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps / frame_skip, (width, height))
        print(f"Saving output video to: {output_video_path}")
    
    # Process video
    frame_count = 0
    saved_count = 0
    detection_stats = {}
    
    with tqdm(total=total_frames // frame_skip, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Preprocess
            img_tensor, ratio = preprocess(frame, (input_size, input_size))
            img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float().to(device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(img_tensor)
                outputs = postprocess(outputs, exp.num_classes, threshold, nms_threshold)[0]
            
            # Process detections
            has_detections = False
            has_vehicles = False
            
            if outputs is not None:
                outputs = outputs.cpu().numpy()
                bboxes = outputs[:, 0:4] / ratio
                cls = outputs[:, 6]
                scores = outputs[:, 4] * outputs[:, 5]
                
                # Count detections
                for cls_id, score in zip(cls, scores):
                    if score >= threshold:
                        class_name = COCO_CLASSES[int(cls_id)]
                        detection_stats[class_name] = detection_stats.get(class_name, 0) + 1
                        has_detections = True
                        
                        if class_name in VEHICLE_CLASSES:
                            has_vehicles = True
                
                # Draw detections
                if has_detections:
                    frame = draw_detections(frame, bboxes, scores, cls, threshold)
            
            # Save frame if it has detections (or vehicles if filtering)
            should_save = has_detections and (not filter_vehicles_only or has_vehicles)
            
            if should_save:
                frame_filename = output_path / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                saved_count += 1
            
            # Save to video
            if save_video and video_writer is not None:
                video_writer.write(frame)
            
            frame_count += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()
    
    # Print statistics
    print(f"\n✅ Processing complete!")
    print(f"   Processed: {frame_count} frames")
    print(f"   Saved: {saved_count} frames to {output_dir}/")
    
    if detection_stats:
        print(f"\n📊 Detection statistics:")
        for class_name, count in sorted(detection_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"   {class_name}: {count}")
    else:
        print("\n⚠️  No detections found above threshold")
    
    if save_video:
        print(f"\n🎥 Output video saved to: {output_video_path}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Process video files for object detection using YOLOX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file (mp4, avi, etc.)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="images",
        help="Directory to save detected frames"
    )
    
    parser.add_argument(
        "-s", "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (1 = all frames, 5 = every 5th frame)"
    )
    
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.6,
        help="Detection confidence threshold (0.0-1.0)"
    )
    
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on"
    )
    
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save output as video file (in addition to frames)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolox_s.pth",
        help="Path to YOLOX model weights"
    )
    
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        help="Input size for model"
    )
    
    parser.add_argument(
        "--nms",
        type=float,
        default=0.45,
        help="NMS threshold"
    )
    
    parser.add_argument(
        "--vehicles-only",
        action="store_true",
        help="Only save frames with vehicles detected"
    )
    
    args = parser.parse_args()
    
    # Process video
    try:
        process_video(
            video_path=args.video,
            output_dir=args.output_dir,
            frame_skip=args.frame_skip,
            threshold=args.threshold,
            model_path=args.model,
            device=args.device,
            save_video=args.save_video,
            input_size=args.size,
            nms_threshold=args.nms,
            filter_vehicles_only=args.vehicles_only
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
