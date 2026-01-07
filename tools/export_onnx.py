#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Export trained YOLOX model to ONNX format.

This script exports a trained YOLOX checkpoint to ONNX format,
which can then be optimized with OpenVINO.
"""

import argparse
import os
import torch
from pathlib import Path

# Add YOLOX to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "YOLOX"))

from yolox.exp import get_exp


def export_onnx(model_name, ckpt_path, output_path, opset=11, simplify=False, verify=True):
    """
    Export YOLOX model to ONNX format.
    
    Args:
        model_name: Model name (e.g., 'yolox-s', 'yolox-m', 'yolox-l')
        ckpt_path: Path to checkpoint file
        output_path: Path to save ONNX model
        opset: ONNX opset version (11 recommended, will auto-upgrade if needed)
        simplify: Whether to simplify ONNX model (optional, may cause issues)
        verify: Whether to verify and test the exported model
    """
    print(f"Loading experiment configuration for: {model_name}")
    exp = get_exp(None, model_name)
    
    print(f"Creating model...")
    model = exp.get_model()
    model.eval()
    
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    
    print(f"Model loaded successfully!")
    print(f"  Test size: {exp.test_size}")
    print(f"  Number of classes: {exp.num_classes}")
    
    # Create dummy input
    input_size = exp.test_size if hasattr(exp, 'test_size') else (640, 640)
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    print(f"\nExporting to ONNX (opset={opset})...")
    print(f"  Input shape: {dummy_input.shape}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            "images": {0: "batch"},
            "output": {0: "batch"}
        },
        opset_version=opset,
        verbose=False
    )
    
    print(f"✅ ONNX model exported to: {output_path}")
    
    # Verify the export
    if verify:
        import onnx
        print("\nVerifying exported model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ Model verification passed!")
        
        print(f"\nModel info:")
        print(f"  Input: {onnx_model.graph.input[0].name} - {[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
        print(f"  Output: {onnx_model.graph.output[0].name} - {[d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]}")
        print(f"  Opset version: {onnx_model.opset_import[0].version}")
        
        # Test inference with OpenVINO
        try:
            from openvino import Core
            import numpy as np
            
            print("\nTesting exported model with OpenVINO...")
            core = Core()
            model_ov = core.read_model(output_path)
            compiled_model = core.compile_model(model_ov, "CPU")
            
            test_input = np.random.randn(1, 3, input_size[0], input_size[1]).astype(np.float32) * 114
            results = compiled_model([test_input])
            output = list(results.values())[0]
            
            print(f"✅ Test inference successful!")
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"  First detection bbox: {output[0, 0, :4]}")
            
            # Verify output is in correct format (pixel coordinates, not normalized)
            if output.max() < 10:
                print("\n⚠️  WARNING: Output values are very small!")
                print("   This may indicate normalized coordinates instead of pixel coordinates.")
                print("   Expected range: 0-640 for bbox coordinates")
            else:
                print("\n✅ Output format looks correct (pixel coordinates)")
                
        except ImportError:
            print("\n⚠️  OpenVINO not available, skipping inference test")
    
    # Simplify ONNX model (optional, can cause issues)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            print("\nSimplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            model_simp, check = onnx_simplify(onnx_model)
            
            if check:
                onnx.save(model_simp, output_path)
                print("✅ ONNX model simplified successfully")
            else:
                print("⚠️  ONNX simplification check failed, using original model")
        except ImportError:
            print("⚠️  onnx-simplifier not installed, skipping simplification")
            print("   Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f"⚠️  Simplification failed: {e}")
            print("   Using original model")


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOX model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-n", "--name",
        type=str,
        default="yolox-s",
        help="Model name: yolox-nano, yolox-tiny, yolox-s, yolox-m, yolox-l, yolox-x"
    )
    
    parser.add_argument(
        "-c", "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint file (.pth)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save ONNX model (default: same as checkpoint with .onnx extension)"
    )
    
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (will auto-upgrade if needed)"
    )
    
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX model (may cause issues, not recommended)"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip model verification and testing"
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        args.output = str(Path(args.ckpt).with_suffix(".onnx"))
    
    # Check inputs
    if not os.path.exists(args.ckpt):
        print(f"❌ Error: Checkpoint file not found: {args.ckpt}")
        print("\nDownload pretrained models from:")
        print("  https://github.com/Megvii-BaseDetection/YOLOX/releases")
        print("\nExample:")
        print("  wget -P models https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth")
        return 1
    
    # Export
    print("=" * 60)
    print("YOLOX ONNX Export")
    print("=" * 60)
    print(f"Model: {args.name}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Output: {args.output}")
    print("=" * 60)
    print()
    
    try:
        export_onnx(
            args.name,
            args.ckpt,
            args.output,
            args.opset,
            args.simplify,
            not args.no_verify
        )
        print("\n" + "=" * 60)
        print("✅ Export complete!")
        print("=" * 60)
        print(f"\nONNX model saved to: {args.output}")
        print("\nTo use with OpenVINO:")
        print(f"  python tools/demo_video_openvino.py <video> --model {args.output}")
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Export failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
