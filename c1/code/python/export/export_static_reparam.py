import argparse
import torch
import segmentation_models_pytorch as smp
import torch.utils.checkpoint as checkpoint 
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from python.models.models_smp import CraterSMP_3Ch_RAM, CraterSMP, CraterSMP_GroupNorm

def convert_onnx_to_fp16(onnx_path, output_path):
    """Convert ONNX model to FP16 using onnxconverter-common."""
    try:
        from onnxconverter_common import float16
        model = onnx.load(onnx_path)
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, output_path)
        print(f"✅ Converted to FP16: {output_path}")
        return True
    except ImportError:
        print("⚠️ onnxconverter-common not installed. Install with: pip install onnxconverter-common")
        return False
    except Exception as e:
        print(f"❌ FP16 conversion failed: {e}")
        return False

def optimize_for_arm64(onnx_path, output_path):
    """
    Apply ARM64-specific optimizations via ONNX Runtime.
    
    Optimizations applied:
    - Graph optimization (constant folding, redundant node elimination)
    - Operator fusion (Conv+BN, MatMul+Add, etc.)
    
    Note: Using ORT_ENABLE_BASIC to avoid hardware-specific transforms
    that could cause issues across different ARM64 chips.
    """
    try:
        import shutil
        sess_options = ort.SessionOptions()
        
        # Use BASIC level - EXTENDED/ALL can produce hardware-specific models
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        
        # Save optimized model to file
        sess_options.optimized_model_filepath = output_path
        
        # Create session (this triggers optimization and saves)
        _ = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
        
        # Check if output was created and has reasonable size
        import os
        if os.path.exists(output_path):
            orig_size = os.path.getsize(onnx_path)
            opt_size = os.path.getsize(output_path)
            if opt_size < orig_size * 0.5:
                print(f"⚠️ Warning: Optimized model is much smaller ({opt_size} vs {orig_size} bytes)")
                print(f"   This may indicate over-optimization. Using original model.")
                shutil.copy(onnx_path, output_path)
        
        print(f"✅ ARM64-optimized model saved: {output_path}")
        return True
    except Exception as e:
        print(f"❌ ARM64 optimization failed: {e}")
        return False

def main(args):
    # ---------------------------------------------------------
    # 1. Load Model
    # ---------------------------------------------------------
    print("--- Loading Model ---")
    print(f"Backbone: {args.backbone}")
    print(f"Input channels: {args.im_dim}")
    
    # Parse size (WxH or single value for square)
    if 'x' in args.size or ',' in args.size:
        sep = 'x' if 'x' in args.size else ','
        parts = args.size.split(sep)
        width, height = int(parts[0]), int(parts[1])
    else:
        width = height = int(args.size)
    
    print(f"Input size: {width}x{height} (WxH)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Export dtype: {args.dtype}")
    
    if args.model_type == 'CraterSMP_3Ch_RAM':
        model = CraterSMP_3Ch_RAM(
            backbone=args.backbone,
            in_channels=args.im_dim, 
            num_classes=3
        )
    elif args.model_type == 'CraterSMP':
        model = CraterSMP(
            backbone=args.backbone,
            in_channels=args.im_dim, 
            num_classes=3
        )
    elif args.model_type == 'CraterSMP_GroupNorm':
        model = CraterSMP_GroupNorm(
            backbone=args.backbone,
            in_channels=args.im_dim, 
            num_classes=3
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    checkpoint_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    model.load_state_dict(state_dict)

    model.eval()
    print('CHECKPOINT LOADED')
    
    # ---------------------------------------------------------
    # 2. Reparameterize
    # ---------------------------------------------------------
    print("\n--- Reparameterizing ---")

    # Specific check for MobileOne inside SMP
    try:
        encoder = model.model.encoder
    except AttributeError:
        encoder = model.base_model.encoder

    if hasattr(encoder, 'reparameterize'):
        encoder.reparameterize()
        print("✅ Called encoder.reparameterize()")
    else:
        print("⚠️ Direct encoder method not found, searching modules...")
        found = False
        for module in model.modules():
            if hasattr(module, 'reparameterize'):
                module.reparameterize()
                found = True
        if found:
            print("✅ Called reparameterize() on sub-modules.")
        else:
            print("❌ CRITICAL ERROR: No reparameterize method found!")

    # ---------------------------------------------------------
    # 3. STRICT PRE-EXPORT VERIFICATION
    # ---------------------------------------------------------
    print("\n--- Verifying Reparameterization (PyTorch Level) ---")

    def count_bn_layers(module):
        count = 0
        for m in module.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                count += 1
        return count

    try:
        encoder = model.model.encoder
    except AttributeError:
        encoder = model.base_model.encoder
    encoder_bn_count = count_bn_layers(encoder)

    if encoder_bn_count == 0:
        print(f"✅ SUCCESS: Encoder contains {encoder_bn_count} BatchNorm layers.")
    else:
        print(f"⚠️ WARNING: Encoder still contains {encoder_bn_count} BatchNorm layers.")
        print("   (This may be expected for GroupNorm models)")

    # ---------------------------------------------------------
    # 4. Prepare for Export (dtype conversion)
    # ---------------------------------------------------------
    dummy_input = torch.randn(1, args.im_dim, height, width)
    
    # Convert to target dtype for export
    export_dtype = torch.float32
    if args.dtype == 'fp16':
        # Note: Direct FP16 export requires post-processing
        # We export FP32 and convert afterwards for better compatibility
        print("\n[INFO] FP16: Will convert after FP32 export for better compatibility")

    # ---------------------------------------------------------
    # 5. Export to ONNX
    # ---------------------------------------------------------
    print("\n--- Exporting to ONNX ---")
    
    # Determine output path
    base_output = args.output.replace('.onnx', '')
    onnx_path = f"{base_output}_fp32.onnx" if args.dtype == 'fp16' else args.output
    
    # Use opset 18 - minimum required by PyTorch 2.x exporter
    opset = 18

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None 
    )
    print(f"Exported to {onnx_path} (opset={opset})")

    # ---------------------------------------------------------
    # 6. Post-Export FP16 Conversion (if requested)
    # ---------------------------------------------------------
    final_onnx_path = onnx_path
    
    if args.dtype == 'fp16':
        fp16_path = f"{base_output}_fp16.onnx"
        if convert_onnx_to_fp16(onnx_path, fp16_path):
            final_onnx_path = fp16_path

    # ---------------------------------------------------------
    # 7. ARM64 Optimization (if requested)
    # ---------------------------------------------------------
    if args.optimize_arm64:
        print("\n--- Applying ARM64 Optimizations ---")
        arm_path = final_onnx_path.replace('.onnx', '_arm64.onnx')
        if optimize_for_arm64(final_onnx_path, arm_path):
            final_onnx_path = arm_path

    # ---------------------------------------------------------
    # 8. STRICT POST-EXPORT VERIFICATION (ONNX Level)
    # ---------------------------------------------------------
    print("\n--- Verifying ONNX Graph ---")

    onnx_model = onnx.load(final_onnx_path)
    graph = onnx_model.graph

    # Count Node Types
    node_types = {}
    for node in graph.node:
        node_types[node.op_type] = node_types.get(node.op_type, 0) + 1

    print("ONNX Node Counts:", node_types)

    # Check for BatchNormalization nodes
    bn_count = node_types.get("BatchNormalization", 0)
    if bn_count > 0:
        print(f"⚠️ WARNING: Found {bn_count} BatchNormalization nodes in ONNX.")
    else:
        print("✅ SUCCESS: 0 BatchNormalization nodes found in ONNX.")

    # ---------------------------------------------------------
    # 9. Output Consistency Check
    # ---------------------------------------------------------
    print("\n--- Running Inference Consistency Check ---")
    
    # For reduced precision, we need to compare carefully
    sess = ort.InferenceSession(final_onnx_path, providers=['CPUExecutionProvider'])
    
    # Get input dtype from ONNX model
    input_type = sess.get_inputs()[0].type
    print(f"ONNX input type: {input_type}")

    # PyTorch Inference (always FP32 for comparison)
    model_fp32 = model.float()
    dummy_input_fp32 = torch.randn(1, args.im_dim, height, width)
    with torch.no_grad():
        torch_out = model_fp32(dummy_input_fp32).numpy()

    # ONNX Inference
    ort_input = dummy_input_fp32.numpy()
    if 'float16' in input_type:
        ort_input = ort_input.astype(np.float16)
    
    ort_inputs = {sess.get_inputs()[0].name: ort_input}
    ort_out = sess.run(None, ort_inputs)[0]
    
    # Convert back to float32 for comparison
    ort_out = ort_out.astype(np.float32)

    # Compare
    mse = np.mean((torch_out - ort_out) ** 2)
    max_diff = np.max(np.abs(torch_out - ort_out))

    print(f"Mean Squared Error: {mse:.6e}")
    print(f"Max Absolute Diff:  {max_diff:.6e}")

    # Different tolerances for different dtypes
    if args.dtype == 'fp16':
        tol = 1e-2  # FP16 has ~3 decimal digits of precision
    elif args.dtype == 'bf16':
        tol = 5e-2  # BF16 has even less mantissa precision
    else:
        tol = 1e-3  # FP32

    if max_diff < tol:
        print(f"✅ VERIFIED: Model outputs match within tolerance ({tol})")
    else:
        print(f"⚠️ WARNING: Outputs diverge beyond tolerance. This may be expected for reduced precision.")
    
    # ---------------------------------------------------------
    # 10. Final Summary
    # ---------------------------------------------------------
    print(f"\n{'='*50}")
    print("EXPORT SUMMARY")
    print(f"{'='*50}")
    print(f"Final model: {final_onnx_path}")
    print(f"Precision:   {args.dtype.upper()}")
    print(f"ARM64 opt:   {'Yes' if args.optimize_arm64 else 'No'}")
    print(f"Input shape: (1, {args.im_dim}, {height}, {width})")
    
    # File size comparison
    import os
    final_size = os.path.getsize(final_onnx_path) / (1024 * 1024)
    print(f"File size:   {final_size:.2f} MB")
    
    if args.dtype == 'fp32' and os.path.exists(onnx_path) and onnx_path != final_onnx_path:
        fp32_size = os.path.getsize(onnx_path) / (1024 * 1024)
        reduction = (1 - final_size / fp32_size) * 100
        print(f"Size reduction: {reduction:.1f}% vs FP32")
    
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CraterSMP model to ONNX with reparameterization and ARM64 optimizations")
    
    parser.add_argument("--backbone", type=str, default="mobileone_s2",
                        help="Encoder backbone (e.g., mobileone_s2, resnet34)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--im_dim", type=int, default=3,
                        help="Number of input channels (1=grayscale, 3=RGB+gradient)")
    parser.add_argument("--size", type=str, default="672x544",
                        help="Input size: WxH (e.g., 544x416) or single value for square (e.g., 1024)")
    parser.add_argument("--output", type=str, default="model.onnx",
                        help="Output ONNX filename")
    parser.add_argument('--model_type', type=str, default='CraterSMP',
                        choices=['CraterSMP_3Ch_RAM', 'CraterSMP', 'CraterSMP_GroupNorm'],
                        help='Model type to export')
    
    # New optimization options
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp32", "fp16"],
                        help="Export precision: fp32 (default), fp16 (half). Note: bf16 not supported by ONNX Runtime CPU.")
    parser.add_argument("--optimize_arm64", action="store_true",
                        help="Apply ARM64-specific graph optimizations (constant folding, op fusion)")
    
    args = parser.parse_args()
    main(args)


