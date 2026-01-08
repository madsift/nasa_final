import argparse
import torch
import segmentation_models_pytorch as smp
import torch.utils.checkpoint as checkpoint 
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from python.models.models_smp import CraterSMP_3Ch_RAM, CraterSMP

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
    if args.model_type=='CraterSMP_3Ch_RAM':
        model = CraterSMP_3Ch_RAM(
            backbone=args.backbone,
            in_channels=args.im_dim, 
            num_classes=3  # Core, Global, Rim
        )
    elif args.model_type=='CraterSMP':
        model = CraterSMP(
            backbone=args.backbone,
            in_channels=args.im_dim, 
            num_classes=3  # Core, Global, Rim
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)

    model.eval()
    print ('CHEAKPOINT LOADED', model)
    # ---------------------------------------------------------
    # 2. Reparameterize
    # ---------------------------------------------------------
    print("\n--- Reparameterizing ---")

    # Specific check for MobileOne inside SMP
    # The encoder is usually at model.model.encoder
    try:
        encoder = model.model.encoder
    except AttributeError:
        encoder = model.base_model.encoder

    if hasattr(encoder, 'reparameterize'):
        encoder.reparameterize()
        print("✅ Called encoder.reparameterize()")
    else:
        # Fallback traversal
        print("⚠️ Direct encoder method not found, searching modules...")
        found = False
        for module in model.modules():
            if hasattr(module, 'reparameterize'):
                module.reparameterize()
                found = True
        if found:
            print("✅ Called reparameterize() on sub-modules.")
        else:
            print("❌ CRITICAL ERROR: No reparameterize method found! Model cannot be exported efficiently.")

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

    # MobileOne reparameterization FUSES BatchNorms into Convs.
    # Therefore, the Encoder should have EXACTLY 0 BatchNorm layers remaining.
    try:
        encoder = model.model.encoder
    except AttributeError:
        encoder = model.base_model.encoder
    encoder_bn_count = count_bn_layers(encoder)

    if encoder_bn_count == 0:
        print(f"✅ SUCCESS: Encoder contains {encoder_bn_count} BatchNorm layers.")
    else:
        # If this triggers, reparameterization failed or didn't happen recursively
        print(f"❌ FAILURE: Encoder still contains {encoder_bn_count} BatchNorm layers.")
        print("   This means the multi-branch structure is still active.")
        print ('bypass')

    # ---------------------------------------------------------
    # 4. Export to ONNX
    # ---------------------------------------------------------
    print("\n--- Exporting to ONNX ---")
    dummy_input = torch.randn(1, args.im_dim, height, width)  # NCHW: height first
    onnx_path = args.output

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=17,
        do_constant_folding=True, # This helps cleanup any remaining static ops
        dynamic_axes=None 
    )
    print(f"Exported to {onnx_path}")

    # ---------------------------------------------------------
    # 5. STRICT POST-EXPORT VERIFICATION (ONNX Level)
    # ---------------------------------------------------------
    print("\n--- Verifying ONNX Graph ---")

    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph

    # Count Node Types
    node_types = {}
    for node in graph.node:
        node_types[node.op_type] = node_types.get(node.op_type, 0) + 1

    print("ONNX Node Counts:", node_types)

    # CHECK 1: Look for explicit BatchNormalization nodes
    # If reparameterized AND constant_folding=True, there should be 0 BatchNormalization nodes
    # anywhere in the model (Decoder BNs usually get folded into Convs by ONNX export itself).
    bn_count = node_types.get("BatchNormalization", 0)

    if bn_count > 0:
        print(f"⚠️ WARNING: Found {bn_count} BatchNormalization nodes in ONNX.")
        print("   Ideally, this should be 0. If these are in the Decoder, it might be acceptable,")
        print("   but if they are in the Encoder, reparameterization failed.")
    else:
        print("✅ SUCCESS: 0 BatchNormalization nodes found in ONNX.")

    # CHECK 2: Output Consistency Check
    print("\n--- Running Inference Consistency Check ---")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # PyTorch Inference
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    # ONNX Inference
    ort_inputs = {sess.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = sess.run(None, ort_inputs)[0]

    # Compare
    mse = np.mean((torch_out - ort_out) ** 2)
    max_diff = np.max(np.abs(torch_out - ort_out))

    print(f"Mean Squared Error: {mse}")
    print(f"Max Absolute Diff:  {max_diff}")

    if max_diff < 1e-3:
        print("✅ ABSOLUTE CERTAINTY: Model is reparameterized and outputs match.")
    else:
        print("❌ WARNING: Outputs diverge. Check opset version or export settings.")
    
    print(f"\n=== Export Complete: {onnx_path} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CraterSMP model to ONNX with reparameterization")
    
    parser.add_argument("--backbone", type=str, default="mobileone_s2",
                        help="Encoder backbone (e.g., mobileone_s2, resnet34)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--im_dim", type=int, default=3,
                        help="Number of input channels (1=grayscale, 3=RGB+gradient)")
    parser.add_argument("--size", type=str, default="1024",
                        help="Input size: WxH (e.g., 544x416) or single value for square (e.g., 1024)")
    parser.add_argument("--output", type=str, default="model.onnx",
                        help="Output ONNX filename")
    parser.add_argument('--model_type', type=str, default='CraterSMP_3Ch_RAM',
                        help='Model type (CraterSMP_3Ch_RAM or CraterSMP)')
    args = parser.parse_args()
    main(args)

