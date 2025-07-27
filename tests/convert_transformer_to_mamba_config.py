#!/usr/bin/env python3
"""
Script to convert transformer configs to Mamba configs.
"""

import json
import os
import sys
from pathlib import Path

def convert_transformer_params_to_mamba(transformer_params, visual_dim):
    """Convert transformer_params to mamba_params."""
    mamba_params = []
    for n_head, dim_feedforward in transformer_params:
        # Convert transformer params to mamba params
        # d_model: feature dimension (use visual_dim from encoder)
        # d_state: hidden state dimension (typically 16)
        # d_conv: convolution width (typically 4)
        # expand: expansion factor (typically 2)
        d_model = visual_dim
        d_state = 16
        d_conv = 4
        expand = 2
        mamba_params.append([d_model, d_state, d_conv, expand])
    return mamba_params

def convert_config_file(input_path, output_path):
    """Convert a single transformer config to Mamba config."""
    print(f"Converting {input_path} -> {output_path}")
    
    # Load the transformer config
    with open(input_path, 'r') as f:
        config = json.load(f)
    
    # Check if it has transformer_params
    if "net" in config and "transformer_params" in config["net"]:
        # Get visual_dim from encoder
        visual_dim = config.get("encoder", {}).get("visual_dim", 256)
        
        # Convert transformer_params to mamba_params
        transformer_params = config["net"]["transformer_params"]
        mamba_params = convert_transformer_params_to_mamba(transformer_params, visual_dim)
        
        # Replace transformer_params with mamba_params
        config["net"]["mamba_params"] = mamba_params
        del config["net"]["transformer_params"]
        
        print(f"  Converted {len(transformer_params)} transformer layers to {len(mamba_params)} Mamba layers")
        print(f"  Visual dim: {visual_dim}")
    else:
        print(f"  No transformer_params found in {input_path}")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the Mamba config
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"  ✅ Saved to {output_path}")
    return True

def convert_directory(input_dir, output_dir, recursive=True):
    """Convert all transformer configs in a directory to Mamba configs."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return False
    
    converted_count = 0
    total_count = 0
    
    # Find all JSON files
    pattern = "**/*.json" if recursive else "*.json"
    for json_file in input_path.glob(pattern):
        total_count += 1
        
        # Create corresponding output path
        relative_path = json_file.relative_to(input_path)
        output_file = output_path / relative_path
        
        try:
            if convert_config_file(str(json_file), str(output_file)):
                converted_count += 1
        except Exception as e:
            print(f"  ❌ Error converting {json_file}: {e}")
    
    print(f"\nConversion complete: {converted_count}/{total_count} files converted")
    return converted_count > 0

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_transformer_to_mamba_config.py <input_dir> <output_dir>")
        print("Example: python convert_transformer_to_mamba_config.py config/rl/static/locotransformer configs/mamba/static")
        return 1
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    print(f"🔄 Converting Transformer configs to Mamba configs")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    
    success = convert_directory(input_dir, output_dir)
    
    if success:
        print(f"\n🎉 Conversion successful! Mamba configs saved to {output_dir}")
        print("\nYou can now train with:")
        print(f"python starter/ppo_locomamba.py --config {output_dir}/thin.json --cuda")
        return 0
    else:
        print(f"\n❌ Conversion failed or no configs converted")
        return 1

if __name__ == "__main__":
    sys.exit(main())