#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

# Import our networks
import torchrl.networks as networks
import torchrl.policies as policies

def test_mamba_models():
    """Test that Mamba models can be created and run forward passes."""
    print("Testing Mamba replacement models...")
    
    # Check if CUDA is available for Mamba
    if not torch.cuda.is_available():
        print("  ⚠️  CUDA not available, Mamba requires CUDA. Skipping Mamba tests.")
        return True
    
    device = torch.device("cuda")
    print(f"  Using device: {device}")
    
    # Test parameters
    batch_size = 4
    state_input_dim = 48
    visual_channels = 16
    visual_size = 64
    action_dim = 12
    token_dim = 64
    
    # Create encoder
    encoder = networks.LocoTransformerEncoder(
        in_channels=visual_channels,
        state_input_dim=state_input_dim,
        hidden_shapes=[256, 256],
        token_dim=token_dim,
        visual_dim=128,
    ).to(device)
    
    # Test data
    visual_input = torch.randn(batch_size, visual_channels, visual_size, visual_size).to(device)
    state_input = torch.randn(batch_size, state_input_dim).to(device)
    full_input = torch.cat([state_input, visual_input.flatten(start_dim=1)], dim=-1)
    
    print(f"Input shapes:")
    print(f"  State: {state_input.shape}")
    print(f"  Visual: {visual_input.shape}")
    print(f"  Combined: {full_input.shape}")
    
    # Mamba parameters (d_model, d_state, d_conv, expand)
    mamba_params = [(encoder.visual_dim, 16, 4, 2)]
    
    try:
        # Test LocoMamba network
        print("\nTesting LocoMamba network...")
        loco_mamba = networks.LocoMamba(
            encoder=encoder,
            output_shape=1,  # For value function
            state_input_shape=state_input_dim,
            visual_input_shape=(visual_channels, visual_size, visual_size),
            mamba_params=mamba_params,
            append_hidden_shapes=[256],
        ).to(device)
        
        value_output = loco_mamba(full_input)
        print(f"  LocoMamba output shape: {value_output.shape}")
        assert value_output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {value_output.shape}"
        print("  ✓ LocoMamba network test passed!")
        
    except Exception as e:
        print(f"  ✗ LocoMamba network test failed: {e}")
        return False
    
    try:
        # Test LocoMamba policy
        print("\nTesting GaussianContPolicyLocoMamba...")
        policy = policies.GaussianContPolicyLocoMamba(
            encoder=encoder,
            output_shape=action_dim,
            state_input_shape=state_input_dim,
            visual_input_shape=(visual_channels, visual_size, visual_size),
            mamba_params=mamba_params,
            append_hidden_shapes=[256],
        ).to(device)
        
        mean, std, log_std = policy(full_input)
        print(f"  Policy output shapes:")
        print(f"    Mean: {mean.shape}")
        print(f"    Std: {std.shape}")
        print(f"    Log std: {log_std.shape}")
        
        assert mean.shape == (batch_size, action_dim), f"Expected mean shape ({batch_size}, {action_dim}), got {mean.shape}"
        assert std.shape == (batch_size, action_dim), f"Expected std shape ({batch_size}, {action_dim}), got {std.shape}"
        print("  ✓ GaussianContPolicyLocoMamba test passed!")
        
    except Exception as e:
        print(f"  ✗ GaussianContPolicyLocoMamba test failed: {e}")
        return False
    
    try:
        # Test MambaTransformer (vision-only)
        print("\nTesting MambaTransformer (vision-only)...")
        visual_encoder = networks.TransformerEncoder(
            in_channels=visual_channels,
            token_dim=token_dim,
        ).to(device)
        
        mamba_transformer = networks.MambaTransformer(
            encoder=visual_encoder,
            output_shape=action_dim,
            visual_input_shape=(visual_channels, visual_size, visual_size),
            mamba_params=mamba_params,
            append_hidden_shapes=[256],
        ).to(device)
        
        visual_flat = visual_input.flatten(start_dim=1)
        action_output = mamba_transformer(visual_flat)
        print(f"  MambaTransformer output shape: {action_output.shape}")
        assert action_output.shape == (batch_size, action_dim), f"Expected shape ({batch_size}, {action_dim}), got {action_output.shape}"
        print("  ✓ MambaTransformer test passed!")
        
    except Exception as e:
        print(f"  ✗ MambaTransformer test failed: {e}")
        return False
    
    print("\n🎉 All Mamba model tests passed successfully!")
    return True


def test_parameter_compatibility():
    """Test that Mamba models have reasonable parameter counts."""
    print("\nTesting parameter compatibility...")
    
    # Check if CUDA is available for Mamba
    if not torch.cuda.is_available():
        print("  ⚠️  CUDA not available, Mamba requires CUDA. Skipping parameter tests.")
        return True
    
    device = torch.device("cuda")
    
    state_input_dim = 48
    visual_channels = 16
    visual_size = 64
    action_dim = 12
    token_dim = 64
    
    # Create encoder
    encoder = networks.LocoTransformerEncoder(
        in_channels=visual_channels,
        state_input_dim=state_input_dim,
        hidden_shapes=[256, 256],
        token_dim=token_dim,
        visual_dim=128,
    ).to(device)
    
    mamba_params = [(encoder.visual_dim, 16, 4, 2)]
    
    # Create Mamba policy
    mamba_policy = policies.GaussianContPolicyLocoMamba(
        encoder=encoder,
        output_shape=action_dim,
        state_input_shape=state_input_dim,
        visual_input_shape=(visual_channels, visual_size, visual_size),
        mamba_params=mamba_params,
        append_hidden_shapes=[256],
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in mamba_policy.parameters())
    trainable_params = sum(p.numel() for p in mamba_policy.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Check that we have a reasonable number of parameters
    assert total_params > 0, "Model should have parameters"
    assert trainable_params == total_params, "All parameters should be trainable by default"
    
    print("  ✓ Parameter compatibility test passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Mamba Replacement Test Suite")
    print("=" * 60)
    
    try:
        success = True
        success &= test_mamba_models()
        success &= test_parameter_compatibility()
        
        if success:
            print("\n🎉 All tests passed! Mamba replacement is working correctly.")
            exit(0)
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            exit(1)
            
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)