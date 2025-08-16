#!/usr/bin/env python3
"""
Check if the system meets the requirements for Mamba training.
"""

import sys

def check_requirements():
    print("🔍 Checking Mamba Requirements...")
    print("=" * 50)
    
    all_good = True
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch: Not installed")
        all_good = False
        return all_good
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA: Available")
        print(f"   - CUDA version: {torch.version.cuda}")
        print(f"   - GPU devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("❌ CUDA: Not available")
        print("   Mamba requires CUDA but PyTorch cannot detect any CUDA devices.")
        all_good = False
    
    # Check mamba-ssm
    try:
        import mamba_ssm
        print(f"✅ mamba-ssm: {mamba_ssm.__version__}")
    except ImportError:
        print("❌ mamba-ssm: Not installed")
        print("   Install with: pip install mamba-ssm")
        all_good = False
    except AttributeError:
        # mamba-ssm might not have __version__
        print("✅ mamba-ssm: Installed (version unknown)")
    
    # Test basic Mamba functionality
    if all_good:
        try:
            from mamba_ssm import Mamba
            print("✅ Mamba import: Success")
            
            # Try creating a small Mamba layer
            mamba_layer = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
            test_input = torch.randn(1, 10, 64).cuda()
            output = mamba_layer(test_input)
            print("✅ Mamba forward pass: Success")
            
        except Exception as e:
            print(f"❌ Mamba functionality test failed: {e}")
            all_good = False
    
    print("=" * 50)
    if all_good:
        print("🎉 All requirements met! You can run Mamba training.")
        print("\nTo start training:")
        print("python starter/ppo_locomamba.py --config your_config.json --cuda")
    else:
        print("💥 Some requirements are missing. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Install CUDA-enabled PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("- Install mamba-ssm: pip install mamba-ssm")
        print("- Verify GPU drivers are installed and working")
    
    return all_good

if __name__ == "__main__":
    success = check_requirements()
    sys.exit(0 if success else 1)