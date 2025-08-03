#!/usr/bin/env python3
"""
Quick test to verify Mamba config and environment setup works.
"""

import sys
import json
import torch

def test_config_loading():
    """Test that configs load properly."""
    print("🧪 Testing Mamba configuration loading...")
    
    # Test sample configs from different categories
    configs_to_test = [
        "configs/example_mamba_config.json",
        "config/mamba/static/thin.json",
        "config/mamba/moving/thin.json",
        "config/mamba/challenge/stairs.json"
    ]
    
    for config_path in configs_to_test:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check required fields
            assert "env_name" in config
            assert "env" in config
            assert "env_build" in config["env"]  # This was the issue!
            assert "net" in config
            assert "mamba_params" in config["net"]
            
            print(f"✅ {config_path} - Valid")
            print(f"   Environment: {config['env_name']}")
            print(f"   Mamba layers: {len(config['net']['mamba_params'])}")
            
        except Exception as e:
            print(f"❌ {config_path} - Failed: {e}")
            return False
    
    return True

def test_mamba_import():
    """Test that Mamba components can be imported."""
    print("\n🧪 Testing Mamba component imports...")
    
    try:
        import torchrl.networks as networks
        import torchrl.policies as policies
        
        # Test that our new classes exist
        assert hasattr(networks, 'LocoMamba')
        assert hasattr(networks, 'MambaTransformer')
        assert hasattr(policies, 'GaussianContPolicyLocoMamba')
        assert hasattr(policies, 'GaussianContPolicyMambaTransformer')
        
        print("✅ All Mamba classes are available")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    print("🔧 Mamba Configuration Test")
    print("=" * 40)
    
    success = True
    success &= test_config_loading()
    success &= test_mamba_import()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! Your Mamba setup is ready.")
        print("\nYou can now run:")
        print("python starter/ppo_locomamba.py --config config/mamba/static/thin.json --cuda")
    else:
        print("❌ Some tests failed. Check the issues above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)