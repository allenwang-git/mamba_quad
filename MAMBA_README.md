# Mamba Replacement for Transformers

This implementation replaces the transformer architectures in the quadruped locomotion system with Mamba (State Space Models) for improved efficiency and linear scaling with sequence length.

## New Files Added

### Core Networks (`torchrl/networks/nets.py`)
- **`MambaTransformer`**: Replaces `Transformer` for vision-only tasks
- **`LocoMamba`**: Replaces `LocoTransformer` for locomotion with state + vision

### Policies (`torchrl/policies/continuous_policy.py`)
- **`GaussianContPolicyMambaTransformer`**: Mamba-based policy for vision-only tasks
- **`GaussianContPolicyLocoMamba`**: Mamba-based policy for locomotion tasks

### Training Scripts (`starter/`)
- **`ppo_locomamba.py`**: Main training entry point (replaces `ppo_locotransformer.py`)
- **`ppo_locomamba_vision_only.py`**: Vision-only training (replaces `ppo_locotransformer_vision_only.py`)

### Testing
- **`test_mamba_replacement.py`**: Comprehensive test suite for Mamba models
- **`configs/example_mamba_config.json`**: Example configuration file

## Usage

### Basic Training

To train with Mamba instead of Transformer:

```bash
# For locomotion with state + vision
python starter/ppo_locomamba.py --config configs/example_mamba_config.json --cuda

# For vision-only
python starter/ppo_locomamba_vision_only.py --config configs/example_mamba_config.json --cuda
```

**⚠️ Important**: Mamba requires CUDA and will automatically enable GPU usage. The training scripts will fail if CUDA is not available.

### Using Existing Transformer Configs

You can use any existing transformer config with Mamba. The script automatically converts `transformer_params` to `mamba_params`:

```bash
# Use existing transformer config directly (automatic conversion)
python starter/ppo_locomamba.py --config config/rl/moving/locotransformer/thin.json --cuda

# Or use the provided Mamba-specific configs
python starter/ppo_locomamba.py --config configs/mamba_thin.json --cuda
```

### Configuration Changes

The main difference in configuration is the replacement of `transformer_params` with `mamba_params`:

**Transformer Config:**
```json
{
  "net": {
    "transformer_params": [[8, 256], [8, 256]]  // [n_heads, dim_feedforward]
  }
}
```

**Mamba Config:**
```json
{
  "net": {
    "mamba_params": [[128, 16, 4, 2], [128, 16, 4, 2]]  // [d_model, d_state, d_conv, expand]
  }
}
```

### Automatic Parameter Conversion

The training scripts automatically convert `transformer_params` to `mamba_params` if found:
- `d_model`: Set to `encoder.visual_dim` (feature dimension)
- `d_state`: Set to 16 (hidden state dimension)
- `d_conv`: Set to 4 (convolution width)
- `expand`: Set to 2 (expansion factor)

## Mamba Parameters Explained

- **`d_model`**: The feature dimension (must match the encoder output dimension)
- **`d_state`**: Hidden state dimension of the SSM (typically 16)
- **`d_conv`**: Width of the causal convolution (typically 4)
- **`expand`**: Expansion factor for the MLP (typically 2)

## Requirements

- **🚨 CUDA Required**: Mamba **requires** CUDA and will not work on CPU-only systems
- **GPU**: CUDA-capable GPU with sufficient memory  
- **mamba-ssm**: Already installed in `environment.yaml`
- **PyTorch with CUDA**: Ensure PyTorch was installed with CUDA support

### CUDA Setup Verification

Before running Mamba training, verify your system meets all requirements:

```bash
python check_mamba_requirements.py
```

This will check PyTorch, CUDA, mamba-ssm installation, and test basic Mamba functionality.

## Testing

Run the test suite to verify the implementation:

```bash
# Full model tests (requires CUDA)
python test_mamba_replacement.py

# Quick config and import tests
python test_mamba_config.py
```

The test suite will:
- Verify that Mamba models can be created and run forward passes
- Check parameter counts and compatibility  
- Test both locomotion and vision-only variants
- Verify configuration file validity
- Gracefully handle systems without CUDA

## Performance Benefits

Mamba offers several advantages over Transformers:

1. **Linear Scaling**: O(n) complexity vs O(n²) for transformers
2. **Memory Efficiency**: Lower memory usage for long sequences
3. **Faster Inference**: Particularly beneficial for long sequences
4. **State Retention**: Better at maintaining long-term dependencies

## Architecture Comparison

| Component | Transformer | Mamba |
|-----------|-------------|-------|
| Attention | Multi-head self-attention | State space model |
| Complexity | O(n²) | O(n) |
| Memory | O(n²) | O(n) |
| Long sequences | Limited by quadratic scaling | Efficient linear scaling |
| State persistence | Through attention | Through hidden states |

## Migration from Transformer

To migrate an existing transformer configuration to Mamba:

1. Replace the training script:
   - `ppo_locotransformer.py` → `ppo_locomamba.py`
   - `ppo_locotransformer_vision_only.py` → `ppo_locomamba_vision_only.py`

2. Update the config (optional, automatic conversion available):
   - Replace `transformer_params` with `mamba_params`
   - Adjust parameters based on your specific needs

3. Ensure CUDA is available for training

The implementation maintains full compatibility with existing configs through automatic parameter conversion.