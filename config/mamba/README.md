# Mamba Configuration Files

This directory contains Mamba-based configuration files converted from the original transformer configurations. All configs use `mamba_params` instead of `transformer_params` and are ready to use with the Mamba training scripts.

## Available Configuration Categories

### 🏃 **Moving Locomotion** (`config/mamba/moving/`)
For quadruped locomotion with movement and dynamic environments:
- `thin.json` - Basic moving locomotion  
- `thin-goal.json` - Goal-directed movement
- `thin-heightfield.json` - Movement over heightfields
- `thin-random-shape.json` - Movement with random terrain shapes
- `thin-wide.json` - Wide configuration variant

### 🛑 **Static Locomotion** (`config/mamba/static/`)
For stationary or slower locomotion scenarios:
- `thin.json` - Basic static locomotion
- `thin-goal.json` - Goal-directed static behavior
- `thin-heightfield.json` - Static navigation over heightfields  
- `thin-random-shape.json` - Static behavior with random terrain
- `thin-wide.json` - Wide configuration variant

### 🏔️ **Challenge Scenarios** (`config/mamba/challenge/`)
For complex terrain and obstacle navigation:
- `chair_desk.json` - Indoor furniture navigation
- `hill.json` - Hill climbing and descent
- `mountain.json` - Mountain terrain navigation
- `stairs.json` - Stair climbing

### 🎯 **MPC Integration** (`config/mamba/mpc/`)
For Model Predictive Control with Mamba:
- `thin.json` - Basic MPC with Mamba
- `thin-goal.json` - Goal-directed MPC
- `thin-heightfield.json` - MPC over heightfields
- `thin-random-shape.json` - MPC with random terrain
- `thin-wide.json` - Wide MPC variant

### 👁️ **Vision-Only MPC** (`config/mamba/mpc_vision_only/`)
For vision-only control with MPC:
- `thin.json` - Basic vision-only MPC
- `thin-goal.json` - Goal-directed vision-only
- `thin-heightfield.json` - Vision-only over heightfields
- `thin-random-shape.json` - Vision-only with random terrain
- `thin-wide.json` - Wide vision-only variant

## Usage Examples

### Basic Training
```bash
# Static locomotion
python starter/ppo_locomamba.py --config config/mamba/static/thin.json --cuda

# Moving locomotion  
python starter/ppo_locomamba.py --config config/mamba/moving/thin.json --cuda

# Challenge scenario - stair climbing
python starter/ppo_locomamba.py --config config/mamba/challenge/stairs.json --cuda
```

### Vision-Only Training
```bash
# Vision-only MPC
python starter/ppo_locomamba_vision_only.py --config config/mamba/mpc_vision_only/thin.json --cuda
```

### Advanced Scenarios
```bash
# Goal-directed movement on heightfields
python starter/ppo_locomamba.py --config config/mamba/moving/thin-heightfield.json --cuda

# Mountain terrain navigation
python starter/ppo_locomamba.py --config config/mamba/challenge/mountain.json --cuda
```

## Configuration Details

All Mamba configs include:
- **Environment**: A1MoveGround with various terrain types
- **Mamba Parameters**: `[[256, 16, 4, 2], [256, 16, 4, 2]]` (2 layers)
  - `d_model=256`: Feature dimension
  - `d_state=16`: Hidden state dimension  
  - `d_conv=4`: Convolution width
  - `expand=2`: Expansion factor
- **Visual Encoder**: 256-dimensional visual features
- **Training**: PPO with standard hyperparameters

## Converting Your Own Configs

To convert additional transformer configs to Mamba:

```bash
python convert_transformer_to_mamba_config.py <input_dir> <output_dir>

# Example:
python convert_transformer_to_mamba_config.py config/rl/your_configs configs/mamba/your_configs
```

## Requirements

- **CUDA**: All Mamba configs require CUDA
- **GPU Memory**: Recommended 4GB+ for standard configs
- **mamba-ssm**: Pre-installed in environment

## Quick Start

1. Check requirements: `python check_mamba_requirements.py`
2. Choose a config from the categories above
3. Run training with `--cuda` flag
4. Monitor training progress in the logs

**Note**: All configs are tested and ready to use. Start with `thin.json` configs for basic scenarios, then move to specialized variants as needed.