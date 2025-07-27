# 🦌 Complete Mamba Configuration Collection

I've successfully generated a comprehensive collection of Mamba configurations by converting all existing transformer configs. Here's what's available:

## 📁 **Configuration Structure**

All Mamba configs are located in: `/config/mamba/`

```
config/mamba/
├── static/           # Static locomotion scenarios
├── moving/           # Dynamic movement scenarios  
├── challenge/        # Complex terrain challenges
├── mpc/             # Model Predictive Control
├── mpc_vision_only/ # Vision-only MPC
└── README.md        # Detailed documentation
```

## 🎯 **Available Configurations (24 total)**

### **Static Locomotion** (5 configs)
```bash
config/mamba/static/thin.json                    # Basic static locomotion
config/mamba/static/thin-goal.json               # Goal-directed static
config/mamba/static/thin-heightfield.json        # Static on heightfields
config/mamba/static/thin-random-shape.json       # Static with random terrain
config/mamba/static/thin-wide.json               # Wide static variant
```

### **Moving Locomotion** (5 configs)  
```bash
config/mamba/moving/thin.json                    # Basic moving locomotion
config/mamba/moving/thin-goal.json               # Goal-directed movement
config/mamba/moving/thin-heightfield.json        # Movement on heightfields
config/mamba/moving/thin-random-shape.json       # Movement with random terrain
config/mamba/moving/thin-wide.json               # Wide moving variant
```

### **Challenge Scenarios** (4 configs)
```bash
config/mamba/challenge/stairs.json               # Stair climbing
config/mamba/challenge/hill.json                 # Hill terrain
config/mamba/challenge/mountain.json             # Mountain navigation
config/mamba/challenge/chair_desk.json           # Indoor furniture
```

### **MPC Integration** (5 configs)
```bash
config/mamba/mpc/thin.json                       # Basic MPC with Mamba
config/mamba/mpc/thin-goal.json                  # Goal-directed MPC
config/mamba/mpc/thin-heightfield.json           # MPC on heightfields
config/mamba/mpc/thin-random-shape.json          # MPC with random terrain
config/mamba/mpc/thin-wide.json                  # Wide MPC variant
```

### **Vision-Only MPC** (5 configs)
```bash
config/mamba/mpc_vision_only/thin.json           # Basic vision-only MPC
config/mamba/mpc_vision_only/thin-goal.json      # Goal-directed vision-only
config/mamba/mpc_vision_only/thin-heightfield.json # Vision-only on heightfields
config/mamba/mpc_vision_only/thin-random-shape.json # Vision-only random terrain
config/mamba/mpc_vision_only/thin-wide.json      # Wide vision-only variant
```

## 🚀 **Quick Start Examples**

### Basic Training
```bash
# Start with basic static locomotion
python starter/ppo_locomamba.py --config config/mamba/static/thin.json --cuda

# Try moving locomotion
python starter/ppo_locomamba.py --config config/mamba/moving/thin.json --cuda

# Challenge yourself with stairs
python starter/ppo_locomamba.py --config config/mamba/challenge/stairs.json --cuda
```

### Vision-Only Training
```bash
# Vision-only locomotion
python starter/ppo_locomamba_vision_only.py --config config/mamba/mpc_vision_only/thin.json --cuda
```

### Advanced Scenarios
```bash
# Goal-directed movement on complex terrain
python starter/ppo_locomamba.py --config config/mamba/moving/thin-heightfield.json --cuda

# Mountain terrain navigation
python starter/ppo_locomamba.py --config config/mamba/challenge/mountain.json --cuda

# Indoor furniture navigation
python starter/ppo_locomamba.py --config config/mamba/challenge/chair_desk.json --cuda
```

## ⚙️ **Technical Details**

### Mamba Parameters (All Configs)
- **Layers**: 2 Mamba layers per config
- **d_model**: 256 (feature dimension)
- **d_state**: 16 (hidden state dimension)
- **d_conv**: 4 (convolution width)  
- **expand**: 2 (expansion factor)

### Environment Details
- **Base Environment**: A1MoveGround
- **Visual Features**: 256-dimensional
- **Training Algorithm**: PPO
- **Image Size**: 64x64
- **Action Space**: 12 DOF quadruped control

## 🛠️ **Tools Provided**

### Conversion Tool
```bash
# Convert additional transformer configs to Mamba
python tests/convert_transformer_to_mamba_config.py <input_dir> <output_dir>
```

### Testing Tools
```bash
# Check system requirements
python tests/check_mamba_requirements.py

# Test configurations
python tests/test_mamba_config.py

# Full model tests  
python tests/test_mamba_replacement.py
```

## 📊 **Configuration Categories by Use Case**

| **Use Case** | **Recommended Config** | **Description** |
|--------------|------------------------|-----------------|
| **Learning Basics** | `static/thin.json` | Start here for basic quadruped locomotion |
| **Dynamic Movement** | `moving/thin.json` | Moving locomotion with target velocity |
| **Complex Terrain** | `challenge/stairs.json` | Stair climbing challenge |
| **Goal Navigation** | `moving/thin-goal.json` | Goal-directed movement |
| **Vision-Only** | `mpc_vision_only/thin.json` | Pure vision-based control |
| **Heightfield Navigation** | `moving/thin-heightfield.json` | Rough terrain locomotion |
| **Indoor Navigation** | `challenge/chair_desk.json` | Furniture obstacle avoidance |
| **Extreme Terrain** | `challenge/mountain.json` | Mountain terrain navigation |

## ✅ **Verification Status**

- ✅ **24 configs generated** from transformer sources
- ✅ **All configs tested** and validated
- ✅ **Proper format** with `mamba_params` instead of `transformer_params`
- ✅ **Ready to use** with existing training scripts
- ✅ **Comprehensive documentation** provided

## 🎯 **Next Steps**

1. **Choose a config** based on your use case
2. **Verify requirements**: `python tests/check_mamba_requirements.py`
3. **Start training**: `python starter/ppo_locomamba.py --config <your_config> --cuda`
4. **Monitor progress** in the log directory
5. **Experiment** with different scenarios

You now have a complete collection of Mamba configurations covering every major locomotion scenario! 🎉