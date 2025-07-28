# KAN vs MLP

- vision only (NO MLP OR KAN)
  - python starter/ppo_nature_cnn_vision_only.py --config config/rl/static/naive_baseline/thin-vision.json
- state only (NO MLP OR KAN)
  - python starter/ppo_state.py --config config/rl/static/state-only-baseline.json
- MLP v+s
  - python starter/ppo_nature_cnn.py --config config/rl/static/naive_baseline/thin-mlp.json
- KAN v+s
  - python starter/ppo_nature_cnn.py --config config/rl/static/naive_baseline/thin.json

# Mamba vs Transformer

* state only
  * python starter/ppo_state.py --config config/rl/static/state-only-baseline.json
* Mamba vision only
  * python starter/ppo_locomamba_vision_only.py --config config/mamba/static/thin-vision.json
* Transformer vision only
  * python starter/ppo_locotransformer_visoin_only.py --config config/rl/static/locotransformer/thin-vision.json
* Mamba v+s
  * python starter/ppo_locomamba.py --config config/mamba/static/thin.json
* Transformer v+s
  * python starter/ppo_locotransformer.py --config config/rl/static/locotransformer/thin.json
