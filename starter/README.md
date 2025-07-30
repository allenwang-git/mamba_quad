# Mamba vs Transformer

- state only
  - python starter/ppo_state.py --config config/rl/static/state-only-baseline.json
- Mamba vision only
  - python starter/ppo_locomamba_vision_only.py --config config/mamba/static/thin-vision.json
- Transformer vision only
  - python starter/ppo_locotransformer_visoin_only.py --config config/rl/static/locotransformer/thin-vision.json
- Mamba v+s
  - python starter/ppo_locomamba.py --config config/mamba/static/thin.json
- Transformer v+s
  - python starter/ppo_locotransformer.py --config config/rl/static/locotransformer/thin.json

## Metrics on hard terrians

- state only

  - python starter/total_randomize_statistics.py   --log_dir ../logmamba/logs/state-only-baseline --env_name A1MoveGround --config config/rl/static/state-only-baseline-obs.json --add_tag state-only --seed 0 1 2 3 4 5 6 7 8 9

- tf-vision-only

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/transformer-vision-only     --env_name A1MoveGround --config config/rl/static/locotransformer/thin-heightfield-vision.json --add_tag tf-vision-only --seed 0 1 2 3 4 5 6 7 8 9

- tf-vision-state

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/transformer-vision-state     --env_name A1MoveGround --config config/rl/static/locotransformer/thin-heightfield.json --add_tag tf-vision-state --seed 0 1 2 3 4 5 6 7 8 9

- mamba-vision-only

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/mamba-vision-only     --env_name A1MoveGround --config config/mamba/static/thin-heightfield-vision.json --add_tag mamba-vision-only --seed 0 1 2 3 4 5 6 7 8 9

- mamba-vision-state

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/mamba-vision-state     --env_name A1MoveGround --config config/mamba/static/thin-heightfield.json --add_tag mamba-vision-state --seed 0 1 2 3 4 5 6 7 8 9

## Metrics on trained terrians
- state only

  - python starter/total_randomize_statistics.py   --log_dir ../logmamba/logs/state-only-baseline --env_name A1MoveGround --config config/rl/static/state-only-baseline.json --add_tag state-only --seed 0 1 2 3 4 5 6 7 8 9

- tf-vision-only

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/transformer-vision-only     --env_name A1MoveGround --config config/rl/static/locotransformer/thin-vision.json --add_tag tf-vision-only --seed 0 1 2 3 4 5 6 7 8 9

- tf-vision-state

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/transformer-vision-state     --env_name A1MoveGround --config config/rl/static/locotransformer/thin.json --add_tag tf-vision-state --seed 0 1 2 3 4 5 6 7 8 9

- mamba-vision-only

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/mamba-vision-only     --env_name A1MoveGround --config config/mamba/static/thin-vision.json --add_tag mamba-vision-only --seed 0 1 2 3 4 5 6 7 8 9

- mamba-vision-state

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/mamba-vision-state     --env_name A1MoveGround --config config/mamba/static/thin.json --add_tag mamba-vision-state --seed 0 1 2 3 4 5 6 7 8 9

# KAN vs MLP

- vision only
  - python starter/ppo_nature_cnn_vision_only.py --config config/rl/static/naive_baseline/thin-vision.json
  - python starter/ppo_nature_cnn_vision_only.py --config config/rl/static/naive_baseline/thin-vision-mlp.json
- state only (NO MLP OR KAN)
  - python starter/ppo_state.py --config config/rl/static/state-only-baseline.json
- MLP v+s
  - python starter/ppo_nature_cnn.py --config config/rl/static/naive_baseline/thin-mlp.json
- KAN v+s
  - python starter/ppo_nature_cnn.py --config config/rl/static/naive_baseline/thin.json

## Metrics on hard terrians

- state only

  - python starter/total_randomize_statistics.py   --log_dir ../logmamba/logs/state-only-baseline --env_name A1MoveGround --config config/rl/static/state-only-baseline-obs.json --add_tag state-only --seed 0 1 2 3 4 5 6 7 8 9
- vision-only MLP

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/transformer-vision-only     --env_name A1MoveGround --config config/rl/static/locotransformer/thin-heightfield-vision.json --add_tag tf-vision-only --seed 0 1 2 3 4 5 6 7 8 9
- vision-state MLP

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/transformer-vision-state     --env_name A1MoveGround --config config/rl/static/locotransformer/thin-heightfield.json --add_tag tf-vision-state --seed 0 1 2 3 4 5 6 7 8 9

- vision-only KAN

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/mamba-vision-only     --env_name A1MoveGround --config config/mamba/static/thin-heightfield-vision.json --add_tag mamba-vision-only --seed 0 1 2 3 4 5 6 7 8 9

- vision-state KAN

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/mamba-vision-state     --env_name A1MoveGround --config config/mamba/static/thin-heightfield.json --add_tag mamba-vision-state --seed 0 1 2 3 4 5 6 7 8 9

## Metrics on trained terrians
- state only

  - python starter/total_randomize_statistics.py   --log_dir ../logmamba/logs/state-only-baseline --env_name A1MoveGround --config config/rl/static/state-only-baseline.json --add_tag state-only --seed 0 1 2 3 4 5 6 7 8 9

- vision-only MLP

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/transformer-vision-only     --env_name A1MoveGround --config config/rl/static/locotransformer/thin-vision.json --add_tag tf-vision-only --seed 0 1 2 3 4 5 6 7 8 9

- vision-state MLP

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/transformer-vision-state     --env_name A1MoveGround --config config/rl/static/locotransformer/thin.json --add_tag tf-vision-state --seed 0 1 2 3 4 5 6 7 8 9

- vision-only KAN

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/mamba-vision-only     --env_name A1MoveGround --config config/mamba/static/thin-vision.json --add_tag mamba-vision-only --seed 0 1 2 3 4 5 6 7 8 9

- vision-state KAN

  - python starter/total_randomize_statistics.py  --log_dir ../logmamba/logs/mamba-vision-state     --env_name A1MoveGround --config config/mamba/static/thin.json --add_tag mamba-vision-state --seed 0 1 2 3 4 5 6 7 8 9
