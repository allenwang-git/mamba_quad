# Mamba

### Metrics on hard terrians

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

  ### Metrics on thin terrians
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
