import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_LOG_LEVEL'] = 'fatal'
import time
import sys
import os.path as osp
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from vision4leg.get_env import get_subprocvec_env, get_vec_env
from torchrl.env import get_vec_env
import random
import gym
from torchrl.collector.on_policy import VecOnPolicyCollector
from torchrl.algo import PPO, VMPO
import torchrl.networks as networks
import torchrl.policies as policies
from torchrl.utils import Logger
from torchrl.replay_buffers.on_policy import OnPolicyReplayBuffer
from torchrl.utils import get_params
from torchrl.utils import get_args
import torch

import warnings
warnings.filterwarnings("ignore")

args = get_args()
params = get_params(args.config)


def experiment(args):

  # Mamba requires CUDA - check availability
  if not torch.cuda.is_available():
    raise RuntimeError(
      "Mamba requires CUDA but CUDA is not available. "
      "Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support installed."
    )
  
  # Force CUDA usage for Mamba
  if not args.cuda:
    print("Warning: Forcing CUDA usage because Mamba requires CUDA")
    args.cuda = True
  
  device = torch.device(
    "cuda:{}".format(args.device) if args.cuda else "cpu")

  env = get_subprocvec_env(
    params["env_name"],
    params["env"],
    args.vec_env_nums,
    args.proc_nums
  )
  eval_env = get_subprocvec_env(
    params["env_name"],
    params["env"],
    max(2, args.vec_env_nums),
    max(2, args.proc_nums),
  )

  if hasattr(env, "_obs_normalizer"):
    eval_env._obs_normalizer = env._obs_normalizer

  env.seed(args.seed)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  buffer_param = params['replay_buffer']

  experiment_name = os.path.split(
    os.path.splitext(args.config)[0])[-1] if args.id is None \
    else args.id
  logger = Logger(
    experiment_name, params['env_name'],
    args.seed, params, args.log_dir, args.overwrite)
  params['general_setting']['env'] = env

  replay_buffer = OnPolicyReplayBuffer(
    env_nums=args.vec_env_nums,
    max_replay_buffer_size=int(buffer_param['size']),
    time_limit_filter=buffer_param['time_limit_filter']
  )
  params['general_setting']['replay_buffer'] = replay_buffer

  params['general_setting']['logger'] = logger
  params['general_setting']['device'] = device

  params['net']['base_type'] = networks.MLPBase
  # params['net']['activation_func'] = torch.nn.Tanh

  encoder = networks.LocoTransformerEncoder(
    in_channels=env.image_channels,
    state_input_dim=env.observation_space.shape[0],
    **params["encoder"]
  )

  # Convert transformer_params to mamba_params if they exist
  net_params = params["net"].copy()
  if "transformer_params" in net_params:
    # Convert transformer parameters to Mamba parameters
    mamba_params = []
    for n_head, dim_feedforward in net_params["transformer_params"]:
      # Map transformer params to mamba params
      # d_model: feature dimension (from encoder)
      # d_state: hidden state dimension (typically 16)
      # d_conv: convolution width (typically 4)
      # expand: expansion factor (typically 2)
      d_model = encoder.visual_dim
      d_state = 16
      d_conv = 4
      expand = 2
      mamba_params.append((d_model, d_state, d_conv, expand))
    
    net_params["mamba_params"] = mamba_params
    del net_params["transformer_params"]
  else:
    # Default Mamba parameters if no transformer_params specified
    net_params["mamba_params"] = [(encoder.visual_dim, 16, 4, 2)]

  pf = policies.GaussianContPolicyLocoMamba(
    encoder=encoder,
    state_input_shape=env.observation_space.shape[0],
    visual_input_shape=(env.image_channels, 64, 64),
    output_shape=env.action_space.shape[0],
    **net_params,
    **params["policy"]
  )

  vf = networks.LocoMamba(
    encoder=encoder,
    state_input_shape=env.observation_space.shape[0],
    visual_input_shape=(env.image_channels, 64, 64),
    output_shape=1,
    **net_params
  )

  # print(pf)
  # print(vf)

  params['general_setting']['collector'] = VecOnPolicyCollector(
    vf, env=env, eval_env=eval_env, pf=pf,
    replay_buffer=replay_buffer, device=device,
    train_render=False,
    **params["collector"]
  )
  params['general_setting']['save_dir'] = osp.join(
    logger.work_dir, "model")
  agent = PPO(
    pf=pf,
    vf=vf,
    **params["ppo"],
    **params["general_setting"]
  )
  agent.train()


if __name__ == "__main__":
  experiment(args)