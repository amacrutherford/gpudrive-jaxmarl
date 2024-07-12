import os
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
# import hydra
# from omegaconf import DictConfig, OmegaConf

# from jaxmarl.wrappers.baselines import SMAXLogWrapper
# from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

# import wandb
# import functools
# import matplotlib.pyplot as plt

# import torch
# import imageio
# from IPython.display import HTML, Image
print('starting')
# Set working directory to the base directory 'gpudrive'
# working_dir = Path.cwd()
# while working_dir.name != 'gpudrive':
#     working_dir = working_dir.parent
#     if working_dir == Path.home():
#         raise FileNotFoundError("Base directory 'gpudrive' not found")
# os.chdir(working_dir)

from pygpudrive.env.config import EnvConfig, RenderConfig
from pygpudrive.env.env_jax import GPUDriveJaxEnv
from pygpudrive.env.wrappers.jaxmarl_wrapper import GPUDriveToJaxMARL

import jax 

# INITALISE ENVIRONMENT
EPISODE_LENGTH = 91  # Number of steps in each episode
MAX_NUM_OBJECTS = 2  # Maximum number of objects in the scene we control
NUM_WORLDS = 2  # Number of parallel environments
VIDEO_PATH = "/videos" # Set the path to where you want to save the videos
SCENE_NAME = "example_scene"
FPS = 4  # Video frames per second
env_config = EnvConfig()
render_config = RenderConfig()

base_env = GPUDriveJaxEnv(
    config=env_config,
    num_worlds=NUM_WORLDS,
    max_cont_agents=MAX_NUM_OBJECTS,  # Number of agents to control
    data_dir="example_data",
    device="cuda",
    render_config=render_config,
)

env = GPUDriveToJaxMARL(base_env)