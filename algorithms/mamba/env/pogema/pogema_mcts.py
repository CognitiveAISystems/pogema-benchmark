import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.wrappers import TransformObservation
from pogema import GridConfig
from pogema import AnimationConfig, AnimationMonitor
from pogema.envs import _make_pogema
from pogema.integrations.pymarl import PyMarlPogema
from pogema_toolbox.create_env import MultiMapWrapper
from pydantic import BaseModel
import numpy as np
from mamba.env.mcts_cpp.preprocessing import ProvideMapWrapper, ProvideGlobalObstacles, PreprocessorConfig, MCTSWrapper
from mamba.env.pogema.pogema_follower import ConcatPositionalFeatures, FixObservation



def make_mcts_pogema(grid_config):
    env = _make_pogema(grid_config)
    env = ProvideGlobalObstacles(env)
    env = ProvideMapWrapper(env)
    env = MCTSWrapper(env=env, config=PreprocessorConfig())
    return env


def make_mcts_pogema_multimap(grid_config):
    print(grid_config)
    env = _make_pogema(grid_config)
    env = ProvideGlobalObstacles(env)
    env = ProvideMapWrapper(env)
    env = MCTSWrapper(env=env, config=PreprocessorConfig())
    env = MultiMapWrapper(env)
    env = ConcatPositionalFeatures(env)
    env = FixObservation(env)
    env = AnimationMonitor(
        env, AnimationConfig(directory="./animations", save_every_idx_episode=1)
    )

    return env
