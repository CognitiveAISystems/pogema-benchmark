import gymnasium
from gymnasium import ObservationWrapper
from pogema import GridConfig
from pogema import AnimationConfig, AnimationMonitor
from pogema.envs import _make_pogema
from pogema_toolbox.create_env import MultiMapWrapper
from pydantic import BaseModel

from follower.follower_python.preprocessing import ConcatPositionalFeatures, FollowerWrapper, wrap_preprocessors, PreprocessorConfig


class PlannerConfig(BaseModel):
    use_precalc_cost: bool = True
    use_dynamic_cost: bool = True
    reset_dynamic_cost: bool = True


# class PreprocessorConfig(PlannerConfig):
#     intrinsic_target_reward: float = 0.01


class ProvideGlobalObstacles(gymnasium.Wrapper):
    def get_global_obstacles(self):
        return self.grid.get_obstacles().astype(int).tolist()

    def get_global_agents_xy(self):
        return self.grid.get_agents_xy()


class FixObservation(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space["obs"]

    def observation(self, observations):
        result = []
        for obs in observations:
            result.append(obs["obs"])
        return result


def make_follower_pogema(grid_config):
    env = _make_pogema(grid_config)
    env = ProvideGlobalObstacles(env)
    env = FollowerWrapper(env=env, config=PreprocessorConfig())
    env = ConcatPositionalFeatures(env)
    env = FixObservation(env)
    return env


def make_follower_pogema_new(grid_config):
    env = _make_pogema(grid_config)
    env = wrap_preprocessors(env, config=PreprocessorConfig())
    return env


def make_follower_pogema_multimap(grid_config):

    env = _make_pogema(grid_config)
    env = ProvideGlobalObstacles(env)
    env = FollowerWrapper(env=env, config=PreprocessorConfig())
    env = ConcatPositionalFeatures(env)
    env = MultiMapWrapper(env)
    env = FixObservation(env)
    env = AnimationMonitor(
        env, AnimationConfig(directory="./animations", save_every_idx_episode=1)
    )
    return env
