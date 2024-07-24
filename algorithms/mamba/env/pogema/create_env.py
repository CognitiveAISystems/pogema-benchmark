import time
import numpy as np
from gymnasium import Wrapper
from loguru import logger
from pogema_toolbox.registry import ToolboxRegistry

from pydantic import BaseModel
from pogema import AnimationConfig, AnimationMonitor
from copy import deepcopy
from pogema import pogema_v0, GridConfig
from pogema.generator import generate_new_target, generate_from_possible_targets
import re

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

class MultiMapWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._configs = []
        self._rnd = np.random.default_rng(self.grid_config.seed)
        pattern = self.grid_config.map_name

        if pattern:
            maps = ToolboxRegistry.get_maps()
            for map_name in sorted(maps):
                if re.match(pattern, map_name):
                    cfg = deepcopy(self.grid_config)
                    cfg.map = maps[map_name]
                    cfg.map_name = map_name
                    cfg = GridConfig(**cfg.dict())
                    self._configs.append(cfg)
            if not self._configs:
                raise KeyError(f"No map matching: {pattern}")

    def reset(self, seed=None, **kwargs):
        self._rnd = np.random.default_rng(seed)
        if self._configs is not None and len(self._configs) >= 1:
            map_idx = self._rnd.integers(0, len(self._configs))
            cfg = deepcopy(self._configs[map_idx])
            self.env.unwrapped.grid_config = cfg
            self.env.unwrapped.grid_config.seed = seed
        return self.env.reset(seed=seed, **kwargs)

class ProvideGlobalObstacles(Wrapper):
    def get_global_obstacles(self):
        return self.grid.get_obstacles().astype(int).tolist()

    def get_global_agents_xy(self):
        return self.grid.get_agents_xy()

    def get_global_targets_xy(self):
        return self.grid.get_targets_xy()

    def get_lifelong_global_targets_xy(self):
        all_goals = []
        cur_goals = self.grid.get_targets_xy()
        generators = deepcopy(self.random_generators)
        for agent_idx in range(self.grid_config.num_agents):
            distance = 0
            cur_goal = cur_goals[agent_idx]
            goals = [cur_goal]
            while distance < self.grid_config.max_episode_steps + 100:
                if self.grid_config.possible_targets_xy is None:
                    new_goal = generate_new_target(generators[agent_idx],
                                                self.grid.point_to_component,
                                                self.grid.component_to_points,
                                                cur_goal)
                else:
                    new_goal = generate_from_possible_targets(generators[agent_idx], self.grid_config.possible_targets_xy, cur_goal)
                    new_goal = (new_goal[0] + self.grid_config.obs_radius, new_goal[1] + self.grid_config.obs_radius)
                distance += abs(cur_goal[0] - new_goal[0]) + abs(cur_goal[1] - new_goal[1])
                cur_goal = new_goal
                goals.append(cur_goal)
            all_goals.append(goals)
        return all_goals


class ProvideMapWrapper(Wrapper):
    def reset(self, **kwargs):
        observations, infos = self.env.reset(seed=self.env.grid_config.seed)
        global_obstacles = self.get_global_obstacles()
        global_agents_xy = self.get_global_agents_xy()
        global_targets_xy = self.get_global_targets_xy()
        if self.env.grid_config.on_target == 'restart':
            global_lifelong_targets_xy = self.get_lifelong_global_targets_xy()
        observations[0]['global_obstacles'] = global_obstacles
        observations[0]['max_episode_steps'] = self.env.grid_config.max_episode_steps
        for idx, obs in enumerate(observations):
            obs['global_xy'] = global_agents_xy[idx]
            obs['global_target_xy'] = global_targets_xy[idx]
            if self.env.grid_config.on_target == 'restart':
                obs['global_lifelong_targets_xy'] = global_lifelong_targets_xy[idx]
        return observations, infos


class RuntimeMetricWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._start_time = None
        self._env_step_time = None

    def step(self, actions):
        env_step_start = time.monotonic()
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        env_step_end = time.monotonic()
        self._env_step_time += env_step_end - env_step_start
        if all(terminated) or all(truncated):
            final_time = time.monotonic() - self._start_time - self._env_step_time
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(runtime=final_time)
        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._start_time = time.monotonic()
        self._env_step_time = 0.0
        return obs

class AgentsInObsWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._avg_agents_density = None

    def count_agents(self, observations):
        avg_agents_density = []
        for obs in observations:
            traversable_cells = np.size(obs['obstacles']) - np.count_nonzero(obs['obstacles'])
            avg_agents_density.append(np.count_nonzero(obs['agents'])/traversable_cells)
        self._avg_agents_density.append(np.mean(avg_agents_density))

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        self.count_agents(observations)
        if all(terminated) or all(truncated):
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(avg_agents_density=float(np.mean(self._avg_agents_density)))
        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        self._avg_agents_density = []
        observations, info = self.env.reset(**kwargs)
        self.count_agents(observations)
        return observations, info

def create_env_base(config):
    env = pogema_v0(grid_config=config)
    env = AgentsInObsWrapper(env)
    env = ProvideGlobalObstacles(env)
    env = ProvideMapWrapper(env)
    env = MultiMapWrapper(env)
    if config.with_animation:
        logger.debug('Wrapping environment with AnimationMonitor')
        env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=1))
    
    # Adding runtime metrics
    env = RuntimeMetricWrapper(env)
    
    return env