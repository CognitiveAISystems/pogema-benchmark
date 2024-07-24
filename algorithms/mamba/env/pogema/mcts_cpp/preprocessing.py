import numpy as np
from copy import deepcopy
from pydantic import BaseModel
from pogema import pogema_v0
from pogema.generator import generate_new_target
from gymnasium import Wrapper, ObservationWrapper

import cppimport.import_hook
from .environment import Environment


class PreprocessorConfig(BaseModel):
    obs_radius: int = 5
    collision_system: str = 'soft'
    steps_on_goal: int = 1
    on_target: str = 'wait'
    progressed_reward: float = 0.1
    

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
                new_goal = generate_new_target(generators[agent_idx],
                                               self.grid.point_to_component,
                                               self.grid.component_to_points,
                                               self.grid.positions_xy[agent_idx])
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
        observations[0]['global_obstacles'] = global_obstacles
        #observations[0]['steps_on_goal'] = self.env.grid_config.steps_on_goal
        for idx, obs in enumerate(observations):
            obs['global_xy'] = global_agents_xy[idx]
            obs['global_target_xy'] = global_targets_xy[idx]
        if self.env.grid_config.on_target in ['wait', 'restart']:
            global_lifelong_targets_xy = self.get_lifelong_global_targets_xy()
            for idx, obs in enumerate(observations):
                obs['global_lifelong_targets_xy'] = global_lifelong_targets_xy[idx]
        return observations, infos
    

class MCTSWrapper(ObservationWrapper):

    def __init__(self, env, config: PreprocessorConfig):
        super().__init__(env)
        self._cfg: PreprocessorConfig = config
        self.cpp_env = None

    def get_observation(self):
        observations = []
        for i in range(self.cpp_env.get_num_agents()):
            obs = self.cpp_env.generate_input(i, self._cfg.obs_radius)
            obs_dict = {}
            obs_dict['agents'] = np.array(obs[0]).reshape(self._cfg.obs_radius*2+1, self._cfg.obs_radius*2+1)
            obs_dict['path'] = np.array(obs[1]).reshape(self._cfg.obs_radius*2+1, self._cfg.obs_radius*2+1)
            obs_dict['goals'] = np.array(obs[2]).reshape(self._cfg.obs_radius*2+1, self._cfg.obs_radius*2+1)
            observations.append(obs_dict)
        return observations

    def init_cpp_env(self, observations):
        self.cpp_env = Environment(self._cfg.obs_radius, self._cfg.collision_system, self._cfg.on_target,
                                   self._cfg.progressed_reward, self._cfg.steps_on_goal)
        self.cpp_env.create_grid(len(observations[0]['global_obstacles']), len(observations[0]['global_obstacles'][0]))
        for i in range(len(observations[0]['global_obstacles'])):
            for j in range(len(observations[0]['global_obstacles'][0])):
                if observations[0]['global_obstacles'][i][j]:
                    self.cpp_env.add_obstacle(i, j)
        self.cpp_env.precompute_cost2go()
        if 'global_lifelong_targets_xy' in observations[0]:
            for agent_idx in range(len(observations)):
                self.cpp_env.add_agent(observations[agent_idx]['global_xy'],
                                    observations[agent_idx]['global_lifelong_targets_xy'])
        else:
            for agent_idx in range(len(observations)):
                self.cpp_env.add_agent(observations[agent_idx]['global_xy'],
                                    [observations[agent_idx]['global_target_xy']])

    def step(self, action):
        _, _, done, tr, info = self.env.step(action)
        assert self.cpp_env is not None
        reward = self.cpp_env.step(action, True)
        observation = self.get_observation()
        return observation, reward, done, tr, info

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        assert 'global_obstacles' in observations[0]
        self.init_cpp_env(observations)
        return self.get_observation(), infos