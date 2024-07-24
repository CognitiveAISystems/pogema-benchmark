import sys

import yaml
from loguru import logger as log
from pogema import GridConfig, pogema_v0
from pogema_toolbox.create_env import MultiMapWrapper
from pogema_toolbox.registry import ToolboxRegistry

from mamba.env.pogema.pogema_follower import make_follower_pogema, make_follower_pogema_new

log.remove()  # remove the old handler. Else, the old one will work along with the new one you've added below'
log.add(sys.stdout, level="INFO")


def follower_pogema_env_from_config(config: GridConfig):
    env = make_follower_pogema(grid_config=config)
    env = MultiMapWrapper(env)
    return env


def follower_pogema_env(num_agents, **kwargs):
    with open("./mamba/env/pogema/simple-maps.yaml", "r") as f:
        maps = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps)
    env = make_follower_pogema(
        grid_config=GridConfig(
            on_target="restart",
            map_name="random_s.+_od5",
            observation_type="POMAPF",
            num_agents=num_agents,
        )
    )
    env = MultiMapWrapper(env)
    return env


def follower_pogema_env_mazes(num_agents, validation_set, on_target):
    fname = (
        "./mamba/env/pogema/validation-mazes.yaml"
        if validation_set
        else "./mamba/env/pogema/training-mazes.yaml"
    )
    with open(fname, "r") as f:
        maps = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps)

    map_name = f"{'validation-mazes' if validation_set else 'training-mazes'}-."
    env = make_follower_pogema(
        grid_config=GridConfig(
            on_target=on_target,
            map_name=map_name,
            observation_type="POMAPF",
            num_agents=num_agents,
            max_episode_steps=256,
            collision_system="soft",
        )
    )
    env = MultiMapWrapper(env)
    return env


def follower_pogema_env_mazes_random(num_agents, validation_set, on_target):
    fnames = ["./experiments/01-random/maps.yaml", "./experiments/02-mazes/maps.yaml"]
    
    for fname in fnames:
        with open(fname, "r") as f:
            maps = yaml.safe_load(f)
        ToolboxRegistry.register_maps(maps)

    env = make_follower_pogema_new(
        grid_config=GridConfig(
            on_target=on_target,
            map_name=f"validation-mazes-.|validation-random-seed-.",
            observation_type="POMAPF",
            num_agents=num_agents,
            max_episode_steps=128 if on_target=='restart' else 64,
            collision_system="soft",
        )
    )
    env = MultiMapWrapper(env)
    return env


def benchmark_pogema_env():
    with open("./mamba/env/pogema/simple-maps.yaml", "r") as f:
        maps = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps)
    env = MultiMapWrapper(
        pogema_v0(
            grid_config=GridConfig(on_target="restart", map_name="random_s.+_od5")
        )
    )
    return env


def main():
    with open("simple-maps.yaml", "r") as f:
        maps = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps)
    env = MultiMapWrapper(
        pogema_v0(
            grid_config=GridConfig(on_target="restart", map_name="random_s.+_od5")
        )
    )

    env.reset()
    env.render()


if __name__ == "__main__":
    main()
