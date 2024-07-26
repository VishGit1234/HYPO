import gymnasium

from envs.mujoco_sparse import HopperSparseEnv, Walker2dSparseEnv, HalfCheetahSparseEnv, AntSparseEnv

gymnasium.logger.set_level(40)


ENVS = {
    'HopperSparse': HopperSparseEnv,
    'Walker2dSparse': Walker2dSparseEnv,
    'HalfCheetahSparse': HalfCheetahSparseEnv,
    'AntSparse': AntSparseEnv
}


def make_env(env_id):
    if env_id not in ENVS.keys():
        return NormalizedEnv(gymnasium.make(env_id))
    else:
        return NormalizedEnv(ENVS[env_id]())


class NormalizedEnv(gymnasium.Wrapper):

    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps

        self.scale = env.action_space.high
        self.action_space.high /= self.scale
        self.action_space.low /= self.scale

    def step(self, action):
        return self.env.step(action * self.scale)
