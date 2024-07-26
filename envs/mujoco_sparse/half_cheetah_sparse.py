import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium import spaces


class HalfCheetahSparseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, sparse_threshold=2.):
        self.sparse_threshold = sparse_threshold
        print(f"Using Sparse HalfCheetah Env.")
        self._max_episode_steps = 1000
        self.reward_flags = np.ones(100000, dtype=bool)
        self.max_level = 0
        render_modes = [
            "human",
            "rgb_array",
            "depth_array",
        ]
        self.metadata["render_modes"] = render_modes
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5, spaces.Box(-np.inf, np.inf, (17,), np.float64), ["human"])
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]
        ob = self._get_obs()

        # old reward
        # reward_ctrl = - 0.1 * np.square(action).sum()
        # reward_run = (xposafter - xposbefore)/self.dt
        # reward = reward_ctrl + reward_run

        # new sparse reward
        level = int((xposafter - self.init_qpos[0]) / self.sparse_threshold)
        if level >= 1 and self.reward_flags[level]:
            reward = 1.
            self.reward_flags[level] = False
        else:
            reward = 0.
        if level > self.max_level:
            self.max_level = level

        done = False
        return ob, reward, done, {}, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.integers(self.model.nv) * .1
        self.set_state(qpos, qvel)
        # new added
        self.reward_flags = np.ones(100000, dtype=bool)
        self.max_level = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
