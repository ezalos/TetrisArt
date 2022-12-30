from collections import deque
import gym
import numpy as np

class WrapperNormalize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_high = env.observation_space.high
        self.original_low = env.observation_space.low
        self.observation_space = gym.spaces.Box(
            low=0., high=1., 
            shape=env.observation_space.shape, 
            dtype=np.float64,
        )

    def observation(self, ob):
        low = self.original_low.flat[0]
        high = self.original_high.flat[0]
        ob_range = high - low
        ob = (ob - low) / ob_range
        return ob