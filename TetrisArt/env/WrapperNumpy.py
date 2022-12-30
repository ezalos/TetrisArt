from collections import deque
import gym
import numpy as np

class WrapperNumpy(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return np.array(observation)