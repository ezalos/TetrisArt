from gym.envs.registration import register
from TetrisArt.env.TetrisEnv import TetrisEnv, optimize_TetrisEnv
from TetrisArt.env.WrapperNormalize import WrapperNormalize
from TetrisArt.env.WrapperNumpy import WrapperNumpy
# Example for the CartPole environment
from gym.wrappers import FrameStack, GrayScaleObservation, NormalizeObservation

from stable_baselines3.common.monitor import Monitor

def optimize_wrap_env(trial):
    return {
		"grayscale": trial.suggest_categorical("grayscale", [True, False]),
		"k_frames": int(trial.suggest_int('k_frames', 1, 32, log=True)),
		"env_kwargs": optimize_TetrisEnv(trial),
	}

def wrap_env(grayscale=True, k_frames=4, env_kwargs={}):
	env = TetrisEnv(**env_kwargs)
	if grayscale:
		env = GrayScaleObservation(env)
	env = WrapperNormalize(env)
	env = Monitor(env=env)
	env = FrameStack(env, num_stack=k_frames)
	env = WrapperNumpy(env)
	return env


register(
    # unique identifier for the env `name-version`
    id="Tetris-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="TetrisArt:wrap_env",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    # max_episode_steps=500,
	reward_threshold=1200.,
)