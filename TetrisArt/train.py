import torch
from TetrisArt.model.FeatureExtractor.NetNDim import NetNDim
from stable_baselines3.common.evaluation import evaluate_policy
from TetrisArt.env.TetrisEnv import TetrisEnv
from TetrisArt.env.WrapperNumpy import WrapperNumpy
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
import time
import gym
# from TetrisBattle.envs.tetris_env import TetrisSingleEnv

best_hp = {
    'gamma': 0.97, 
    'learning_rate': 0.0002,
    'policy_kwargs': dict(
            features_extractor_class=NetNDim,
            # activation_fn=torch.nn.ReLU,
            net_arch=[dict(pi=[32], vf=[32])],
        ),
}

def render_one_game(env, model):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action.item())
        env.render()
        time.sleep(.02)
    env.close()


def train_and_save(model_name: str="ppo_tetris_custom_cnn", steps:int = 250_000):
    env = gym.make("Tetris-v1")
    check_env(env)
    model_path = f"models/{model_name}"

    if os.path.exists(f"{model_path}.zip"):
        print("Loading model!")
        model = PPO.load(model_path, env=env, tensorboard_log=f".logs/{model_name}/")
    else:
        print("Creating new model!")
        model = PPO("MlpPolicy", env, verbose=1, **best_hp, tensorboard_log=f".logs/{model_name}/")
        # model = PPO("CnnPolicy", env, verbose=1)

    print(f"{model.policy = }")
    
    while True:
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"{mean_reward = }")
        print(f"{std_reward = }")
        model.save(model_path)

        # render_one_game(env, model)

