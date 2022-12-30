import time
import os

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import gym


def render_one_game(env, model):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action.item())
        env.render()
        time.sleep(.02)
    env.close()

def load_model(model_name: str, env):
    model_path = f"models/{model_name}"

    if not os.path.exists(f"{model_path}.zip"):
        raise Exception(f"File do not exist: {model_path}.zip")
    print("Loading model!")
    model = PPO.load(model_path, env=env)
    return model
    
def create_env():
    env = gym.make("Tetris-v1")
    ob = env.reset()
    check_env(env)
    return env

def play_n_games(model_name: str, nb_games:int):
    env = create_env()
    model = load_model(model_name=model_name, env=env)
    for _ in range(nb_games):
        render_one_game(env, model)