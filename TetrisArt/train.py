import torch
from TetrisArt.model.CNNFeatureExtractor import CNNFeatureExtractor
from TetrisArt.TetrisEnv import TetrisEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
import time

best_hp = {
	'gamma': 0.9693554805549407, 
	'learning_rate': 0.00019655203272629694,
	'policy_kwargs': dict(
            features_extractor_class=CNNFeatureExtractor,
            activation_fn=torch.nn.LeakyReLU,
            net_arch=[dict(pi=[64, 64, 64, 64], vf=[64, 64, 64, 64])],
        ),
}

def render_one_game(env, model):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action.item())
        env.render()
        time.sleep(.01)


def train_and_save(model_name: str="ppo_tetris_custom_cnn", steps:int = 250_000):
	width, height = 10, 24
	env = TetrisEnv(width, height)

	check_env(env)
	model_path = f"models/{model_name}"

	if os.path.exists(f"{model_path}.zip"):
		print("Loading model!")
		model = PPO.load(model_path, env=env)
	else:
		print("Creating new model!")
		model = PPO("MlpPolicy", env, verbose=1, **best_hp)

	while True:
		print(f"{model.policy = }")

		model.learn(total_timesteps=steps)
		model.save(model_path)
		
		render_one_game(env, model)

