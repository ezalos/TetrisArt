from TetrisArt.train import train_and_save
from TetrisArt.optuna import optuna_do

from fire import Fire


class Main():
	def train(self, model_name: str="ppo_tetris_custom_cnn", steps:int = 250_000):
		train_and_save(model_name, steps)
	
	def optimize():
		optuna_do()

if __name__ == "__main__":
	Fire(Main)