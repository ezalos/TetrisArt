from TetrisArt.train import train_and_save
from TetrisArt.optuna import optuna_do
from TetrisArt.predict import play_n_games
from fire import Fire


class Main():
	def train(self, model_name: str="ppo_tetris_custom_cnn", steps:int = 250_000):
		train_and_save(model_name, steps)
	
	def optimize(self, study_name:str):
		optuna_do(study_name)

	def predict(self, model_name: str="ppo_tetris_custom_cnn", nb_games:int = 5):
		play_n_games(model_name, nb_games=nb_games)

if __name__ == "__main__":
	Fire(Main)