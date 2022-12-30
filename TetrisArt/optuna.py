import optuna
import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from TetrisArt.model.FeatureExtractor.NetNDim import NetNDim, optimize_NetNDim
from TetrisArt.env.TetrisEnv import TetrisEnv
import torch

from TetrisArt import optimize_wrap_env, wrap_env

def optimize_ppo(trial):
    """ Learning hyperparamters we want to optimise"""
    nb_of_neurons = trial.suggest_int('nb_of_neurons', 8, 128, log=True)
    nb_of_layers = trial.suggest_int('nb_of_layers', 1, 6, log=False)
    net_arch = [nb_of_neurons] * nb_of_layers
    activation_function = trial.suggest_categorical(
        "activation_fn",
        ["ReLU","Tanh","LeakyReLU"]
    )
    batch_size = int(2 ** int(trial.suggest_int('batch_size', 4, 14)))
    return {
        'n_steps': batch_size,
        'batch_size': batch_size,
        'gamma': trial.suggest_float('gamma', 0.9, 0.99999, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1., log=True),
        'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'clip_range_vf': trial.suggest_float('clip_range_vf', 0.1, 0.4),
        # 'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 1.),
        'policy_kwargs': dict(
            features_extractor_class=NetNDim,
            features_extractor_kwargs=optimize_NetNDim(trial),
            # share_features_extractor=True,
            activation_fn={
                "ReLU": torch.nn.ReLU, 
                "Tanh": torch.nn.Tanh,
                "LeakyReLU": torch.nn.LeakyReLU,
            }[activation_function],
            net_arch=[dict(pi=net_arch, vf=net_arch)]
        ),
    }


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    model_params = optimize_ppo(trial)

    # env = make_vec_env(lambda: GoLeftEnv(), n_envs=16, seed=0)
    env = wrap_env(**optimize_wrap_env(trial))
    model = PPO('MlpPolicy', env, verbose=0, **model_params)
    for step in range(5):
        model.learn(total_timesteps=50_000)
        episodes_reward, episodes_len = evaluate_policy(
            model, 
            env, 
            n_eval_episodes=25, 
            return_episode_rewards=True
        )
        mean_episode_len = sum(episodes_len) / len(episodes_len)
        trial.report(mean_episode_len, step)
    return mean_episode_len


def optuna_do(study_name:str):
    storage="sqlite:///optuna.db"
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )
    # metric_name = "target"
    # tb_callback = optuna.integration.TensorBoardCallback("./logs/optuna/", metric_name)
    try:
        study.optimize(
            optimize_agent, 
            n_trials=100, 
            n_jobs=4, 
            catch=(Exception,),
            # callbacks=[tb_callback]
        )
        print(f"{study.best_params = }")
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')
