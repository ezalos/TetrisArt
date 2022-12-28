import optuna
import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from TetrisArt.model.CNNFeatureExtractor import CNNFeatureExtractor
from TetrisArt.TetrisEnv import TetrisEnv
import torch

def optimize_ppo(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        # 'n_steps': int(trial.suggest_int('n_steps', 16, 2048, log=True)),
        'gamma': trial.suggest_float('gamma', 0.9, 0.99999, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1., log=True),
        # 'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True),
        # 'clip_range': trial.suggest_float('cliprange', 0.1, 0.4),
        # 'clip_range_vf': trial.suggest_float('cliprange', 0.1, 0.4),
        # 'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        # 'gae_lambda': trial.suggest_float('lam', 0.8, 1.),
        'policy_kwargs': dict(
            features_extractor_class=CNNFeatureExtractor,
            activation_fn=trial.suggest_categorical(
                "activation_fn",
                [
                    torch.nn.ReLU, 
                    torch.nn.Tanh, 
                    torch.nn.LeakyReLU
                ]
                ),
            net_arch=trial.suggest_categorical(
                "net_arch",
                [
                    [dict(pi=[64, 64], vf=[64, 64])],
                    [dict(pi=[32, 32, 32], vf=[32, 32, 32])],
                    [dict(pi=[64, 64, 64], vf=[64, 64, 64])],
                    [dict(pi=[64, 64, 64, 64], vf=[64, 64, 64, 64])],
                ]
            )
        ),
    }


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    model_params = optimize_ppo(trial)

    # env = make_vec_env(lambda: GoLeftEnv(), n_envs=16, seed=0)
    env = TetrisEnv(10, 24)
    model = PPO('MlpPolicy', env, verbose=0, **model_params)
    model.learn(total_timesteps=25_000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    return -1 * mean_reward


def optuna_do():
    study = optuna.create_study()
    try:
        study.optimize(optimize_agent, n_trials=100, n_jobs=4, catch=(Exception,))
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')
