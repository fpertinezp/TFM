from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from sinergym.utils.callbacks import *
from sinergym.utils.constants import *
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *
import sys
import os
import numpy as np
import torch
import random


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def play_agent(n, seed):

    set_seed(seed)

    # Configuración de entorno
    environment = "Eplus-radiant_free_heating-mixed-continuous-stochastic-v1"
    episodes = 10
    evaluation_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    evaluation_name = f'SB3_PPO-EVAL-{environment}-episodes-{episodes}_{evaluation_date}'
    workspace_dir = f'n_{n}/'
    experiment_path = os.path.join(workspace_dir, evaluation_name)

    # Carga de media y varianza calculadas en entrenamiento
    mean = np.loadtxt(f'./Eplus-env-n_{n}/mean.txt')
    var = np.loadtxt(f'./Eplus-env-n_{n}/var.txt')

    # Definición entorno de evaluación
    play_env = gym.make(environment, env_name=experiment_path)
    play_env = HeatPumpEnergyWrapper(play_env)
    play_env = MultiObsWrapper(play_env, n=n, flatten=True)
    play_env = NormalizeObservation(play_env, mean=mean, var=var, automatic_update=False)
    play_env = ExtremeFlowControlWrapper(play_env)
    play_env = NormalizeAction(play_env)
    play_env = LoggerWrapper(play_env)

    # Carga de modelo
    model_path = f'./Eplus-env-n_{n}/final_model.zip'
    model = PPO.load(model_path)

    # Evaluación del modelo
    for i in range(episodes):
        obs, info = play_env.reset()
        rewards = []
        truncated = terminated = False
        current_month = 0
        while not (terminated or truncated):
            a, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = play_env.step(a)
            rewards.append(reward)
            if info['month'] != current_month:
                current_month = info['month']
                print(info['month'], sum(rewards))
        print(
            'Episode ',
            i,
            'Mean reward: ',
            np.mean(rewards),
            'Cumulative reward: ',
            sum(rewards))
    
    play_env.close()

    print('\n\n###### EVALUACIÓN FINALIZADA ######\n')

if len(sys.argv) > 1:
    n = int(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 3 else 42

    play_agent(n, seed)
else:
    print("No se pasaron argumentos.")