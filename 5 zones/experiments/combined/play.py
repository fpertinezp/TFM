from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from weatherforecastingwrapper import WeatherForecastingWrapper
from sinergym.utils.callbacks import *
from sinergym.utils.constants import *
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *
import numpy as np
import sys
import torch

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def play_agent(seed):
    set_seed(seed)

    # Configuración de entorno
    environment = "Eplus-5zone-mixed-continuous-stochastic-v1"
    episodes = 10
    evaluation_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    evaluation_name = f'SB3_PPO-EVAL-{environment}-episodes-{episodes}_{evaluation_date}'

    # Carga de media y varianza calculadas en entrenamiento
    mean = np.loadtxt(f'./mean.txt')
    var = np.loadtxt(f'./var.txt')

    # Definición entorno de evaluación
    play_env = gym.make(environment, env_name=evaluation_name)
    play_env = MultiObsWrapper(play_env, n=10, flatten=True)
    play_env = WeatherForecastingWrapper(play_env, n=5, delta=2, weather_variability=[1.0, 0.0, 0.001])
    play_env = NormalizeObservation(play_env, mean=mean, var=var, automatic_update=False)
    play_env = NormalizeAction(play_env)
    play_env = LoggerWrapper(play_env)

    # Carga de modelo
    model_path = f'./final_model.zip'
    model = PPO.load(model_path)

    # Evaluación el modelo
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

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42

play_agent(seed=seed)
