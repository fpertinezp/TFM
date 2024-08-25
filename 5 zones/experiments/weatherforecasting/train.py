from datetime import datetime
from weatherforecastingwrapper import WeatherForecastingWrapper
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


def train_agent(n, delta, seed=42):
    set_seed(seed)

    # Configuración de entorno
    environment = "Eplus-5zone-mixed-continuous-stochastic-v1"
    episodes = 20
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    experiment_name = f'SB3_PPO-{environment}-episodes-{episodes}_{experiment_date}'
    workspace_dir = f'n_{n}/delta_{delta}'
    experiment_path = os.path.join(workspace_dir, experiment_name)

    # Definición de entorno de entrenamiento y validación
    train_env = gym.make(environment, env_name=experiment_path)
    eval_env = gym.make(environment, env_name=experiment_path+'_EVALUATION')
    
    train_env = LoggerWrapper(
        NormalizeAction(
        NormalizeObservation(
        WeatherForecastingWrapper(train_env, n=n, delta=delta, 
                                  weather_variability=[1.0, 0.0, 0.001]))))
    eval_env = LoggerWrapper(
        NormalizeAction(
        NormalizeObservation(
        WeatherForecastingWrapper(eval_env, n=n, delta=delta, 
                                  weather_variability=[1.0, 0.0, 0.001]))))

    # Definición de modelo DRL
    model = PPO('MlpPolicy', train_env, verbose=1, seed=seed)

    # Configuramos los callbacks para guardar el mejor modelo evaluado
    callbacks = []
    eval_callback = LoggerEvalCallback(
        eval_env=eval_env,
        best_model_save_path=eval_env.get_wrapper_attr('workspace_path') +
        '/best_model/',
        log_path=eval_env.get_wrapper_attr('workspace_path') +
        '/best_model/',
        eval_freq=(eval_env.get_wrapper_attr('timestep_per_episode') - 1) * 2 - 1,
        deterministic=True,
        render=False,
        n_eval_episodes=1)
    callbacks.append(eval_callback)
    callback = CallbackList(callbacks)

    # Definición del número de pasos temporales
    timesteps = episodes * (train_env.get_wrapper_attr('timestep_per_episode') - 1)

    # Entrenamiento
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        log_interval=1)
    
    if is_wrapped(train_env, NormalizeObservation):
        mean = train_env.get_wrapper_attr('mean')
        var = train_env.get_wrapper_attr('var')
        np.savetxt(f'./Eplus-env-n_{n}/delta_{delta}/mean.txt', mean)
        np.savetxt(f'./Eplus-env-n_{n}/delta_{delta}/var.txt', var)

    model.save(f'./Eplus-env-n_{n}/delta_{delta}/final_model.zip')

    train_env.close()
    eval_env.close()
    
    print('\n\n###### ENTRENAMIENTO FINALIZADO ######\n')
    

if len(sys.argv) > 1:
    n = int(sys.argv[1])
    delta = int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    train_agent(n, delta, seed)
else:
    print("No se pasaron argumentos.")
