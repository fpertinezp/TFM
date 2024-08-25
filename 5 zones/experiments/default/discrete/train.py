from datetime import datetime
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList
from sinergym.utils.callbacks import *
from sinergym.utils.constants import *
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *
import numpy as np
import torch
import random
import sys


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_agent(seed):
    set_seed(seed)
    
    # Configuración de entorno
    environment = "Eplus-5zone-mixed-discrete-stochastic-v1"
    episodes = 20
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    experiment_name = f'SB3_DQN-{environment}-episodes-{episodes}_{experiment_date}'

    # Definición entorno de entrenamiento y validación
    train_env = gym.make(environment, env_name=experiment_name)
    eval_env = gym.make(environment, env_name=experiment_name+'_EVALUATION')

    train_env = LoggerWrapper(
        NormalizeObservation(train_env))
    eval_env = LoggerWrapper(
        NormalizeObservation(eval_env))

    # Definición de modelo DRL
    model = DQN('MlpPolicy', train_env, verbose=1, seed=seed)


    # Configuramos los callbacks para guardar el mejor modelo evaluado
    # durante el entrenamiento
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

    # Definición el número de pasos temporales
    timesteps = episodes * (train_env.get_wrapper_attr('timestep_per_episode') - 1)

    # Entrenamiento
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        log_interval=1)
    
    if is_wrapped(train_env, NormalizeObservation):
        mean = train_env.get_wrapper_attr('mean')
        var = train_env.get_wrapper_attr('var')
        np.savetxt(f'./mean.txt', mean)
        np.savetxt(f'./var.txt', var)

    model.save(f'./final_model.zip')

    train_env.close()
    eval_env.close()
    
    print('\n\n###### ENTRENAMIENTO FINALIZADO ######\n')
    


seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42

train_agent(seed)