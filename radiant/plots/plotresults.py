import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def plot(experiment, n):
    # Leer los archivos CSV
    default = pd.read_csv('./default/progress.csv')

    if experiment == "weatherforecasting":
        delta1 = pd.read_csv(f'./{experiment}/n_{n}/delta_1/progress.csv')
        delta2 = pd.read_csv(f'./{experiment}/n_{n}/delta_2/progress.csv')
        delta3 = pd.read_csv(f'./{experiment}/n_{n}/delta_3/progress.csv')

        # Extraer las recompensas medias
        default_rewards = default['mean_reward']
        delta1_rewards = delta1['mean_reward']
        delta2_rewards = delta2['mean_reward']
        delta3_rewards = delta3['mean_reward']

        # Crear un DataFrame para los datos
        data = pd.DataFrame({
            'default': default_rewards,
            'delta = 1': delta1_rewards,
            'delta = 2': delta2_rewards,
            'delta = 3': delta3_rewards
        })
    elif experiment == "multiobservation":
        n4 = pd.read_csv(f'./{experiment}/n_4/progress.csv')
        n10 = pd.read_csv(f'./{experiment}/n_10/progress.csv')
        n20 = pd.read_csv(f'./{experiment}/n_20/progress.csv')

        # Extraer las recompensas medias
        default_rewards = default['mean_reward']
        n4_rewards = n4['mean_reward']
        n10_rewards = n10['mean_reward']
        n20_rewards = n20['mean_reward']

        # Crear un DataFrame para los datos
        data = pd.DataFrame({
            'default': default_rewards,
            'n = 4': n4_rewards,
            'n = 10': n10_rewards,
            'n = 20': n20_rewards
        })
    else:
        combined = pd.read_csv(f'./{experiment}/progress.csv')

        default_rewards = default['mean_reward']
        combined_rewards = combined['mean_reward']

        data = pd.DataFrame({
            'default': default_rewards,
            'combined': combined_rewards
        })

    x = 'configuration'

    # Calcular la media y desviación típica
    statistics = data.describe().loc[['mean', 'std']].round(3)
    
    # Guardar las estadísticas en un archivo CSV
    if experiment == "weatherforecasting":
        stats_file_path = f'./{experiment}/n_{n}/stats.csv'
    else:
        stats_file_path = f'./{experiment}/stats.csv'
    statistics.to_csv(stats_file_path)

    # Reestructurar los datos para el boxplot
    data_melted = data.melt(var_name = x, value_name='mean_reward')

    # Crear y personalizar el boxplot
    plt.figure()
    sns.set(style="whitegrid")
    sns.boxplot(x = x, y='mean_reward', data=data_melted, fill=False, gap=.1, palette="Set1")


    plt.ylabel("Recompensa media por episodio")
    plt.xlabel("")
    # Añadir títulos y etiquetas
    if experiment == "weatherforecasting":
        plt.title(f"n = {n}")
        # Guardar la imagen del gráfico
        plt.savefig(f'./{experiment}/n_{n}/plot.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'./{experiment}/plot.png', dpi=300, bbox_inches='tight')
    
    plt.close()

if len(sys.argv) > 1:
    experiment = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    plot(experiment, n)
else:
    print("No se pasaron argumentos.")
