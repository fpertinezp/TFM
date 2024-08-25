from sinergym.envs.eplus_env import EplusEnv
from opyplus import WeatherData
from sinergym.utils.callbacks import *
from sinergym.utils.wrappers import *


def apply_ornstein_uhlenbeck_variability(
        df,
        columns: List[str] = ['drybulb'],
        variation: Optional[Tuple[float, float, float]] = None) -> str:

    if variation is not None:

        sigma = variation[0]  # Desviación estándar.
        mu = variation[1]  # Media.
        tau = variation[2]  # Constante de tiempo.

        T = 1.
        
        n = len(df[columns[0]])
        dt = T / n

        sigma_bis = sigma * np.sqrt(2. / tau)
        sqrtdt = np.sqrt(dt)

        x = np.zeros(n)

        # Crear ruido
        for i in range(n - 1):
            x[i + 1] = x[i] + dt * (-(x[i] - mu) / tau) + \
                sigma_bis * sqrtdt * np.random.randn()

        for column in columns:
            # Añadir ruido
            df[column] += x

    return df

class WeatherForecastingWrapper(gym.Wrapper):
    
    logger = Logger().getLogger(name='WRAPPER WeatherForecastingWrapper',
                                level=LOG_WRAPPERS_LEVEL)

    def __init__(self, env: EplusEnv, n: int = 5, delta: int = 1, columns: List[str] = ['drybulb', 'relhum', 'winddir', 'windspd', 'dirnorrad', 'difhorrad'], 
                 weather_variability: Optional[Tuple[float, float, float]] = None) -> None:
        
        super(WeatherForecastingWrapper, self).__init__(env)
        self.n = n
        self.delta = delta
        shape = self.get_wrapper_attr('observation_space').shape
        new_shape = (shape[0] + (len(columns) * n),)
        self.observation_space = gym.spaces.Box(
            low=-5e6, high=5e6, shape=new_shape, dtype=np.float32)
        
        data = WeatherData.from_epw(env.get_wrapper_attr('weather_path')).get_weather_series()[['month', 'day', 'hour'] + columns]
        
        if weather_variability:
            self.weather_data = apply_ornstein_uhlenbeck_variability(data, variation=weather_variability, columns=columns)
        else:
            self.weather_data = data

        self.logger.info('Wrapper initialized.')

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        
        obs, info = self.env.reset(seed=seed, options=options)
        filter = (self.weather_data['month'] == info['month']) & (self.weather_data['day'] == info['day']) & (self.weather_data['hour'] == (info['hour'] + 1))
        i = self.weather_data[filter].index[0]

        # Crear una lista de índices con el salto especificado
        indices = list(range(i + self.delta, i + self.delta * self.n + 1, self.delta))
        
        # Asegurarse de no exceder los límites del DataFrame
        indices = [idx for idx in indices if idx < len(self.weather_data)]
        if len(indices) == 0:
            indices = [i]
        
        # Obtener las filas necesarias
        selected_rows = self.weather_data.iloc[indices, 3:].values
        
        # Si no hay suficientes filas, repetir la última fila hasta completar
        if len(selected_rows) < self.n:
            last_row = selected_rows[-1]
            needed_rows = self.n - len(selected_rows)
            selected_rows = np.vstack([selected_rows, np.tile(last_row, (needed_rows, 1))])
        
        # Aplanar la matriz de filas seleccionadas
        info_forecasting = selected_rows.flatten()
        
        # Concatenar la información del pronóstico con la observación
        obs = np.concatenate((obs, info_forecasting))

        return obs.reshape(-1,), info

    def step(self, action: Union[int, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        obs, reward, terminated, truncated, info = self.env.step(action)
        
        filter = (self.weather_data['month'] == info['month']) & (self.weather_data['day'] == info['day']) & (self.weather_data['hour'] == (info['hour'] + 1))
        i = self.weather_data[filter].index[0]
        
        indices = list(range(i + self.delta, i + self.delta * self.n + 1, self.delta))

        indices = [idx for idx in indices if idx < len(self.weather_data)]
        
        if len(indices) == 0:
            indices = [i]

        selected_rows = self.weather_data.iloc[indices, 3:].values
        
        if len(selected_rows) < self.n:
            last_row = selected_rows[-1]
            needed_rows = self.n - len(selected_rows)
            selected_rows = np.vstack([selected_rows, np.tile(last_row, (needed_rows, 1))])
        
        info_forecasting = selected_rows.flatten()
        
        obs = np.concatenate((obs, info_forecasting))

        return obs.reshape(-1,), reward, terminated, truncated, info 