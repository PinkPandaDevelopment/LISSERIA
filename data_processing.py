#####Important Packages#####
import pandas as pd
import numpy as np
from hampel import hampel
import scipy as sp
from scipy import signal as sig
import warnings 
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from math import sqrt
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from typing import Dict
import pytz
import os
import glob


warnings.filterwarnings("ignore")
from typing import List
import pandas as pd
from datetime import datetime
import io

import pdb


#########Reading data##########
# def load_data_txt(mission) -> pd.DataFrame:
#     # Reading first n_max lines
#     lines = [line.decode('latin-1').strip() for line in mission.readlines()]
    
#     # Find the index containing "DATA STRUCTURE" to identify where columns start
#     index_data = next(i for i, line in enumerate(lines) if 'DATA' in line) + 1
    
#     # Extract column names and clean them up
#     columns = lines[index_data].split(',')
#     columns[0] = 'Date'
#     columns[1] = 'Time'
    
#     # Extract the data part of the file
#     data_lines = lines[index_data + 1:]
    
#     # Convert data lines to lists of values
#     f_data = [line.split(',')[:-1] for line in data_lines]

    
#     # Create DataFrame
#     df = pd.DataFrame(f_data, columns=columns)
    
#     # Convert columns to the appropriate data types
#     for col in df.columns[2:]:
#         df[col] = df[col].astype(float)

#     df.rename(columns={' speed [knots]': 'speed','Time':'time',' Altitude':'A_1',' Lon':'lot',' Lat':'lat',' T[CÂ°]': 'T_3',' CO[ppm]': 'CO', ' NO2[ppm]': 'NO₂', ' C3H8(CO)[ppm]': 'Propano C₃H₈', ' Iso-butano(CO)[ppm]':'Butano C₄H₁₀', ' CH4(CO)[ppm]':'Metano CH₄', ' H2(NO2)[ppm]':'H₂', ' Etanol(CO)[ppm]':'Etanol C₂H₅OH'}, inplace=True)

#     # # Convert the time column to datetime
#     # df['time'] = pd.to_datetime(df['time'])

#     # # Set the timezone to UTC
#     # df['time'] = df['time'].dt.tz_localize(pytz.utc)

#     # # Convert the timezone from UTC to UTC-5
#     # df['time'] = df['time'].dt.tz_convert(pytz.FixedOffset(-300))  # UTC-5 is 300 minutes behind UTC

#     # # Extract time part only, without the date
#     # df['time'] = df['time'].dt.strftime('%H:%M:%S.%f').str.slice(0, -3)
    

#     #sln 1

#     # # Combine 'Date' and 'Time' into a single datetime column
#     # df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['time'], format='%d-%m-%y %H:%M:%S.%f', errors='coerce')

#     # # Handle invalid datetime parsing (if any)
#     # if df['datetime'].isnull().any():
#     #     print("Warning: Some datetime values could not be parsed.")

#     # # Set the timezone to UTC
#     # df['datetime'] = df['datetime'].dt.tz_localize(pytz.utc)

#     # # Convert the timezone from UTC to UTC-5
#     # df['datetime'] = df['datetime'].dt.tz_convert(pytz.FixedOffset(-300))  # UTC-5

#     # # Extract time part only (without date) for the 'time' column
#     # df['time'] = df['datetime'].dt.strftime('%H:%M:%S.%f').str.slice(0, -3)

#  #sln 2
#      # Fix the time format by ensuring two digits for hours and minutes
#     df['time'] = df['time'].apply(lambda x: '{:02}:{:02}:{:02}.{:03}'.format(
#         int(x.split(':')[0]),  # Hours
#         int(x.split(':')[1]),  # Minutes
#         int(float(x.split(':')[2])),  # Seconds (integer part)
#         int(float(x.split(':')[2]) * 1000 % 1000)  # Milliseconds
#     ))

#     # Convert the time and date columns to datetime
#     df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['time'], format='%d-%m-%y %H:%M:%S.%f')

#     # Set the timezone to UTC
#     df['datetime'] = df['datetime'].dt.tz_localize(pytz.utc)

#     # Convert the timezone from UTC to UTC-5
#     df['datetime'] = df['datetime'].dt.tz_convert(pytz.FixedOffset(-300))  # UTC-5 is 300 minutes behind UTC

#     # Extract time part only, without the date
#     df['time'] = df['datetime'].dt.strftime('%H:%M:%S.%f').str.slice(0, -3)
    
#     return df

def load_data_txt(mission) -> pd.DataFrame:
    # Leer líneas del archivo con codificación adecuada
    lines = [line.decode('latin-1').strip() for line in mission.readlines()]
    print(f"respuesta archivo txt: {lines}")
    print("FIN")
    # Identificar el índice donde comienzan las columnas
    index_data = next(i for i, line in enumerate(lines) if 'DATA' in line) + 1
    
    # Extraer y limpiar nombres de columnas
    columns = lines[index_data].split(',')
    columns[0] = 'Date'
    columns[1] = 'Time'
    
    # Extraer las líneas de datos
    data_lines = lines[index_data + 1:]
    f_data = [line.split(',')[:-1] for line in data_lines]

    # Crear DataFrame
    df = pd.DataFrame(f_data, columns=columns)
    
    # Reemplazar valores "ovf" con NaN
    for col in df.columns[2:]:
        df[col] = df[col].replace('ovf', np.nan)

    # Convertir columnas a los tipos de datos adecuados
    for col in df.columns[2:]:
        df[col] = df[col].astype(float)

    # Interpolar los valores NaN
    df = df.interpolate(method='linear', axis=0)

    # Renombrar columnas para simplificar nombres
    df.rename(columns={
        ' speed [knots]': 'speed', 'Time': 'time', ' Altitude': 'A_1',
        ' Lon': 'lot', ' Lat': 'lat', ' T[CÂ°]': 'T_3', ' CO[ppm]': 'CO',
        ' NO2[ppm]': 'NO₂', ' C3H8(CO)[ppm]': 'Propano C₃H₈',
        ' Iso-butano(CO)[ppm]': 'Butano C₄H₁₀', ' CH4(CO)[ppm]': 'Metano CH₄',
        ' H2(NO2)[ppm]': 'H₂', ' Etanol(CO)[ppm]': 'Etanol C₂H₅OH'
    }, inplace=True)

    # Asegurar formato correcto para día, mes y hora
    def fix_date(date_str):
        day, month, year = date_str.split('-')
        return f"{int(day):02}-{int(month):02}-{int(year):02}"

    def fix_time(time_str):
        hours, minutes, seconds = time_str.split(':')
        secs, millis = seconds.split('.')
        return f"{int(hours):02}:{int(minutes):02}:{int(secs):02}.{millis}"

    # Aplicar las correcciones a las columnas de fecha y hora
    df['Date'] = df['Date'].apply(fix_date)
    df['time'] = df['time'].apply(fix_time)

    # Convertir columnas a datetime
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['time'], format='%y-%m-%d %H:%M:%S.%f', errors='coerce')

    # Manejar errores en la conversión de datetime
    if df['datetime'].isnull().any():
        print("Warning: Some datetime values could not be parsed.")

    # Localizar en UTC y convertir a UTC-5
    df['datetime'] = df['datetime'].dt.tz_localize(pytz.utc).dt.tz_convert(pytz.FixedOffset(-300))

    # Extraer la hora formateada sin la fecha
    df['time'] = df['datetime'].dt.strftime('%H:%M:%S.%f').str.slice(0, -3)
    df.dropna(subset=['time'])

    #Make sure there are no nan values

    return df

def load_data(path: str, mission: str) -> pd.DataFrame:
    """
    Carga el primer archivo CSV que comienza con el nombre de misión proporcionado.
    
    Parámetros:
    - path: Ruta del directorio.
    - mission: Prefijo del archivo (por ejemplo, 'GSULOG33').
    
    Retorna:
    - DataFrame con los datos.
    """
    pattern = os.path.join(path, mission + "*.csv")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No se encontró ningún archivo que comience con '{mission}' en {path}")
    
    return pd.read_csv(files[0])

##################Anomalies#####################

def create_binary_df(df, window_size, gases) -> pd.DataFrame:
    """
    Crea un DataFrame binario basado en las mediciones de gases para identificar anomalías con el identificador de Hampel.
    Si ya existen columnas *_anomaly en el DataFrame original, se eliminan antes de concatenar.

    Parámetros:
    - df (pd.DataFrame): DataFrame de entrada con mediciones de gases.
    - window_size (int): Tamaño de la ventana para el identificador de Hampel.
    - gases (list): Lista de gases (nombres de columna) a analizar.

    Retorna:
    - pd.DataFrame: DataFrame original con columnas adicionales *_anomaly.
    """

    df_anomalies = pd.DataFrame()

    for gas in gases:
        gas_measurements = df[gas].values
        gas_measurements_pd = pd.Series(gas_measurements)
        hampel_indx = hampel(gas_measurements_pd, window_size, 3)
        df_anomalies[gas] = [hampel_indx]

    n = len(gas_measurements)
    df_binary = pd.DataFrame()

    for gas in gases:
        index_anomalies = df_anomalies[gas][0]
        binary = np.zeros(n)
        for i in index_anomalies:
            binary[i] = 1
        df_binary[gas] = binary

    # Generar nombres de columnas *_anomaly
    gases_anomaly = [gas + "_anomaly" for gas in gases]
    df_binary.columns = gases_anomaly

    # Eliminar columnas duplicadas del df original si ya existen
    df_clean = df.drop(columns=[col for col in gases_anomaly if col in df.columns], errors="ignore")

    # Concatenar
    df_final = pd.concat([df_clean, df_binary], axis=1)

    return df_final

def signal_anomaly_neighborhood(df, gases, n, lat='lat', lon='lon') -> pd.DataFrame:
    df_neighborhoods = pd.DataFrame()

    df_neighborhoods[lon] = df[lon].rolling(n).mean()
    df_neighborhoods[lat] = df[lat].rolling(n).mean()

    for gas in gases:
        anomaly_col = gas + '_anomaly'
        neighborhood_col = gas + '_neighborhood'
        df_neighborhoods[neighborhood_col] = 0
        if anomaly_col in df.columns:
            mask = df[anomaly_col].rolling(n).sum() == n
            df_neighborhoods.loc[mask, neighborhood_col] = 1
        else:
            raise ValueError(f"Columna de anomalía no encontrada: {anomaly_col}")

    return df_neighborhoods




##################Denoising#####################

def fill_nan_with_average_and_zero(df):
    """
    Fill NaN values in a DataFrame:
    - Replace NaN with the average of adjacent values for internal NaNs.
    - Replace NaN at the start or end of a column with 0.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with potential NaN values.
    
    Returns:
    - pd.DataFrame: DataFrame with NaN values filled.
    """
    # Interpolate NaN values for internal gaps
    df_filled = df.interpolate(method='linear', axis=0, limit_direction='both')
    
    # Replace remaining NaNs (at the extremes) with 0
    df_filled = df_filled.fillna(0)
    
    return df_filled

def butterworth_filter(signal, cutoff=0.1, fs=1, order=5) -> np.ndarray:

    """
    Apply a Butterworth filter to the input signal.

    Parameters:
    - signal (array-like): The input signal to be filtered.
    - cutoff (float, optional): The cutoff frequency of the filter in Hertz.
    - fs (float, optional): The sampling frequency of the signal.
    - order (int, optional): The order of the Butterworth filter.

    Returns:
    - array-like: The filtered signal.

    The function applies a Butterworth filter to the input signal to suppress
    frequencies beyond a specified cutoff. 
    """
    #Normalizing the cutoff frequency
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    #Applying the butterworth filter
    b, a = sig.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = sig.filtfilt(b, a, signal)
    return filtered_signal

def get_denoised_signal(df,cutoff,gases) -> pd.DataFrame:
    """
    Obtain denoised signals for specified gases using a Butterworth filter.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing gas measurements.
    - cutoff (float): Cutoff frequency of the Butterworth filter.
    - gases (list): List of gases to denoise.

    Returns:
    - pd.DataFrame: DataFrame containing denoised signals for each specified gas.

    
    """
    df_denoised=pd.DataFrame()
    for gas in gases:
        gas_measurements = df[gas].values
        #finding the best order for the filter
        #Making it stop when the next order is worse than the previous one in 
        # terms of the MSE between the original and the filtered signal
        mse_prev=10000
        for order in range(1,10):
            gas_measurements_denoised=butterworth_filter(gas_measurements,cutoff,1,order)
            mse=np.mean((gas_measurements-gas_measurements_denoised)**2)
            if mse>mse_prev:
                break   
            mse_prev=mse
        #Adding the filtered signal to the df
        df_denoised[gas]=gas_measurements_denoised      
    return df_denoised


##################Classification#####################

def resample_dataframe(df, time_column, target_len) -> pd.DataFrame:
    # Convertir la columna de tiempo a datetime e indexar
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    df = df.set_index(time_column).sort_index()

    # Calcular frecuencia adecuada
    div_factor = len(df) / target_len
    div_factor = round(div_factor + 0.5)

    if div_factor < 1:
        freq = 'T'
    elif div_factor > 1:
        freq = str(int(div_factor)) + 'T'
    else:
        freq = '1T'

    # Seleccionar columnas numéricas y convertir a float
    df_numeric = df.select_dtypes(include=[np.number]).astype(float)

    # Resamplear usando promedio
    df_resampled = df_numeric.resample(freq).mean()

    # Eliminar filas completamente vacías
    df_resampled = df_resampled.dropna(how='all')

    # Restaurar columna de tiempo
    df_resampled = df_resampled.reset_index()

    return df_resampled, freq


def load_model()-> tuple:
    """
    Load a pre-trained classification model and associated scaler parameters.

    Returns:
    - tuple: A tuple containing the loaded classification model and scaler parameters.

    The function loads a pre-trained classification model and associated scaler parameters
    """
    model = pickle.load(open('Pipeline/models/kmeans_model.sav', 'rb'))
    scaler_parameters = pickle.load(open('Pipeline/models/scaler.sav', 'rb'))
    return model, scaler_parameters


def classify_route(df,speed='speed',altitude='A_1') -> int:
    """
    Classify a route as airplane or helicopter using a pre-trained K-means model.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing gas measurements.
    - speed (str, optional): Name of the column containing speed measurements.
    - altitude (str, optional): Name of the column containing altitude measurements.

    Returns:
    - int: A binary value indicating whether the route is classified as an airplane (0) or helicopter (1).

    The function takes a DataFrame with gas measurements, a column name for speed measurements,
    and a column name for altitude measurements. It resamples the speed and altitude columns to a
    target length of 65 and then uses a pre-trained K-means model to classify the route as an airplane
    or helicopter.
    """

    model,scaler=load_model()
    var=scaler.var_
    mean=scaler.mean_
    
    #If it has less than 65 values, it will be classified as 1 (helicopter)
    label=0
    #Resample the speed variable to have 65 values (minimum from previous analysis)
    if len(df)<65:
        label=1
        #print('The route is classified as helicopter')
        #print(f'The route has {len(df)} values, which is less than 63')
    else:
        resample=df[[speed,altitude,'time']]
        resample['time']=pd.date_range(start='01/01/23', periods=len(resample), freq='T')
        resample.set_index('time', inplace=True)
        resample=resample_dataframe(resample,65)
        #Cut the dataframe to have the minimum len according to the previous analysis
        min_len=62
        #print(f'The length of the dataframe is {len(resample)}')
        resample=resample.iloc[:min_len,:]
        #print(f'The length of the resampled dataframe is {len(resample)}')

        
        #Get parameters for the classification
        point=np.array([resample[speed].max(),resample[altitude].max()])
        #print(f'The speed is {point[0]} m/s and the altitude is {point[1]} m')
        point_max_speed_std=(point[0]-mean[0])/sqrt(var[0])
        point_max_alt_std=(point[1]-mean[1])/sqrt(var[1])
        point_std=np.array([point_max_speed_std,point_max_alt_std])
        # print(f'The standardized speed is {point_std[0]} and the standardized altitude is {point_std[1]}')
    
        #Predict the label
        label=model.predict(point_std.reshape(1,-1))

        #Get the index of the minimum distance
        if label == 1:
            label = 1
            #print('The route is classified as Helicopter')
        else:
            label = 0
            #print('The route is classified as Airplane')

    return label


##################Neighborhood#####################
def normalize_columns(df, col1, col2) -> pd.DataFrame:
    normalized_df = df.copy()

    for col in [col1, col2]:
        if col in normalized_df.columns:
            normalized_df[col] = normalized_df[col].fillna(0) 

            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val - min_val == 0:
                normalized_df[col] = 0  # o 1, según lo que desees
            else:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Columna '{col}' no encontrada")

    return normalized_df




##########Prediction##########
#Load the models
def load_pred_models(file):
    """
    Load a pre-trained prediction model.
    """
    model=pickle.load(open(file, 'rb'))
    return model

def normalize_columns(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    normalized_df = df.copy()

    # Normalización min-max para col1
    min_col1, max_col1 = normalized_df[col1].min(), normalized_df[col1].max()
    normalized_df[col1] = (normalized_df[col1] - min_col1) / (max_col1 - min_col1)

    # Normalización min-max para col2
    min_col2, max_col2 = normalized_df[col2].min(), normalized_df[col2].max()
    normalized_df[col2] = (normalized_df[col2] - min_col2) / (max_col2 - min_col2)

    # Manejo de valores infinitos y NaN
    normalized_df = normalized_df.replace([np.inf, -np.inf], np.nan)
    normalized_df = normalized_df.where(pd.notnull(normalized_df), None)

    return normalized_df

def predict_signal_h2(df,file,variables)-> pd.DataFrame:
    """
    Predict the H2 signal using a pre-trained model.

    Parameters:
    - df: DataFrame
    - file: str, the name of the file containing the model
    - variables: list, the list of variables to use for the prediction

    Returns:
    - DataFrame with the predicted values

    The function takes a DataFrame with gas measurements, a file containing a pre-trained model,
    and a list of variables to use for the prediction. It predicts the H2 signal and returns a DataFrame
    with the predicted values.

    """
    #Load the model
    model=load_pred_models(file)
    #Get the predictionsdf
    df_pred=pd.DataFrame()
    df = df.dropna(subset=['time'])
    df_pred['time']=df['time']
    df_pred['H2']=df['H₂']

    df_pred = df_pred.dropna(subset=['time'])
    

    # Check for any parsing errors
    if df_pred['time'].isna().any():
        print("Warning: Some time values couldn't be parsed.")
    df_pred['h2_pred']=model.predict(df[variables])
    #Normalize the variables
    df_pred=normalize_columns(df_pred,'H2','h2_pred')
    #Get the confidence interval
       
    df_pred['h2_pred_lower']=df_pred['h2_pred']-1.96*df_pred['h2_pred'].std()
    df_pred['h2_pred_upper']=df_pred['h2_pred']+1.96*df_pred['h2_pred'].std()
    
 
    #Getting confidence interval of the predictions

    #plot the predictions and the original signal with the confidence interval
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.plot(df_pred['time'], df_pred['H2'], label='Señal del sensor')
    ax.plot(df_pred['time'], df_pred['h2_pred'], label='Señal estimada')
    ax.fill_between(df_pred['time'], df_pred['h2_pred_lower'], df_pred['h2_pred_upper'], alpha=0.3)
    #Rotate the xticks
    plt.xticks(rotation=45)

    ax.set_xlabel('Hora')
    ax.set_ylabel('H₂ [ppm]')
    ax.set_xticks(df_pred['time'][::250])
    ax.legend()

    # Agregar un título al gráfico
    ax.set_title('Estimación de la señal de H₂ y su intervalo de confianza')

    plt.show()

    return df_pred, fig


def predict_signal_ch4(df,file,variables)-> pd.DataFrame:
    """
    Predict the CH4 signal using a pre-trained model.

    Parameters:
    - df: DataFrame
    - file: str, the name of the file containing the model
    - variables: list, the list of variables to use for the prediction

    Returns:
    - DataFrame with the predicted values

    The function takes a DataFrame with gas measurements, a file containing a pre-trained model,
    and a list of variables to use for the prediction. It predicts the CH4 signal and returns a DataFrame
    with the predicted values.

    """

    #Load the model
    model=load_pred_models(file)
    #Get the predictions
    df_pred=pd.DataFrame()
    df_pred['time']=df['time']
    df_pred['CH4']=df['Metano CH₄']


    df_pred = df_pred.dropna(subset=['time'])
    df = df.dropna(subset=['time'])

    # Convert 'time' to datetime
    df_pred['time'] = pd.to_datetime(df_pred['time'], format='%H:%M:%S.%f', errors='coerce')

    # Check for any parsing errors
    if df_pred['time'].isna().any():
        print("Warning: Some time values couldn't be parsed.")
    
  
    df_pred['ch4_pred']=model.predict(df[variables])
    #Reflect the signal 
    #Normalize the variables
    df_pred=normalize_columns(df_pred,'CH4','ch4_pred')
    #Get the confidence interval
       
    df_pred['ch4_pred_lower']=df_pred['ch4_pred']-1.96*df_pred['ch4_pred'].std()
    df_pred['ch4_pred_upper']=df_pred['ch4_pred']+1.96*df_pred['ch4_pred'].std()
    

    #plot the predictions and the original signal with the confidence interval
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df_pred['time'], df_pred['CH4'], label='Señal del sensor')
    ax.plot(df_pred['time'], df_pred['ch4_pred'], label='Señal estimada')
    ax.fill_between(df_pred['time'], df_pred['ch4_pred_lower'], df_pred['ch4_pred_upper'], alpha=0.3)
    ax.set_xlabel('Hora')
    ax.set_ylabel('Metano CH₄ [ppm]')
    plt.xticks(rotation=45)
    ax.set_xticks(df_pred['time'][::250])
    ax.set_title('Señal de Metano CH₄ y su intervalo de confianza')
    ax.legend()

    return df_pred, fig


##################Geopandas#####################

def df_to_geojson_anomalies(df,variables,lat_='lat',lon_='lot')-> gpd.GeoDataFrame:
    """ 
    Turn a mission in df formt into a geoson, using lat and lot variables

    Parameters:
    - df: DataFrame
    - mission: str, the name of the mission
    - variables: list, the list of variables to use for the prediction
    - lat_: str, the name of the column containing latitude measurements
    - lon_: str, the name of the column containing longitude measurements
    - path: str, the path to save the file

    Returns:
    - GeoDataFrame
        
    """
    df1=df[variables]
    df1['lat']=df[lat_]
    df1['lon']=df[lon_]
    df1['geometry'] = df1.apply(lambda x: Point((float(x.lon), float(x.lat))), axis=1)
    gdf1 = gpd.GeoDataFrame(df1, geometry='geometry')
    #gdf1.to_file(path+mission+'.geojson', driver='GeoJSON') 
    return gdf1

def df_to_geojson_neighborhood(df,gases,lat_='lat',lon_='lot',rad=0.1)-> gpd.GeoDataFrame:
    """
    Turn a binary variable neighborhood in df format into a geoson, using lat
     and lot variables, setting a geometry of a circle with a radius of 3 where 
     the value is 1

    Parameters:
    - df: DataFrame
    - gases: list, the list of gases to analyze for anomalies
    - lat_: str, the name of the column containing latitude measurements
    - lon_: str, the name of the column containing longitude measurements
    - rad: float, the radius of the circle

    Returns:
    - GeoDataFrame

    """
    df1=df[[lat_,lon_]]
    variables=[]
    for gas in gases:
        variables.append(gas+'_neighborhood')
    df1[variables]=df[variables]
    #Eliminate the rows with nan values
    df1.dropna(inplace=True)
    #Create a new column with the geometry of the point
    df1['geometry'] = df1.apply(lambda x: Point((float(x.lot), float(x.lat))), axis=1)
    gdf1 = gpd.GeoDataFrame(df1, geometry='geometry')
    #Create a new column with the geometry of a circle with a radius of 3
    gdf1['geometry']=gdf1.apply(lambda x: x.geometry.buffer(rad),axis=1)
    #Put the geometry of the circle in every column with a value of 1. If the value is 0, the geometry will be a point
    #gdf1.to_file(path+'neighborhood.geojson', driver='GeoJSON')
    return gdf1


#######################Geopandas plotter#############################

def plot_geojson(gdf_anomalies, gdf_neighborhood, gases, plot_anomalies=True, plot_neighborhood=True) -> Dict[str, plt.Figure]:
    plots_dict = {}
    # Plot the anomalies
    if plot_anomalies:
        for gas in gases:
            var_anomaly = gas + '_anomaly'
            gas_measures = gdf_anomalies[gas].values
            outliers = np.where(gdf_anomalies[var_anomaly] == 1)[0]
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            ax.plot(gdf_anomalies['time'], gas_measures)
            ax.plot(gdf_anomalies['time'][outliers], gas_measures[outliers], 'ro')
            ax.set_title('Anomalías ' + gas)
            ax.set_xlabel('Hora')
            ax.set_xticks(gdf_anomalies['time'][::250])
            ax.set_ylabel(gas+ ' [ppm]')
            plt.xticks(rotation=45)
            ax.grid(True)
            
            plots_dict[gas + '_anomalies'] = fig

    # Plot the neighborhoods
    if plot_neighborhood:
        for gas in gases:
            var_neighborhood = gas + '_neighborhood'
            filtered_gdf = gdf_neighborhood[gdf_neighborhood[var_neighborhood] == 1]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            gdf_anomalies.plot(alpha=0.5, edgecolor='k', ax=ax)
            filtered_gdf.plot(edgecolor='k', color='red', ax=ax)
            ax.set_title('Vecindarios '+ gas)
            ax.set_xlabel('Longitud')
            ax.set_ylabel('Latitud')
            
            plots_dict[gas + '_neighborhoods'] = fig

    return plots_dict


def path_plot_3d(df,gas):
    # Assuming your data is stored in a pandas DataFrame called 'df'
    # with columns: 'longitude', 'latitude', 'altitude', 'concentration'

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract the data from the DataFrame
    x = df['lot']
    y = df['lat']
    z = df['A_1']
    c = df[gas]
    # Plot the route as a line with color-coded concentration
    sc = ax.scatter(x, y, z, c=c, cmap='hot', alpha=0.8, s=30, linewidths=0.5, edgecolors='k')
    ax.plot(x, y, z, color='gray', linewidth=1.5, alpha=0.7)

    # Add a colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label(gas+' Concentration')

    # Set labels and title
    ax.set_xlabel('Longitud', fontsize=12)
    ax.set_ylabel('Latitud', fontsize=12)
    ax.set_zlabel('Altitud', fontsize=12)
    ax.set_title('Ruta y concentración de '+gas, fontsize=14, fontweight='bold')

    # Set axis tick label size
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)

    # Set the background color of the plot
    ax.set_facecolor('white')

  # Mark the first point with a marker
    ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], c='blue', marker='o', s=100)

  # Mark the last point with a marker
    ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], c='red', marker='o', s=100)
  # Show grid lines
    ax.grid(True)

    # Show axis lines
    ax.set_axis_on()

    # Set plot limits and equal aspect ratio
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(min(z), max(z))
    ax.set_box_aspect([1, 1, 1])

    # Set isometric view
    ax.view_init(elev=10, azim=130)

    return fig

# Funciones de formato
def format_date(date_str):
    """Convert a date from yy-m-d format to YYYY-MM-DD."""
    try:
        return datetime.strptime(date_str, '%y-%m-%d').strftime('%Y-%m-%d')
    except ValueError:
        return date_str  # Return original if it cannot be parsed

def format_time(time_str):
    """Convert a time string to HH:MM:SS format."""
    try:
        return datetime.strptime(time_str, '%H:%M:%S.%f').strftime('%H:%M:%S')
    except ValueError:
        try:
            return datetime.strptime(time_str, '%H:%M:%S').strftime('%H:%M:%S')
        except ValueError:
            return time_str  # Return original if it cannot be parsed

def process_geodataframe(gdf):
    """Process GeoDataFrame to format date, time, and rename columns."""
    # Formatear columnas de fecha y hora
    gdf['Date'] = gdf['Date'].apply(format_date)
    gdf['time'] = gdf['time'].apply(format_time)
    
    # Renombrar columnas
    gdf.rename(columns={
        'lot': 'Longitud',
        'lat': 'Latitud',
        'time': 'Time'
    }, inplace=True)

    # Convertir a CSV
    csv_buffer = io.StringIO()
    gdf.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    return csv_content



