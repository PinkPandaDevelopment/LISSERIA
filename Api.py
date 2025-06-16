import base64
import difflib
from math import sqrt
import pickle
import tempfile
from typing import List
import unicodedata
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from shapely import Point
from data_processing import load_data_txt, load_data, classify_route, get_denoised_signal, create_binary_df, df_to_geojson_anomalies, load_model, normalize_columns, resample_dataframe, signal_anomaly_neighborhood, df_to_geojson_neighborhood, predict_signal_h2, predict_signal_ch4, plot_geojson, path_plot_3d, process_geodataframe, fill_nan_with_average_and_zero
from io import BytesIO
from fastapi import Query
from fastapi.responses import JSONResponse, StreamingResponse
import io
import json
import glob
import os
from fastapi.responses import FileResponse
import uuid
import scipy.signal as sig
import math
import geopandas as gpd



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/load_data_summary")
async def resumen_txt(file: UploadFile = File(...)):
    try:
        file_buffer = BytesIO(await file.read())
        df = load_data_txt(file_buffer)
        resumen = df.describe().to_dict()
        return resumen
    except Exception as e:
        return {"error": str(e)}
    



@app.post("/filtrar_datos")
async def filtrar_datos(
    file: UploadFile = File(...),
    variable: str = Query(...),
    hora_inicio: str = Query(default="00:00:00"),
    hora_fin: str = Query(default="23:59:59")
):
    try:
        file_buffer = BytesIO(await file.read())
        df = load_data_txt(file_buffer)
        df_filtrado = df[(df['time'] >= hora_inicio) & (df['time'] <= hora_fin)]
        if variable in df_filtrado.columns:
            return df_filtrado[['datetime', 'time', variable]].to_dict(orient="records")
        else:
            return {"error": f"La variable '{variable}' no está disponible."}
    except Exception as e:
        return {"error": str(e)}
    
    
@app.post("/exportar_csv")
async def exportar_csv(file: UploadFile = File(...)):
    try:
        file_buffer = BytesIO(await file.read())
        df = load_data_txt(file_buffer)
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)
        return StreamingResponse(stream, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=resultados.csv"})
    except Exception as e:
        return {"error": str(e)}   
    
    
    
@app.get("/load_mission_csv")
async def load_mission_csv(
    mission: str = Query(...),
    path: str = Query(...)
):
    try:
        df = load_data(path, mission)
        return {
            "archivo_cargado": mission,
            "registros": json.loads(df.to_json(orient="records"))
        }
    except Exception as e:
        return {"error": str(e)}
    
    
@app.get("/load_mission_summary", summary="Resumen estadístico de una misión CSV")
async def load_mission_summary(
    mission: str = Query(..., description="Prefijo del nombre del archivo CSV (ej. GSULOG33)"),
    path: str = Query(default="./data/", description="Ruta donde están los archivos .csv")
):
    """
    Carga el archivo CSV correspondiente a la misión y devuelve un resumen estadístico
    de las columnas numéricas.
    """
    try:
        # Buscar archivo
        pattern = os.path.join(path, mission + "*.csv")
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(f"No se encontró ningún archivo que comience con '{mission}' en {path}")

        selected_file = files[0]
        df = pd.read_csv(selected_file)

        # Generar resumen estadístico
        resumen = df.describe(include='all').fillna("").to_dict()

        return {
            "archivo_resumido": os.path.basename(selected_file),
            "resumen": resumen
        }

    except Exception as e:
        return {"error": str(e)}
    
    
@app.post("/detect_anomalies", summary="Detección de anomalías en gases con el identificador de Hampel")
async def detect_anomalies(
    file: UploadFile = File(..., description="Archivo CSV con mediciones de gases"),
    window_size: int = Form(..., description="Tamaño de ventana para el identificador de Hampel"),
    gases_str: str = Form(..., description="Lista de gases separados por coma (ej. CO, NO₂, CH₄)")
):
    """
    Este endpoint recibe un archivo CSV con mediciones de gases, una lista separada por comas y un tamaño de ventana.
    Calcula anomalías usando el identificador de Hampel y devuelve solo una muestra (primeras 100 filas).
    """
    try:
        # Leer archivo
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        # Convertir string separado por comas a lista
        gases = [g.strip() for g in gases_str.split(",")]

        # Validar existencia de columnas
        columnas_disponibles = df.columns.tolist()
        faltantes = [g for g in gases if g not in columnas_disponibles]
        if faltantes:
            return {"error": f"Columnas no encontradas: {faltantes}"}

        # Detectar anomalías
        df_resultado = create_binary_df(df, window_size, gases)

        # Devolver solo las primeras 100 filas como preview
        preview = df_resultado.head(100)
        return json.loads(preview.to_json(orient="records"))

    except Exception as e:
        return {"error": str(e)}
    
    
@app.post("/fill_missing_values", summary="Rellenar valores NaN y permitir descarga del CSV corregido")
async def fill_missing_values(file: UploadFile = File(..., description="Archivo CSV con posibles NaN")):
    """
    Este endpoint recibe un archivo CSV, rellena los valores faltantes y devuelve:
    - Una muestra de los primeros 100 registros
    - Un enlace para descargar el archivo corregido completo
    """
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        # Aplicar la corrección
        df_filled = fill_nan_with_average_and_zero(df)

        # Guardar archivo corregido en el mismo directorio del script
        output_filename = f"filled_output_{uuid.uuid4().hex}.csv"
        output_path = os.path.join(".", output_filename)
        df_filled.to_csv(output_path, index=False)

        return {
            "preview": json.loads(df_filled.head(100).to_json(orient="records")),
            "download_link": f"/download_filled_csv?filename={output_filename}"
        }

    except Exception as e:
        return {"error": str(e)}

# Endpoint de descarga
@app.get("/download_filled_csv", summary="Descargar el archivo CSV corregido")
async def download_filled_csv(filename: str = Query(..., description="Nombre del archivo generado por el servidor")):
    """
    Permite descargar el archivo generado luego de llenar los valores NaN.
    """
    filepath = os.path.join(".", filename)
    if os.path.exists(filepath):
        return FileResponse(path=filepath, filename=filename, media_type='text/csv')
    return {"error": "Archivo no encontrado"}


# Función de filtrado Butterworth
def butterworth_filter(signal, cutoff=0.1, fs=1, order=5) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='low', analog=False)
    return sig.filtfilt(b, a, signal)

# Endpoint que admite múltiples columnas
@app.post("/apply_butterworth", summary="Aplicar filtro Butterworth a múltiples columnas")
async def apply_butterworth(
    file: UploadFile = File(..., description="Archivo CSV con señales"),
    columns_str: str = Form(..., description="Lista de columnas separadas por coma (ej. CO, NO₂, CH₄)"),
    cutoff: float = Form(0.1, description="Frecuencia de corte (Hz)"),
    fs: float = Form(1.0, description="Frecuencia de muestreo (Hz)"),
    order: int = Form(5, description="Orden del filtro")
):
    """
    Aplica filtro de Butterworth a múltiples columnas de un archivo CSV.
    Devuelve un preview y link para descargar el archivo completo con columnas filtradas.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        # Procesar columnas
        columns = [c.strip() for c in columns_str.split(",")]

        # Validar existencia de columnas
        columnas_faltantes = [c for c in columns if c not in df.columns]
        if columnas_faltantes:
            return {"error": f"Estas columnas no existen en el archivo: {columnas_faltantes}"}

        # Aplicar filtro a cada columna
        for col in columns:
            df[col + "_filtered"] = butterworth_filter(df[col].values, cutoff=cutoff, fs=fs, order=order)

        # Guardar archivo corregido
        output_filename = f"filtered_output_{uuid.uuid4().hex}.csv"
        output_path = os.path.join(".", output_filename)
        df.to_csv(output_path, index=False)

        return {
            "preview": json.loads(df.head(100).to_json(orient="records")),
            "download_link": f"/download_filtered_csv?filename={output_filename}"
        }

    except Exception as e:
        return {"error": str(e)}

# Endpoint para descarga del archivo
@app.get("/download_filtered_csv", summary="Descargar CSV filtrado")
async def download_filtered_csv(filename: str):
    path = os.path.join(".", filename)
    if os.path.exists(path):
        return FileResponse(path=path, filename=filename, media_type='text/csv')
    return {"error": "Archivo no encontrado"}


@app.post("/denoise_signals", summary="Aplicar filtro adaptativo para eliminar ruido en señales de gases")
async def denoise_signals(
    file: UploadFile = File(..., description="Archivo CSV con señales de gases"),
    gases_str: str = Form(..., description="Lista de gases separada por coma (ej: CO, NO₂, CH₄)"),
    cutoff: float = Form(0.1, description="Frecuencia de corte para el filtro (Hz)")
):
    """
    Este endpoint limpia señales de gases aplicando un filtro Butterworth con orden adaptativo.
    Devuelve una muestra y un enlace para descargar el CSV completo.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        gases = [g.strip() for g in gases_str.split(",")]

        # Verificar columnas
        faltantes = [g for g in gases if g not in df.columns]
        if faltantes:
            return {"error": f"Columnas no encontradas en el archivo: {faltantes}"}

        # Aplicar filtro adaptativo
        df_denoised = get_denoised_signal(df, cutoff, gases)

        # Combinar con original
        df_final = pd.concat([df, df_denoised], axis=1)

        # Guardar en archivo temporal
        output_filename = f"denoised_output_{uuid.uuid4().hex}.csv"
        output_path = os.path.join(".", output_filename)
        df_final.to_csv(output_path, index=False)

        return {
            "preview": json.loads(df_final.head(100).to_json(orient="records")),
            "download_link": f"/download_denoised_csv?filename={output_filename}"
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/download_denoised_csv", summary="Descargar señales de gases denoised")
async def download_denoised_csv(filename: str):
    path = os.path.join(".", filename)
    if os.path.exists(path):
        return FileResponse(path=path, filename=filename, media_type="text/csv")
    return {"error": "Archivo no encontrado"}

@app.post("/resample_dataframe", summary="Resamplear DataFrame a longitud deseada con promedio")
async def resample_endpoint(
    file: UploadFile = File(..., description="Archivo CSV con columna de tiempo"),
    time_column: str = Form(..., description="Nombre de la columna de tiempo (ej: datetime)"),
    target_len: int = Form(..., description="Longitud deseada del DataFrame")
):
    """
    Este endpoint re-muestrea un DataFrame temporal a una longitud objetivo usando promedio.
    Devuelve una muestra del resultado y un link para descargar el archivo completo.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        if time_column not in df.columns:
            return {"error": f"La columna de tiempo '{time_column}' no existe en el archivo."}

        # Convertir a datetime e indexar
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        df = df.set_index(time_column)
        df = df.sort_index()

        # Aplicar resampleo
        df_resampled = resample_dataframe(df, target_len)

        # Guardar archivo temporal
        output_filename = f"resampled_output_{uuid.uuid4().hex}.csv"
        output_path = os.path.join(".", output_filename)
        df_resampled.to_csv(output_path)

        return {
            "preview": json.loads(df_resampled.head(100).to_json(orient="records")),
            "download_link": f"/download_resampled_csv?filename={output_filename}"
        }

    except Exception as e:
        return {"error": str(e)}

# Endpoint de descarga
@app.get("/download_resampled_csv", summary="Descargar DataFrame re-muestreado")
async def download_resampled_csv(filename: str):
    path = os.path.join(".", filename)
    if os.path.exists(path):
        return FileResponse(path=path, filename=filename, media_type="text/csv")
    return {"error": "Archivo no encontrado"}


@app.post("/resample_dataframe", summary="Resamplear DataFrame a longitud deseada (promedio)")
async def resample_endpoint(
    file: UploadFile = File(..., description="Archivo CSV con columna de tiempo"),
    time_column: str = Form(..., description="Nombre de la columna de tiempo (ej: datetime)"),
    target_len: int = Form(..., description="Longitud deseada del DataFrame")
):
    """
    Re-muestrea un DataFrame temporal a una longitud objetivo usando promedio.
    Solo columnas numéricas se promedian. Se conserva la columna de tiempo.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        if time_column not in df.columns:
            return {"error": f"La columna de tiempo '{time_column}' no existe en el archivo."}

        df_resampled, freq_used = resample_dataframe(df, time_column, target_len)

        # Guardar archivo
        output_filename = f"resampled_output_{uuid.uuid4().hex}.csv"
        output_path = os.path.join(".", output_filename)
        df_resampled.to_csv(output_path, index=False)

        return {
            "resample_freq": freq_used,
            "preview": json.loads(df_resampled.head(100).to_json(orient="records")),
            "download_link": f"/download_resampled_csv?filename={output_filename}"
        }

    except Exception as e:
        return {"error": str(e)}

# Endpoint descarga
@app.get("/download_resampled_csv", summary="Descargar DataFrame resampleado")
async def download_resampled_csv(filename: str):
    path = os.path.join(".", filename)
    if os.path.exists(path):
        return FileResponse(path=path, filename=filename, media_type="text/csv")
    return {"error": "Archivo no encontrado"}

# Función auxiliar para renombrar columnas
def map_columns(df, mapping):
    df = df.copy()
    mapeo_realizado = {}
    for alias, original in mapping.items():
        if original in df.columns:
            df.rename(columns={original: alias}, inplace=True)
            mapeo_realizado[alias] = original
    return df, mapeo_realizado

@app.post("/classify_route", summary="Clasifica una ruta como avión o helicóptero")
async def classify_route_endpoint(
    file: UploadFile = File(..., description="Archivo CSV con datos de la ruta"),
    speed_column: str = Form(default="max_speed"),
    altitude_column: str = Form(default="A_1"),
    time_column: str = Form(default="datetime"),
    target_len: int = Form(default=65)
):
    """
    Clasifica una ruta como avión (0) o helicóptero (1) utilizando un modelo KMeans.
    """
    try:
        # Leer archivo CSV
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        df.columns = df.columns.str.strip()

        # Intentar mapear columnas si no existen directamente
        df, mapeo = map_columns(df, {
            'speed': speed_column,
            'altitude': altitude_column,
            'time': time_column
        })

        # Verificar que las columnas requeridas estén presentes
        columnas_requeridas = ['speed', 'altitude', 'time']
        faltantes = [col for col in columnas_requeridas if col not in df.columns]
        if faltantes:
            return {
                "error": f"Faltan columnas requeridas para el modelo: {faltantes}",
                "columnas_encontradas": df.columns.tolist(),
                "mapeo_realizado": mapeo
            }

        # Cargar modelo y scaler
        model, scaler = load_model()
        var, mean = scaler.var_, scaler.mean_

        # Si la ruta es muy corta, clasificar automáticamente como helicóptero
        if len(df) < target_len:
            return {
                "label": 1,
                "clase": "Helicóptero",
                "razon": f"La ruta tiene solo {len(df)} registros (mínimo requerido: {target_len}).",
                "mapeo_columnas": mapeo
            }

        # Resamplear dataframe con columnas requeridas
        df_filtrado = df[['speed', 'altitude', 'time']].copy()
        df_resampled, freq = resample_dataframe(df_filtrado, time_column='time', target_len=target_len)
        df_resampled = df_resampled.iloc[:target_len - 3, :]

        # Obtener valores máximos
        max_speed = df_resampled['speed'].max()
        max_alt = df_resampled['altitude'].max()

        # Generar advertencias si hay valores 0
        advertencias = []
        if max_speed == 0:
            advertencias.append("La velocidad máxima es 0. Verifica la columna de velocidad.")
        if max_alt == 0:
            advertencias.append("La altitud máxima es 0. Verifica la columna de altitud.")

        # Estandarizar valores
        speed_std = (max_speed - mean[0]) / sqrt(var[0])
        alt_std = (max_alt - mean[1]) / sqrt(var[1])
        point_std = np.array([speed_std, alt_std]).reshape(1, -1)

        # Clasificar con modelo
        label = int(model.predict(point_std)[0])
        clase = "Avión" if label == 0 else "Helicóptero"

        return {
            "label": label,
            "clase": clase,
            "valores": {
                "max_speed": max_speed,
                "max_altitude": max_alt,
                "speed_std": speed_std,
                "altitude_std": alt_std
            },
            "advertencias": advertencias if advertencias else None,
            "mapeo_columnas": mapeo
        }

    except Exception as e:
        return {"error": str(e)}
    
##################Neighborhood#####################
@app.post("/anomaly_neighborhoods", summary="Detecta vecindarios de anomalías en señales")
async def signal_anomaly_neighborhood_endpoint(
    file: UploadFile = File(...),
    gases: List[str] = Form(...),
    n: int = Form(...),
    lat: str = Form(default="lat"),
    lon: str = Form(default="lon")
):
    try:
        # Arreglo para corregir envío como string plano
        if len(gases) == 1 and "," in gases[0]:
            gases = [g.strip() for g in gases[0].split(",")]

        # Leer el archivo CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Validar existencia de columnas necesarias
        missing_cols = []
        for gas in gases:
            anomaly_col = f"{gas}_anomaly"
            if anomaly_col not in df.columns:
                missing_cols.append(anomaly_col)
        if lat not in df.columns:
            missing_cols.append(lat)
        if lon not in df.columns:
            missing_cols.append(lon)
        if missing_cols:
            return {
                "error": "Faltan columnas necesarias",
                "columnas_faltantes": missing_cols,
                "columnas_disponibles": df.columns.tolist()
            }

        # Ejecutar análisis de vecindarios
        df_neigh = signal_anomaly_neighborhood(df, gases, n, lat=lat, lon=lon)

        return {
            "mensaje": "Cálculo realizado correctamente",
            "columnas": df_neigh.columns.tolist(),
            "preview": df_neigh.head(10).to_dict(orient="records")  # solo 10 filas para evitar cuelgues
        }

    except Exception as e:
        return {"error": str(e)}
        
##########Prediction##########
#Load the models

@app.post("/load-model/")
async def load_pred_models(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        model = pickle.loads(contents)

        # Podemos hacer una verificación básica del tipo de modelo
        model_type = str(type(model))
        return {"message": "Modelo cargado exitosamente", "tipo_modelo": model_type}

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"No se pudo cargar el modelo: {str(e)}"})
    

@app.post("/normalize-columns/")
async def normalize_columns_api(
    file: UploadFile = File(...),
    col1: str = Form(...),
    col2: str = Form(...)
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Verificar existencia
        if col1 not in df.columns or col2 not in df.columns:
            return JSONResponse(status_code=400, content={
                "error": "Una o ambas columnas no existen.",
                "columnas_disponibles": df.columns.tolist()
            })

        # Asegurarse de que sean columnas numéricas
        if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
            return JSONResponse(status_code=400, content={
                "error": "Una o ambas columnas no son numéricas.",
                "tipo_col1": str(df[col1].dtype),
                "tipo_col2": str(df[col2].dtype)
            })

        # Normalización
        def min_max_normalize(series):
            min_val, max_val = series.min(), series.max()
            if min_val == max_val:
                return None, None, None  # No normalizable
            return (series - min_val) / (max_val - min_val), min_val, max_val

        col1_norm, min1, max1 = min_max_normalize(df[col1])
        col2_norm, min2, max2 = min_max_normalize(df[col2])

        if col1_norm is None:
            return JSONResponse(status_code=400, content={
                "error": f"No se puede normalizar porque el rango de {col1} es cero.",
                "min_col1": float(min1),
                "max_col1": float(max1)
            })

        if col2_norm is None:
            return JSONResponse(status_code=400, content={
                "error": f"No se puede normalizar porque el rango de {col2} es cero.",
                "min_col2": float(min2),
                "max_col2": float(max2)
            })

        # Reemplazar NaN/infinito
        df_resultado = pd.DataFrame({
            col1: col1_norm.replace([np.inf, -np.inf], np.nan).fillna(value=np.nan).replace({np.nan: None}),
            col2: col2_norm.replace([np.inf, -np.inf], np.nan).fillna(value=np.nan).replace({np.nan: None})
        })

        return JSONResponse(content=df_resultado.to_dict(orient="records"))

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    
@app.post("/predict-h2/")
async def predict_signal_h2_api(
    data_file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    variables: str = Form(...)
):
    try:
        # Leer CSV
        data_bytes = await data_file.read()
        df = pd.read_csv(io.StringIO(data_bytes.decode("utf-8")))

        # Parsear variables (vienen como string tipo: "col1,col2,col3")
        var_list = [v.strip() for v in variables.split(",")]

        # Verificación de columnas requeridas
        if 'time' not in df.columns or 'H₂' not in df.columns:
            return JSONResponse(status_code=400, content={
                "error": "Faltan columnas requeridas: 'time' y 'H₂'.",
                "columnas_disponibles": df.columns.tolist()
            })

        df = df.dropna(subset=['time'])
        df_pred = pd.DataFrame()
        df_pred['time'] = pd.to_datetime(df['time'], errors='coerce')
        df_pred['H2'] = df['H₂']
        df_pred = df_pred.dropna(subset=['time', 'H2'])

        # Guardar modelo temporalmente para cargarlo con pickle
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await model_file.read())
            tmp_path = tmp.name

        model = load_pred_models(tmp_path)

        # Predecir
        df_pred['h2_pred'] = model.predict(df[var_list])

        # Normalizar
        df_pred = normalize_columns(df_pred, 'H2', 'h2_pred')

        # Intervalo de confianza
        std_dev = df_pred['h2_pred'].std()
        df_pred['h2_pred_lower'] = df_pred['h2_pred'] - 1.96 * std_dev
        df_pred['h2_pred_upper'] = df_pred['h2_pred'] + 1.96 * std_dev

        # Crear figura
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(df_pred['time'], df_pred['H2'], label='Señal del sensor')
        ax.plot(df_pred['time'], df_pred['h2_pred'], label='Señal estimada')
        ax.fill_between(df_pred['time'], df_pred['h2_pred_lower'], df_pred['h2_pred_upper'], alpha=0.3)
        ax.set_xlabel('Hora')
        ax.set_ylabel('H₂ [ppm]')
        ax.set_title('Estimación de la señal de H₂')
        ax.legend()
        plt.xticks(rotation=45)

        # Convertir a base64
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches="tight")
        plt.close(fig)
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

        # Respuesta JSON con predicciones y gráfico
        return {
            "predicciones": df_pred.to_dict(orient="records"),
            "grafico_base64": img_base64
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict-ch4/")
async def predict_signal_ch4_api(
    data_file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    variables: str = Form(...)
):
    try:
        # Leer CSV
        data_bytes = await data_file.read()
        df = pd.read_csv(io.StringIO(data_bytes.decode("utf-8")))

        # Parsear variables (vienen como "var1,var2,var3")
        var_list = [v.strip() for v in variables.split(",")]

        # Verificar columnas necesarias
        if 'time' not in df.columns or 'Metano CH₄' not in df.columns:
            return JSONResponse(status_code=400, content={
                "error": "Faltan columnas requeridas: 'time' y 'Metano CH₄'.",
                "columnas_disponibles": df.columns.tolist()
            })

        # Preparar DataFrame para predicción
        df_pred = pd.DataFrame()
        df_pred['time'] = df['time']
        df_pred['CH4'] = df['Metano CH₄']

        df_pred = df_pred.dropna(subset=['time', 'CH4'])
        df = df.dropna(subset=['time'])

        # Convertir 'time' a datetime
        df_pred['time'] = pd.to_datetime(df_pred['time'], format='%H:%M:%S.%f', errors='coerce')
        if df_pred['time'].isna().any():
            print("Advertencia: Algunos valores de tiempo no se pudieron convertir.")

        # Guardar modelo temporalmente
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await model_file.read())
            model_path = tmp.name

        model = load_pred_models(model_path)

        # Predecir
        df_pred['ch4_pred'] = model.predict(df[var_list])

        # Normalizar
        df_pred = normalize_columns(df_pred, 'CH4', 'ch4_pred')

        # Calcular intervalos de confianza
        std_dev = df_pred['ch4_pred'].std()
        df_pred['ch4_pred_lower'] = df_pred['ch4_pred'] - 1.96 * std_dev
        df_pred['ch4_pred_upper'] = df_pred['ch4_pred'] + 1.96 * std_dev

        # Crear gráfico
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(df_pred['time'], df_pred['CH4'], label='Señal del sensor')
        ax.plot(df_pred['time'], df_pred['ch4_pred'], label='Señal estimada')
        ax.fill_between(df_pred['time'], df_pred['ch4_pred_lower'], df_pred['ch4_pred_upper'], alpha=0.3)
        ax.set_xlabel('Hora')
        ax.set_ylabel('Metano CH₄ [ppm]')
        ax.set_title('Señal de Metano CH₄ y su intervalo de confianza')
        ax.legend()
        plt.xticks(rotation=45)

        # Convertir imagen a base64
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches="tight")
        plt.close(fig)
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

        return {
            "predicciones": df_pred.to_dict(orient="records"),
            "grafico_base64": img_base64
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    
    
@app.post("/predict-ch4/")
async def predict_signal_ch4_api(
    data_file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    variables: str = Form(...)
):
    try:
        # Leer CSV
        data_bytes = await data_file.read()
        df = pd.read_csv(io.StringIO(data_bytes.decode("utf-8")))

        # Parsear variables (vienen como "var1,var2,var3")
        var_list = [v.strip() for v in variables.split(",")]

        # Verificar columnas necesarias
        if 'time' not in df.columns or 'Metano CH₄' not in df.columns:
            return JSONResponse(status_code=400, content={
                "error": "Faltan columnas requeridas: 'time' y 'Metano CH₄'.",
                "columnas_disponibles": df.columns.tolist()
            })

        # Preparar DataFrame para predicción
        df_pred = pd.DataFrame()
        df_pred['time'] = df['time']
        df_pred['CH4'] = df['Metano CH₄']

        df_pred = df_pred.dropna(subset=['time', 'CH4'])
        df = df.dropna(subset=['time'])

        # Convertir 'time' a datetime
        df_pred['time'] = pd.to_datetime(df_pred['time'], format='%H:%M:%S.%f', errors='coerce')
        if df_pred['time'].isna().any():
            print("Advertencia: Algunos valores de tiempo no se pudieron convertir.")

        # Guardar modelo temporalmente
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await model_file.read())
            model_path = tmp.name

        model = load_pred_models(model_path)

        # Predecir
        df_pred['ch4_pred'] = model.predict(df[var_list])

        # Normalizar
        df_pred = normalize_columns(df_pred, 'CH4', 'ch4_pred')

        # Calcular intervalos de confianza
        std_dev = df_pred['ch4_pred'].std()
        df_pred['ch4_pred_lower'] = df_pred['ch4_pred'] - 1.96 * std_dev
        df_pred['ch4_pred_upper'] = df_pred['ch4_pred'] + 1.96 * std_dev

        # Crear gráfico
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(df_pred['time'], df_pred['CH4'], label='Señal del sensor')
        ax.plot(df_pred['time'], df_pred['ch4_pred'], label='Señal estimada')
        ax.fill_between(df_pred['time'], df_pred['ch4_pred_lower'], df_pred['ch4_pred_upper'], alpha=0.3)
        ax.set_xlabel('Hora')
        ax.set_ylabel('Metano CH₄ [ppm]')
        ax.set_title('Señal de Metano CH₄ y su intervalo de confianza')
        ax.legend()
        plt.xticks(rotation=45)

        # Convertir imagen a base64
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches="tight")
        plt.close(fig)
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

        return {
            "predicciones": df_pred.to_dict(orient="records"),
            "grafico_base64": img_base64
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
        


    
