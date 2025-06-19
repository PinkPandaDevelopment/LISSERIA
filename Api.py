from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, StreamingResponse
import pandas as pd
import io

from utils import sanitize_dataframe
from data_processing import (
    fill_nan_with_average_and_zero,
    get_denoised_signal,
    create_binary_df,
    classify_route,
    df_to_geojson_anomalies,
    signal_anomaly_neighborhood,
    df_to_geojson_neighborhood,
    predict_signal_h2,
    process_geodataframe,
    load_data_txt
)

# Configuración de la aplicación FastAPI
app = FastAPI(
    title="API Gases Atmosféricos",
    description="Procesamiento y visualización de datos de sensores atmosféricos para misiones aéreas.",
    version="1.0"
)

# Configuración de CORS para permitir peticiones desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload", response_class=ORJSONResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Carga y preprocesa archivos .csv o .txt.

    - Entrada: archivo .csv o .txt cargado por el usuario.
    - Proceso: lee el archivo, lo convierte en DataFrame, limpia las columnas.
    - Salida: columnas y primeras 5 filas del DataFrame.
    """
    try:
        content = await file.read()
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(".txt"):
            df = load_data_txt(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Tipo de archivo no soportado. Usa .csv o .txt")

        df = sanitize_dataframe(df)

        return {
            "columns": list(df.columns),
            "rows": df.head(5).to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar archivo: {str(e)}")


@app.post("/process", response_class=ORJSONResponse)
async def process_data(payload: dict):
    """
    Procesa datos de sensores: limpieza, suavizado, clasificación de rutas y binarización de gases.

    - Entrada: dict con 'data' (listado de registros) y 'gases' (lista de columnas de gases).
    - Salida: etiquetas de ruta, datos procesados y estadísticas.
    """
    try:
        df = pd.DataFrame(payload["data"])
        gases = payload["gases"]

        for col in ['speed', 'A_1']:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Falta la columna requerida: '{col}'")

        if "time" not in df.columns:
            df["time"] = pd.date_range(start="2023-01-01", periods=len(df), freq="T")

        df = fill_nan_with_average_and_zero(df)
        df_denoised = get_denoised_signal(df, 0.3, gases)
        df[gases] = df_denoised[gases]
        df = create_binary_df(df, 15, gases)

        route_label = classify_route(df)

        df = sanitize_dataframe(df)
        stats = sanitize_dataframe(df.describe())

        return {
            "route_label": int(route_label),
            "data": df.to_dict(orient="records"),
            "stats": stats.to_dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en procesamiento: {str(e)}")


@app.post("/geojson", response_class=ORJSONResponse)
async def generate_geojson(payload: dict):
    """
    Genera archivos GeoJSON con anomalías y resumen por vecindario.

    - Entrada: dict con 'data' y 'gases'.
    - Salida: geojson_anomalies y geojson_neighborhood.
    """
    try:
        df = pd.DataFrame(payload["data"])
        gases = payload["gases"]

        df = create_binary_df(df, 15, gases)

        gdf_anomal = df_to_geojson_anomalies(df, gases)
        signal_neigh = signal_anomaly_neighborhood(df, gases, 3)
        gdf_neigh = df_to_geojson_neighborhood(signal_neigh, gases)

        return {
            "geojson_anomalies": gdf_anomal.to_json(),
            "geojson_neighborhood": gdf_neigh.to_json()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar GeoJSON: {str(e)}")


@app.post("/predict-h2", response_class=ORJSONResponse)
async def predict_h2_api(payload: dict):
    """
    Predice la señal de H2 usando un modelo preentrenado.

    - Entrada: dict con 'data' (datos), 'model_path' (ruta al modelo) y 'predictors' (variables).
    - Salida: registros con columnas ['time', 'H2', 'h2_pred', 'h2_pred_lower', 'h2_pred_upper'].
    """
    try:
        df = pd.DataFrame(payload["data"])
        model_path = payload["model_path"]
        predictors = payload["predictors"]

        if "time" not in df.columns:
            df["time"] = pd.date_range(start="2023-01-01", periods=len(df), freq="T")

        df_pred, _ = predict_signal_h2(df, model_path, predictors)
        df_pred = sanitize_dataframe(df_pred)

        return df_pred.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/export-csv")
async def export_csv(payload: dict):
    """
    Exporta los datos en formato CSV limpio (fecha, hora, coordenadas, señales).

    - Entrada: dict con 'data'.
    - Salida: archivo CSV como `StreamingResponse`.
    """
    try:
        df = pd.DataFrame(payload["data"])
        csv_str = process_geodataframe(df)
        return StreamingResponse(io.StringIO(csv_str), media_type="text/csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exportando CSV: {str(e)}")
