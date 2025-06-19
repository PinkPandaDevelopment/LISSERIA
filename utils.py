import pandas as pd
import numpy as np

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia un DataFrame eliminando columnas vacías, filas duplicadas,
    y reemplazando valores nulos o infinitos por NaN.

    Parámetros:
    - df (pd.DataFrame): El DataFrame a limpiar.

    Retorna:
    - pd.DataFrame: El DataFrame limpio.
    """
    # Reemplaza valores infinitos por NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Elimina columnas completamente vacías
    df.dropna(axis=1, how='all', inplace=True)

    # Elimina filas duplicadas
    df.drop_duplicates(inplace=True)

    # Reinicia el índice después de eliminar duplicados
    df.reset_index(drop=True, inplace=True)

    return df
