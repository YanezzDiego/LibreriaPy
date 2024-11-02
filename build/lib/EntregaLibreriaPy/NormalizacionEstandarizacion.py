import pandas as pd
import numpy as np

def transform_data(data, method="normalizacion"):
    # Validar el método seleccionado
    if method not in ["normalizacion", "estandarizacion"]:
        raise ValueError("El metodo debe ser 'normalizacion' o 'estandarizacion'")

    # Función para estandarizar entre 0 y 1
    def estandarizacion(x):
        x_estandarizado = (x - x.min()) / (x.max() - x.min())
        return x_estandarizado

    # Función para normalizar a media 0 y desviación estándar 1
    def normalizacion(v):
        vmedio = v.mean()
        dest = v.std()
        v_normalizado = (v - vmedio) / dest
        return v_normalizado

    # Aplicar la transformación a cada columna numérica del dataset
    for col in data.columns:
        if np.issubdtype(data[col].dtype, np.number):
            if method == "estandarizacion":
                data[col] = estandarizacion(data[col])
                print(f"La variable {col} ha sido estandarizada con éxito")
            elif method == "normalizacion":
                data[col] = normalizacion(data[col])
                print(f"La variable {col} ha sido normalizada con éxito")

    return data
