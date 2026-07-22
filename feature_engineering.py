import pandas as pd
import numpy as np


WINDOW = 20


def crear_dataset(historial):

    if len(historial) <= WINDOW:
        return pd.DataFrame()

    datos = []

    for i in range(WINDOW, len(historial)):

        ventana = historial[i-WINDOW:i]

        fila = {
            "promedio5": np.mean(ventana[-5:]),
            "promedio10": np.mean(ventana[-10:]),
            "promedio20": np.mean(ventana),

            "maximo": np.max(ventana),
            "minimo": np.min(ventana),

            "desviacion": np.std(ventana),

            "menor2": sum(x < 2 for x in ventana),
            "entre2y5": sum(2 <= x < 5 for x in ventana),
            "mayor5": sum(x >= 5 for x in ventana),
            "mayor10": sum(x >= 10 for x in ventana),
            "mayor20": sum(x >= 20 for x in ventana),

            "ultima": ventana[-1],
            "objetivo": 1 if historial[i] >= 10 else 0
        }

        datos.append(fila)

    return pd.DataFrame(datos)
