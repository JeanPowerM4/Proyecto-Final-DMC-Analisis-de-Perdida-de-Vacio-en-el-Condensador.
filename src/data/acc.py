"""Preparación del aerocondensador.

El archivo trae factor de capacidad, no megavatios. La potencia es CF × 300 MW,
como quedó indicado para este extracto. La contrapresión de baja presión
`TV_BP_Pout` es el análogo del vacío del condensador de superficie.
"""

import numpy as np
import pandas as pd

RATED_MW = 300

COMPASS_DEGREES = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}

ACC_FEATURES = [
    "temp_ambiente",
    "potencia_mw",
    "viento",
    "flujo_bp",
    "viento_sin",
    "viento_cos",
]
ACC_BASELINE = ["temp_ambiente", "potencia_mw"]
ACC_HOT = "temp_salida_bp"
ACC_BASELINE_NAME = "lineal_ambiente_y_carga"


def prepare_acc(df: pd.DataFrame, rated_mw: float = RATED_MW) -> pd.DataFrame:
    frame = df.copy()
    if "Time" not in frame.columns:
        frame["Time"] = pd.to_datetime(frame["date"], errors="coerce")
    else:
        frame["Time"] = pd.to_datetime(frame["Time"], errors="coerce")

    direction = frame["Wind_dir"].astype(str).str.strip()
    direction = direction.mask(direction.isin({"---", "nan", "None", ""}))
    degrees = direction.map(COMPASS_DEGREES)
    radians = np.deg2rad(degrees.astype(float))

    prepared = pd.DataFrame(
        {
            "Time": frame["Time"],
            "vacio": pd.to_numeric(frame["TV_BP_Pout"], errors="coerce"),
            "potencia_mw": pd.to_numeric(frame["CF"], errors="coerce") * float(rated_mw),
            "temp_ambiente": pd.to_numeric(frame["T_ambient"], errors="coerce"),
            "viento": pd.to_numeric(frame["Wind_speed"], errors="coerce"),
            "flujo_bp": pd.to_numeric(frame["BP_Flow"], errors="coerce"),
            "temp_salida_bp": pd.to_numeric(frame["TV_BP_Tout"], errors="coerce"),
            "viento_sin": np.sin(radians),
            "viento_cos": np.cos(radians),
            "direccion": direction.where(degrees.notna()),
            "cf": pd.to_numeric(frame["CF"], errors="coerce"),
        }
    )
    prepared = prepared.dropna(subset=["Time", "vacio", *ACC_FEATURES])
    return prepared.sort_values("Time").reset_index(drop=True)


def load_acc(csv_path: str, rated_mw: float = RATED_MW) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(csv_path)
    prepared = prepare_acc(raw, rated_mw=rated_mw)
    meta = {
        "filas_archivo": int(len(raw)),
        "filas_validas": int(len(prepared)),
        "potencia_nominal_mw": float(rated_mw),
    }
    return prepared, meta
