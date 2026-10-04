"""Reglas de régimen, partición temporal, clases y ventanas de pronóstico."""

import numpy as np
import pandas as pd

from src.data.dictionary import (
    BASELINE_FEATURES,
    CONDENSER_TEMP_ALIAS,
    FEATURES,
    TARGET_ALIAS,
)


def operating_frame(
    df: pd.DataFrame,
    min_tv_mw: float,
    min_vacuum: float,
    max_vacuum: float,
) -> pd.DataFrame:
    """Conserva la turbina de vapor en servicio y el vacío dentro de banda operativa.

    El cluster cercano a 30 (atmosférico) es la unidad fuera de servicio. Mezclarlo
    con la operación normal hace que cualquier modelo 'detecte' una parada, que no
    es la pérdida de vacío que interesa en performance.
    """
    mask = (
        df["potencia_tv"].ge(min_tv_mw)
        & df[TARGET_ALIAS].between(min_vacuum, max_vacuum)
        & df[FEATURES].notna().all(axis=1)
    )
    return df.loc[mask].sort_values("Time").reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("Las fracciones train/val deben dejar un bloque de test.")
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Alguna partición temporal quedó vacía.")
    return train, val, test


def degradation_class(residual: float, q_alerta: float, q_critico: float) -> str:
    if residual <= q_alerta:
        return "normal"
    if residual <= q_critico:
        return "alerta"
    return "critico"


def assign_degradation(
    df: pd.DataFrame,
    baseline_prediction: np.ndarray,
    q_alerta: float,
    q_critico: float,
) -> pd.DataFrame:
    frame = df.copy()
    frame["vacio_base"] = baseline_prediction
    frame["residual_base"] = frame[TARGET_ALIAS] - frame["vacio_base"]
    frame["clase"] = [
        degradation_class(value, q_alerta, q_critico)
        for value in frame["residual_base"]
    ]
    return frame


def terminal_difference(
    df: pd.DataFrame,
    hot_col: str = CONDENSER_TEMP_ALIAS,
    cold_col: str = "temp_agua_mar",
) -> pd.Series:
    """Diferencia entre la temperatura caliente del condensador y el fluido frío.

    No entra al modelo. Sirve para contrastar si el residual de vacío coincide
    con un lado caliente más alto de lo que el fluido frío explica.
    """
    if hot_col not in df.columns or cold_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[hot_col] - df[cold_col]


def add_forecast_columns(
    df: pd.DataFrame,
    lags: list[int],
    horizon: int,
    step_minutes: float,
) -> pd.DataFrame:
    """Arma un problema supervisado usando solo información conocida en t.

    Un lag solo es válido si el reloj realmente retrocedió `lag` pasos de 15 min.
    Así una parada no se disfraza de observación anterior.
    """
    frame = df.sort_values("Time").copy()
    tolerance = 0.25
    for lag in lags:
        frame[f"vacio_lag_{lag}"] = frame[TARGET_ALIAS].shift(lag)
        delta = (frame["Time"] - frame["Time"].shift(lag)).dt.total_seconds() / 60
        low = lag * step_minutes * (1 - tolerance)
        high = lag * step_minutes * (1 + tolerance)
        frame[f"ok_lag_{lag}"] = delta.between(low, high)

    frame["vacio_futuro"] = frame[TARGET_ALIAS].shift(-horizon)
    future_delta = (frame["Time"].shift(-horizon) - frame["Time"]).dt.total_seconds() / 60
    frame["ok_futuro"] = future_delta.between(
        horizon * step_minutes * (1 - tolerance),
        horizon * step_minutes * (1 + tolerance),
    )
    frame["vacio_ahora"] = frame[TARGET_ALIAS]
    return frame


def forecast_feature_names(lags: list[int], features: list[str] | None = None) -> list[str]:
    columns = FEATURES if features is None else features
    return [*columns, *[f"vacio_lag_{lag}" for lag in lags]]


def ready_forecast_frame(
    df: pd.DataFrame,
    lags: list[int],
    features: list[str] | None = None,
) -> pd.DataFrame:
    ok_cols = [f"ok_lag_{lag}" for lag in lags] + ["ok_futuro"]
    mask = df[ok_cols].all(axis=1) & df["vacio_futuro"].notna()
    columns = ["Time", "vacio_futuro", "vacio_ahora", *forecast_feature_names(lags, features)]
    seasonal = "vacio_lag_96"
    if seasonal in df.columns and seasonal not in columns:
        columns.append(seasonal)
    return df.loc[mask, columns].reset_index(drop=True)


def baseline_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    return df[BASELINE_FEATURES if columns is None else columns]
