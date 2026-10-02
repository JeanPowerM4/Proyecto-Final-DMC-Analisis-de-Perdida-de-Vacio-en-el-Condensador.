from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data.loader import clean_frame, to_numeric_tag
from src.features.engineering import (
    add_forecast_columns,
    chronological_split,
    degradation_class,
    forecast_feature_names,
    ready_forecast_frame,
)
from src.models.metrics import classification_scores, regression_scores
from src.monitoring.drift import population_stability_index, psi_band


def test_textos_de_adquisicion_pasan_a_nan():
    series = pd.Series(["Configure", "Bad", "Tag not found", "1.25", "  "])
    numeric = to_numeric_tag(series)
    assert numeric.isna().tolist()[:3] == [True, True, True]
    assert numeric.iloc[3] == 1.25
    assert pd.isna(numeric.iloc[4])


def test_fecha_excel_y_limpieza():
    frame = pd.DataFrame(
        {
            "Time": [45292.0, 45292.0104167, 45292.0],
            "FNX:S1_EV_P.PV": ["1.30", "Configure", "1.31"],
            "FNX:S1_EV_P.PV.1": [1.30, 1.40, 1.31],
        }
    )
    cleaned = clean_frame(frame)
    assert "FNX:S1_EV_P.PV.1" not in cleaned.columns
    assert cleaned["Time"].iloc[0] == datetime(2024, 1, 1)
    assert len(cleaned) == 2
    assert pd.isna(cleaned["FNX:S1_EV_P.PV"].iloc[1])


def test_particion_cronologica_no_mezcla_el_futuro():
    frame = pd.DataFrame(
        {
            "Time": pd.date_range("2024-01-01", periods=100, freq="15min"),
            "vacio": np.arange(100),
        }
    )
    train, val, test = chronological_split(frame, 0.70, 0.15)
    assert train["Time"].max() < val["Time"].min()
    assert val["Time"].max() < test["Time"].min()
    assert len(train) + len(val) + len(test) == 100


def test_pronostico_no_usa_el_futuro_como_feature_y_rechaza_huecos():
    times = pd.date_range("2024-01-01", periods=120, freq="15min")
    frame = pd.DataFrame(
        {
            "Time": times,
            "vacio": np.linspace(1.2, 1.5, len(times)),
            "temp_agua_mar": 60,
            "potencia_tv": 150,
            "potencia_tg11": 140,
            "potencia_tg12": 140,
            "presion_eyectores": 150,
            "presion_vapor_lp": 50,
            "flujo_fw_lp": 180,
        }
    )
    # Abre un hueco de 6 horas para que el lag deje de ser válido.
    frame.loc[frame.index > 80, "Time"] = frame.loc[frame.index > 80, "Time"] + timedelta(hours=6)
    enriched = add_forecast_columns(frame, lags=[1, 4], horizon=4, step_minutes=15)
    ready = ready_forecast_frame(enriched, lags=[1, 4])
    features = forecast_feature_names([1, 4])
    assert "vacio_futuro" not in features
    assert ready["Time"].is_monotonic_increasing
    # La fila inmediatamente posterior al hueco no puede usar el lag como si fueran 15 min.
    assert frame.loc[81, "Time"] not in set(ready["Time"])


def test_clases_de_degradacion():
    assert degradation_class(0.01, 0.05, 0.20) == "normal"
    assert degradation_class(0.10, 0.05, 0.20) == "alerta"
    assert degradation_class(0.25, 0.05, 0.20) == "critico"


def test_metricas_de_regresion_y_clasificacion():
    scores = regression_scores([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert scores["mae"] == 0
    assert scores["r2"] == 1
    labels = classification_scores(["normal", "critico"], ["normal", "alerta"])
    assert 0 <= labels["f1_macro"] <= 1


def test_psi_estable_y_con_drift():
    reference = np.random.default_rng(0).normal(size=500)
    same = population_stability_index(reference, reference, bins=10)
    shifted = population_stability_index(reference, reference + 3, bins=10)
    assert same < 0.01
    assert psi_band(same) == "estable"
    assert shifted > 0.25
    assert psi_band(shifted) == "drift"
