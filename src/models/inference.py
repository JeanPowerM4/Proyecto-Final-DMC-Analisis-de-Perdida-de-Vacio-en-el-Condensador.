"""Lectura del artefacto y puntuación de un instante operativo."""

import math
from pathlib import Path

import joblib
import pandas as pd

from src.config import ROOT
from src.data.dictionary import API_FIELDS
from src.features.engineering import degradation_class

MODEL_PATH = ROOT / "artifacts" / "models" / "vacuum_bundle.joblib"


def load_bundle(path: Path | None = None) -> dict:
    path = Path(path) if path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No está el modelo en {path}. Ejecuta primero: python main.py train"
        )
    return joblib.load(path)


def _predict(model, frame: pd.DataFrame) -> float:
    names = list(getattr(model, "feature_names_in_", frame.columns))
    return float(model.predict(frame[names])[0])


def score_record(bundle: dict, record: dict) -> dict:
    features = {API_FIELDS[key]: record[key] for key in API_FIELDS}
    frame = pd.DataFrame([features])
    expected = _predict(bundle["pipeline"], frame)
    baseline = _predict(bundle["baseline"], frame)
    result = {
        "vacio_esperado": round(expected, 4),
        "vacio_linea_base_agua_carga": round(baseline, 4),
        "modelo": bundle["model_name"],
    }
    observed = record.get("vacio_observado")
    if observed is None or not math.isfinite(float(observed)):
        return result

    residual_model = float(observed) - expected
    residual_base = float(observed) - baseline
    threshold = float(bundle["anomaly_threshold"])
    q_alerta = float(bundle["class_thresholds"]["alerta"])
    q_critico = float(bundle["class_thresholds"]["critico"])
    result.update(
        {
            "vacio_observado": round(float(observed), 4),
            "residual_modelo": round(residual_model, 4),
            "residual_linea_base": round(residual_base, 4),
            "clase": degradation_class(residual_base, q_alerta, q_critico),
            "anomalia": bool(residual_base > threshold),
            "umbral_anomalia": round(threshold, 4),
        }
    )
    return result
