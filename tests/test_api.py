import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from api.dependencies import get_bundle
from api.main import app
from src.data.dictionary import BASELINE_FEATURES, FEATURES


def _bundle(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(40, len(FEATURES))), columns=FEATURES)
    frame["potencia_tv"] = np.abs(frame["potencia_tv"]) + 50
    target = 1.2 + 0.02 * frame["temp_agua_mar"] + 0.001 * frame["potencia_tv"]
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    pipeline.fit(frame[FEATURES], target)
    baseline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    baseline.fit(frame[BASELINE_FEATURES], target)
    path = tmp_path / "vacuum_bundle.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "baseline": baseline,
            "model_name": "ridge",
            "anomaly_threshold": 0.20,
            "class_thresholds": {"alerta": 0.05, "critico": 0.20},
        },
        path,
    )
    monkeypatch.setattr("src.models.inference.MODEL_PATH", path)
    get_bundle.cache_clear()
    return TestClient(app)


SNAPSHOT = {
    "temperatura_agua_mar": 60.0,
    "potencia_tv_mw": 170.0,
    "potencia_tg11_mw": 160.0,
    "potencia_tg12_mw": 155.0,
    "presion_eyectores": 150.0,
    "presion_vapor_lp": 55.0,
    "flujo_fw_lp": 180.0,
    "vacio_observado": 1.35,
}


def test_root_health_y_predict(tmp_path, monkeypatch):
    client = _bundle(tmp_path, monkeypatch)
    assert client.get("/").status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    response = client.post("/predict", json=SNAPSHOT)
    assert response.status_code == 200
    body = response.json()
    assert "vacio_esperado" in body
    assert body["clase"] in {"normal", "alerta", "critico"}
    metrics = client.get("/metrics")
    assert "condenser_predictions_total" in metrics.text


def test_predict_rechaza_potencia_negativa(tmp_path, monkeypatch):
    client = _bundle(tmp_path, monkeypatch)
    payload = {**SNAPSHOT, "potencia_tv_mw": -5}
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_csv_y_archivo_invalido(tmp_path, monkeypatch):
    client = _bundle(tmp_path, monkeypatch)
    csv_text = ",".join(SNAPSHOT) + "\n" + ",".join(str(value) for value in SNAPSHOT.values()) + "\n"
    valid = client.post(
        "/predict/batch",
        files={"file": ("lote.csv", csv_text, "text/csv")},
    )
    assert valid.status_code == 200
    assert "vacio_esperado" in valid.text
    invalid = client.post(
        "/predict/batch",
        files={"file": ("nota.txt", b"hola", "text/plain")},
    )
    assert invalid.status_code == 400
    empty = client.post(
        "/predict/batch",
        files={"file": ("vacio.csv", b"", "text/csv")},
    )
    assert empty.status_code == 400
