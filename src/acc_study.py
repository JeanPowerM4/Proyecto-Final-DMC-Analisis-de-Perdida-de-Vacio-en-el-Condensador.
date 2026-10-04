"""Mismo estudio del condensador de superficie, sobre el aerocondensador."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ROOT, load_params
from src.data.acc import (
    ACC_BASELINE,
    ACC_BASELINE_NAME,
    ACC_FEATURES,
    ACC_HOT,
    load_acc,
)
from src.features.engineering import (
    baseline_matrix,
    chronological_split,
    terminal_difference,
)
from src.monitoring.drift import population_stability_index, psi_band
from src.reporting.acc_report import write_acc_report
from src.reporting.figures import save_figures
from src.training import (
    TARGET_ALIAS,
    _jsonable,
    _learning_curves,
    _vif,
    run_classification,
    run_clusters,
    run_forecast,
    run_hypotheses,
    run_regression,
)

ACC_FORECAST = {
    "horizon_steps": 60,
    "lags": [15, 30, 60, 120],
    "step_minutes": 1,
}


def run_acc_study(params: dict | None = None) -> dict:
    params = params or load_params()
    root = Path(params["_root"])
    random_state = int(params["models"]["random_state"])
    csv_path = (ROOT / params["acc"]["csv_path"]).resolve()
    rated_mw = float(params["acc"]["rated_mw"])

    print("Cargando aerocondensador...")
    operating, meta = load_acc(str(csv_path), rated_mw=rated_mw)
    train, val, test = chronological_split(
        operating,
        float(params["split"]["train"]),
        float(params["split"]["val"]),
    )
    print(
        f"Filas válidas: {len(operating):,}. "
        f"Train {len(train):,} / val {len(val):,} / test {len(test):,}"
    )

    print("Modelos de contrapresión...")
    regression = run_regression(
        train,
        val,
        test,
        random_state,
        feature_cols=ACC_FEATURES,
        baseline_cols=ACC_BASELINE,
        baseline_name=ACC_BASELINE_NAME,
    )
    baseline = regression["baseline_train"]
    train_residual = train[TARGET_ALIAS] - baseline.predict(baseline_matrix(train, ACC_BASELINE))
    quantiles = {
        "alerta": float(np.quantile(train_residual, params["anomaly"]["alerta_quantile"])),
        "critico": float(np.quantile(train_residual, params["anomaly"]["critico_quantile"])),
    }

    print("Clasificación de degradación...")
    classification = run_classification(
        train,
        val,
        test,
        baseline,
        quantiles,
        random_state,
        feature_cols=ACC_FEATURES,
        baseline_cols=ACC_BASELINE,
    )
    print("Pronóstico a 1 hora...")
    forecast_params = {
        **params,
        "forecast": ACC_FORECAST,
    }
    forecast = run_forecast(
        operating,
        forecast_params,
        random_state,
        feature_cols=ACC_FEATURES,
        preview_points=60 * 12,
    )
    print("Clusters, pruebas y drift...")
    clusters = run_clusters(train, random_state, feature_cols=ACC_FEATURES)
    hypotheses = run_hypotheses(
        classification["frames"]["test"],
        hot_col=ACC_HOT,
        cold_col="temp_ambiente",
        exhaust_col=ACC_HOT,
        contrast_cols=["temp_ambiente", "viento", "potencia_mw"],
    )
    psi = {
        feature: {
            "psi": population_stability_index(
                train[feature].to_numpy(),
                test[feature].to_numpy(),
                bins=int(params["monitoring"]["psi_bins"]),
            ),
        }
        for feature in ACC_FEATURES + [TARGET_ALIAS]
    }
    for item in psi.values():
        item["banda"] = psi_band(item["psi"])

    def slice_profile(frame: pd.DataFrame) -> dict:
        return {
            "inicio": str(frame["Time"].min()),
            "fin": str(frame["Time"].max()),
            "vacio": float(frame[TARGET_ALIAS].median()),
            "temp_ambiente": float(frame["temp_ambiente"].median()),
            "potencia_mw": float(frame["potencia_mw"].median()),
        }

    direction = (
        operating.groupby("direccion", dropna=True)[TARGET_ALIAS]
        .median()
        .sort_values()
        .reset_index()
    )
    test_prediction = regression["modelo_comparacion"].predict(test[ACC_FEATURES])
    curves = _learning_curves(train, TARGET_ALIAS, random_state, feature_cols=ACC_FEATURES)
    vif_rows = _vif(train, feature_cols=ACC_FEATURES)
    served_name = (
        regression["mejor_modelo_flexible"]
        if regression["flexible_supera_mediana"]
        else ACC_BASELINE_NAME
    )

    summary = {
        "tecnologia": "aerocondensador",
        "filas_archivo": meta["filas_archivo"],
        "filas_limpias": meta["filas_validas"],
        "filas_operacion": int(len(operating)),
        "potencia_nominal_mw": rated_mw,
        "inicio": str(operating["Time"].min()),
        "fin": str(operating["Time"].max()),
        "train": int(len(train)),
        "val": int(len(val)),
        "test": int(len(test)),
        "particiones": {
            "train": slice_profile(train),
            "val": slice_profile(val),
            "test": slice_profile(test),
        },
        "vacio_operacion": {
            "media": float(operating[TARGET_ALIAS].mean()),
            "mediana": float(operating[TARGET_ALIAS].median()),
            "p95": float(operating[TARGET_ALIAS].quantile(0.95)),
            "min": float(operating[TARGET_ALIAS].min()),
            "max": float(operating[TARGET_ALIAS].max()),
        },
        "cf": {
            "min": float(operating["cf"].min()),
            "mediana": float(operating["cf"].median()),
            "max": float(operating["cf"].max()),
        },
        "features": ACC_FEATURES,
        "target": TARGET_ALIAS,
        "cluster_vars": ["temp_ambiente", "potencia_mw", "vacio"],
        "residual_title": "Residual en test (observado − línea ambiente/carga)",
        "regresion": regression["tabla"].to_dict(orient="records"),
        "ganador_regresion": regression["ganador_validacion"],
        "modelo_flexible": regression["mejor_modelo_flexible"],
        "flexible_supera_mediana": regression["flexible_supera_mediana"],
        "modelo_servido": served_name,
        "baseline_name": ACC_BASELINE_NAME,
        "params_xgb": regression["mejor_params_xgb"],
        "mae_cv_xgb": regression["mae_cv_xgb"],
        "importancia": regression["importancia"],
        "coef_linea_base_escalada": regression["coef_linea_base_escalada"],
        "test_modelo_desplegado": regression["test_modelo_desplegado"],
        "clasificacion": classification["tabla"].to_dict(orient="records"),
        "ganador_clasificacion": classification["ganador_validacion"],
        "modelo_matriz": classification["modelo_matriz"],
        "matriz_test": classification["matriz_test"],
        "clases": classification["clases"],
        "conteo_train": classification["conteo_train"],
        "umbral_residual": quantiles,
        "pronostico": forecast["tabla"].to_dict(orient="records"),
        "ganador_pronostico": forecast["ganador_validacion"],
        "pronostico_filas": forecast["filas"],
        "horizonte_min": forecast["horizonte_min"],
        "adf_estadistico": forecast["adf_estadistico"],
        "adf_p_valor": forecast["adf_p_valor"],
        "clusters": clusters,
        "hipotesis": hypotheses,
        "psi": psi,
        "vif": vif_rows,
        "curvas": curves,
        "vacio_por_direccion": direction.to_dict(orient="records"),
    }

    metrics_dir = root / "reports" / "aero" / "metrics"
    figures_dir = root / "reports" / "aero" / "figures"
    model_dir = root / "artifacts" / "models"
    for directory in (metrics_dir, figures_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    regression["tabla"].to_csv(metrics_dir / "regresion.csv", index=False)
    classification["tabla"].to_csv(metrics_dir / "clasificacion.csv", index=False)
    forecast["tabla"].to_csv(metrics_dir / "pronostico.csv", index=False)
    (metrics_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    test_frame = classification["frames"]["test"].copy()
    test_frame["prediccion"] = test_prediction
    test_frame["ttd"] = terminal_difference(test_frame, hot_col=ACC_HOT, cold_col="temp_ambiente")
    save_figures(summary, test_frame, forecast["preview_test"], figures_dir)
    write_acc_report(summary, root / "reports" / "aero" / "INFORME.md")

    served_pipeline = (
        regression["modelo_desplegado"] if regression["flexible_supera_mediana"] else baseline
    )
    joblib.dump(
        {
            "pipeline": served_pipeline,
            "baseline": baseline,
            "model_name": served_name,
            "features": ACC_FEATURES,
            "baseline_features": ACC_BASELINE,
            "anomaly_threshold": quantiles["critico"],
            "class_thresholds": quantiles,
        },
        model_dir / "acc_bundle.joblib",
        compress=3,
    )
    print(f"Listo aerocondensador. Ganador de validación: {regression['ganador_validacion']}")
    return summary
