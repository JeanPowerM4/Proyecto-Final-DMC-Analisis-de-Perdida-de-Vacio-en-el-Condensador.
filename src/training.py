"""Entrenamiento, comparación y artefactos del estudio de vacío."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, learning_curve
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBClassifier, XGBRegressor

from src.data.dictionary import BASELINE_FEATURES, EXHAUST_TEMP_ALIAS, FEATURES, TARGET_ALIAS
from src.data.loader import load_historian
from src.features.engineering import (
    add_forecast_columns,
    assign_degradation,
    baseline_matrix,
    chronological_split,
    forecast_feature_names,
    operating_frame,
    ready_forecast_frame,
    terminal_difference,
)
from src.models.metrics import classification_scores, regression_scores
from src.monitoring.drift import population_stability_index, psi_band
from src.reporting.figures import save_figures
from src.reporting.report import write_report

CLASS_ORDER = ["normal", "alerta", "critico"]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _scaled_linear() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def _ridge() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def _xgb_regressor(random_state: int, **kwargs) -> Pipeline:
    params = {
        "n_estimators": 160,
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": 4,
        "random_state": random_state,
    }
    params.update(kwargs)
    return Pipeline([("model", XGBRegressor(**params))])


def _score_splits(model, splits: dict, target: str) -> list[dict]:
    rows = []
    for name, frame in splits.items():
        prediction = model.predict(frame[FEATURES])
        rows.append(
            {
                "modelo": model.__dict__.get("_study_name", "modelo"),
                "particion": name,
                **regression_scores(frame[target], prediction),
            }
        )
    return rows


def _fit_named(name: str, estimator, train: pd.DataFrame, target: str):
    model = clone(estimator)
    model.fit(train[FEATURES], train[target])
    model._study_name = name
    return model


def _importance(estimator, names: list[str]) -> list[dict]:
    model = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_)).astype(float)
    else:
        return []
    order = np.argsort(values)[::-1]
    return [
        {"variable": names[index], "importancia": float(values[index])}
        for index in order
    ]


def _vif(frame: pd.DataFrame, feature_cols: list[str] | None = None) -> list[dict]:
    feature_cols = FEATURES if feature_cols is None else feature_cols
    import statsmodels.api as sm

    matrix = sm.add_constant(frame[feature_cols].astype(float), has_constant="add")
    rows = []
    for index, name in enumerate(matrix.columns):
        if name == "const":
            continue
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        rows.append(
            {
                "variable": str(name),
                "vif": float(variance_inflation_factor(matrix.values, index)),
            }
        )
    return rows


def _mann_whitney(left: pd.Series, right: pd.Series) -> dict:
    left = left.dropna()
    right = right.dropna()
    if len(left) < 20 or len(right) < 20:
        return {"n_normal": int(len(left)), "n_critico": int(len(right)), "p_valor": None}
    stat = mannwhitneyu(left, right, alternative="two-sided")
    return {
        "n_normal": int(len(left)),
        "n_critico": int(len(right)),
        "mediana_normal": float(left.median()),
        "mediana_critico": float(right.median()),
        "estadistico": float(stat.statistic),
        "p_valor": float(stat.pvalue),
    }


def _learning_curves(
    train: pd.DataFrame,
    target: str,
    random_state: int,
    feature_cols: list[str] | None = None,
) -> dict:
    feature_cols = FEATURES if feature_cols is None else feature_cols
    estimators = {
        "ridge": _ridge(),
        "arbol_poco_profundo": DecisionTreeRegressor(max_depth=3, random_state=random_state),
        "arbol_profundo": DecisionTreeRegressor(max_depth=12, random_state=random_state),
    }
    curves = {}
    splitter = TimeSeriesSplit(n_splits=3)
    for name, estimator in estimators.items():
        sizes, train_score, val_score = learning_curve(
            estimator,
            train[feature_cols],
            train[target],
            train_sizes=np.linspace(0.2, 1.0, 5),
            cv=splitter,
            scoring="neg_mean_absolute_error",
            shuffle=False,
            n_jobs=1,
        )
        curves[name] = {
            "muestras": [int(size) for size in sizes],
            "mae_train": [float(value) for value in -train_score.mean(axis=1)],
            "mae_val": [float(value) for value in -val_score.mean(axis=1)],
        }
    return curves


def run_regression(
    train,
    val,
    test,
    random_state: int,
    feature_cols: list[str] | None = None,
    baseline_cols: list[str] | None = None,
    baseline_name: str = "lineal_agua_y_carga",
) -> dict:
    feature_cols = FEATURES if feature_cols is None else feature_cols
    baseline_cols = BASELINE_FEATURES if baseline_cols is None else baseline_cols
    y_name = TARGET_ALIAS
    splits = {"val": val, "test": test}
    rows = []
    fitted = {}

    median_value = float(train[y_name].median())
    for part_name, frame in splits.items():
        prediction = np.full(len(frame), median_value)
        rows.append(
            {
                "modelo": "mediana_entrenamiento",
                "particion": part_name,
                **regression_scores(frame[y_name], prediction),
            }
        )

    baseline = _scaled_linear()
    baseline.fit(baseline_matrix(train, baseline_cols), train[y_name])
    for part_name, frame in {**{"train": train}, **splits}.items():
        prediction = baseline.predict(baseline_matrix(frame, baseline_cols))
        if part_name == "train":
            continue
        rows.append(
            {
                "modelo": baseline_name,
                "particion": part_name,
                **regression_scores(frame[y_name], prediction),
            }
        )

    catalog = {
        "ridge": _ridge(),
        "arbol": DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=8,
            n_jobs=4,
            random_state=random_state,
        ),
        "gbm": GradientBoostingRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.08,
            random_state=random_state,
        ),
        "xgboost": _xgb_regressor(random_state),
    }
    for name, estimator in catalog.items():
        print(f"  regresión: {name}")
        model = estimator if name == "xgboost" else clone(estimator)
        if name != "xgboost":
            model = clone(estimator)
        else:
            model = clone(estimator)
        model.fit(train[feature_cols], train[y_name])
        fitted[name] = model
        for part_name, frame in splits.items():
            rows.append(
                {
                    "modelo": name,
                    "particion": part_name,
                    **regression_scores(frame[y_name], model.predict(frame[feature_cols])),
                }
            )

    print("  regresión: xgboost_gridsearch")
    search = GridSearchCV(
        _xgb_regressor(random_state, n_estimators=120),
        param_grid={
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
        },
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        refit=True,
    )
    search.fit(train[feature_cols], train[y_name])
    fitted["xgboost_gridsearch"] = search.best_estimator_
    for part_name, frame in splits.items():
        rows.append(
            {
                "modelo": "xgboost_gridsearch",
                "particion": part_name,
                **regression_scores(frame[y_name], search.predict(frame[feature_cols])),
            }
        )

    table = pd.DataFrame(rows)
    val_fitted = table[
        (table["particion"] == "val") & (table["modelo"].isin(fitted))
    ].sort_values("mae")
    winner_model_name = str(val_fitted.iloc[0]["modelo"])
    winner_model = fitted[winner_model_name]
    val_all = table[table["particion"] == "val"].sort_values("mae")
    winner = str(val_all.iloc[0]["modelo"])
    median_val = float(
        table[(table["modelo"] == "mediana_entrenamiento") & (table["particion"] == "val")]["mae"].iloc[0]
    )
    flexible_beats_median = float(val_fitted.iloc[0]["mae"]) < median_val

    deployed = clone(winner_model)
    train_val = pd.concat([train, val], axis=0)
    deployed.fit(train_val[feature_cols], train_val[y_name])
    deployed_test = regression_scores(test[y_name], deployed.predict(test[feature_cols]))

    signed = {}
    linear = baseline.named_steps["model"]
    for name, coef in zip(baseline_cols, np.ravel(linear.coef_), strict=True):
        signed[name] = float(coef)

    return {
        "tabla": table,
        "ganador_validacion": winner,
        "mejor_modelo_flexible": winner_model_name,
        "flexible_supera_mediana": flexible_beats_median,
        "mejor_params_xgb": {key: _jsonable(value) for key, value in search.best_params_.items()},
        "mae_cv_xgb": float(-search.best_score_),
        "importancia": _importance(winner_model, feature_cols),
        "coef_linea_base_escalada": signed,
        "baseline_train": baseline,
        "modelo_comparacion": winner_model,
        "modelo_desplegado": deployed,
        "test_modelo_desplegado": deployed_test,
        "mediana_entrenamiento": median_value,
    }


def run_classification(
    train,
    val,
    test,
    baseline,
    quantiles: dict,
    random_state: int,
    feature_cols: list[str] | None = None,
    baseline_cols: list[str] | None = None,
) -> dict:
    feature_cols = FEATURES if feature_cols is None else feature_cols
    baseline_cols = BASELINE_FEATURES if baseline_cols is None else baseline_cols
    frames = {}
    for name, frame in {"train": train, "val": val, "test": test}.items():
        prediction = baseline.predict(baseline_matrix(frame, baseline_cols))
        frames[name] = assign_degradation(
            frame,
            prediction,
            quantiles["alerta"],
            quantiles["critico"],
        )

    def xy(frame):
        return frame[feature_cols], frame["clase"]

    sample_weight = compute_sample_weight("balanced", frames["train"]["clase"])
    models = {
        "dummy_frecuente": DummyClassifier(strategy="most_frequent"),
        "arbol": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=4,
            random_state=random_state,
        ),
        "gbm": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.08,
            random_state=random_state,
        ),
        "xgboost": XGBClassifier(
            n_estimators=140,
            max_depth=4,
            learning_rate=0.08,
            objective="multi:softprob",
            tree_method="hist",
            n_jobs=4,
            random_state=random_state,
        ),
        "adaboost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=random_state),
            n_estimators=40,
            learning_rate=0.8,
            random_state=random_state,
        ),
    }
    weighted = {"gbm", "xgboost", "adaboost"}
    class_to_int = {label: index for index, label in enumerate(CLASS_ORDER)}
    int_to_class = {index: label for label, index in class_to_int.items()}
    rows = []
    fitted = {}

    def predict_labels(name: str, model, frame: pd.DataFrame):
        prediction = model.predict(frame[feature_cols])
        if name == "xgboost":
            prediction = [int_to_class[int(value)] for value in prediction]
        return prediction

    for name, estimator in models.items():
        print(f"  clasificación: {name}")
        model = clone(estimator)
        x_train, y_train = xy(frames["train"])
        y_fit = y_train.map(class_to_int).astype(int) if name == "xgboost" else y_train
        if name in weighted:
            model.fit(x_train, y_fit, sample_weight=sample_weight)
        else:
            model.fit(x_train, y_fit)
        fitted[name] = model
        for part_name in ("val", "test"):
            _, y_part = xy(frames[part_name])
            scores = classification_scores(y_part, predict_labels(name, model, frames[part_name]))
            rows.append({"modelo": name, "particion": part_name, **scores})

    table = pd.DataFrame(rows)
    winner = str(
        table[table["particion"] == "val"].sort_values("f1_macro", ascending=False).iloc[0]["modelo"]
    )
    learned = table[table["modelo"] != "dummy_frecuente"]
    learned_name = str(
        learned[learned["particion"] == "val"].sort_values("f1_macro", ascending=False).iloc[0]["modelo"]
    )
    matrix_name = learned_name if winner == "dummy_frecuente" else winner
    test_pred = predict_labels(matrix_name, fitted[matrix_name], frames["test"])
    matrix = confusion_matrix(
        frames["test"]["clase"],
        test_pred,
        labels=CLASS_ORDER,
    )
    counts = frames["train"]["clase"].value_counts().to_dict()
    return {
        "tabla": table,
        "ganador_validacion": winner,
        "modelo_matriz": matrix_name,
        "matriz_test": matrix.tolist(),
        "clases": CLASS_ORDER,
        "conteo_train": {str(key): int(value) for key, value in counts.items()},
        "frames": frames,
    }


def run_forecast(
    operating: pd.DataFrame,
    params: dict,
    random_state: int,
    feature_cols: list[str] | None = None,
    preview_points: int = 96 * 7,
) -> dict:
    feature_cols = FEATURES if feature_cols is None else feature_cols
    forecast_cfg = params["forecast"]
    lags = list(forecast_cfg["lags"])
    horizon = int(forecast_cfg["horizon_steps"])
    enriched = add_forecast_columns(
        operating,
        lags=lags,
        horizon=horizon,
        step_minutes=float(forecast_cfg["step_minutes"]),
    )
    ready = ready_forecast_frame(enriched, lags, feature_cols)
    train, val, test = chronological_split(
        ready,
        params["split"]["train"],
        params["split"]["val"],
    )
    features = forecast_feature_names(lags, feature_cols)
    target = "vacio_futuro"
    rows = []

    def add_constant_model(name: str, column: str):
        for part_name, frame in {"val": val, "test": test}.items():
            rows.append(
                {
                    "modelo": name,
                    "particion": part_name,
                    **regression_scores(frame[target], frame[column]),
                }
            )

    add_constant_model("persistencia", "vacio_ahora")
    if "vacio_lag_96" in ready.columns:
        add_constant_model("persistencia_diaria", "vacio_lag_96")

    catalog = {
        "ridge_lags": _ridge(),
        "random_forest_lags": RandomForestRegressor(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=8,
            n_jobs=4,
            random_state=random_state,
        ),
        "mlp_ventanas": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        max_iter=80,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    fitted = {}
    for name, estimator in catalog.items():
        print(f"  pronóstico: {name}")
        model = clone(estimator)
        model.fit(train[features], train[target])
        fitted[name] = model
        for part_name, frame in {"val": val, "test": test}.items():
            rows.append(
                {
                    "modelo": name,
                    "particion": part_name,
                    **regression_scores(frame[target], model.predict(frame[features])),
                }
            )

    table = pd.DataFrame(rows)
    winner = str(table[table["particion"] == "val"].sort_values("mae").iloc[0]["modelo"])
    if winner in fitted:
        test_prediction = fitted[winner].predict(test[features])
    elif winner == "persistencia":
        test_prediction = test["vacio_ahora"].to_numpy()
    else:
        test_prediction = test["vacio_lag_96"].to_numpy()

    preview = test[["Time", target]].copy()
    preview["prediccion"] = test_prediction
    preview = preview.tail(preview_points)
    adf_stat, adf_p, *_ = adfuller(
        train["vacio_ahora"].dropna(),
        autolag="AIC",
        result_object=False,
    )
    return {
        "tabla": table,
        "ganador_validacion": winner,
        "filas": int(len(ready)),
        "horizonte_min": horizon * int(forecast_cfg["step_minutes"]),
        "adf_estadistico": float(adf_stat),
        "adf_p_valor": float(adf_p),
        "preview_test": preview,
    }


def run_clusters(
    train: pd.DataFrame,
    random_state: int,
    feature_cols: list[str] | None = None,
) -> dict:
    feature_cols = FEATURES if feature_cols is None else feature_cols
    scaler = StandardScaler()
    matrix = scaler.fit_transform(train[feature_cols])
    sample_size = min(6000, len(matrix))
    sample_index = np.linspace(0, len(matrix) - 1, sample_size, dtype=int)
    sample = matrix[sample_index]
    scores = []
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(sample)
        scores.append({"k": k, "silhouette": float(silhouette_score(sample, labels))})
    best_k = max(scores, key=lambda item: item["silhouette"])["k"]
    model = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    labels = model.fit_predict(matrix)
    profile = train[feature_cols + [TARGET_ALIAS]].copy()
    profile["cluster"] = labels
    medians = profile.groupby("cluster").median(numeric_only=True).reset_index()
    counts = profile["cluster"].value_counts().sort_index()
    medians["n"] = medians["cluster"].map(counts).astype(int)
    return {
        "silhouette": scores,
        "k": int(best_k),
        "perfil": medians.to_dict(orient="records"),
    }


def run_hypotheses(
    test_frame: pd.DataFrame,
    hot_col: str = "temp_condensador",
    cold_col: str = "temp_agua_mar",
    exhaust_col: str = EXHAUST_TEMP_ALIAS,
    contrast_cols: list[str] | None = None,
) -> dict:
    normal = test_frame["clase"] == "normal"
    critico = test_frame["clase"] == "critico"
    test_frame = test_frame.copy()
    test_frame["ttd"] = terminal_difference(test_frame, hot_col=hot_col, cold_col=cold_col)
    results = {
        "ttd": _mann_whitney(test_frame.loc[normal, "ttd"], test_frame.loc[critico, "ttd"]),
    }
    if exhaust_col in test_frame.columns:
        results[exhaust_col] = _mann_whitney(
            test_frame.loc[normal, exhaust_col],
            test_frame.loc[critico, exhaust_col],
        )
    for column in contrast_cols or ["presion_eyectores", "temp_agua_mar"]:
        if column in test_frame.columns:
            results[column] = _mann_whitney(
                test_frame.loc[normal, column],
                test_frame.loc[critico, column],
            )
    return results


def run_training(params: dict) -> dict:
    root = Path(params["_root"])
    random_state = int(params["models"]["random_state"])
    print("Cargando historiador...")
    frame = load_historian(
        params["data"]["excel_path"],
        params["data"]["sheet"],
        params["data"]["time_col"],
        params["data"]["target"],
    )
    operating = operating_frame(
        frame,
        min_tv_mw=float(params["filters"]["min_tv_mw"]),
        min_vacuum=float(params["filters"]["min_vacuum"]),
        max_vacuum=float(params["filters"]["max_vacuum"]),
    )
    train, val, test = chronological_split(
        operating,
        float(params["split"]["train"]),
        float(params["split"]["val"]),
    )
    print(f"Régimen operativo: {len(operating):,} filas. Train {len(train):,} / val {len(val):,} / test {len(test):,}")

    print("Modelos de vacío esperado...")
    regression = run_regression(train, val, test, random_state)
    baseline = regression["baseline_train"]
    train_residual = train[TARGET_ALIAS] - baseline.predict(baseline_matrix(train))
    quantiles = {
        "alerta": float(np.quantile(train_residual, params["anomaly"]["alerta_quantile"])),
        "critico": float(np.quantile(train_residual, params["anomaly"]["critico_quantile"])),
    }

    print("Clasificación de degradación...")
    classification = run_classification(train, val, test, baseline, quantiles, random_state)
    print("Pronóstico a 1 hora...")
    forecast = run_forecast(operating, params, random_state)
    print("Clusters, pruebas y drift...")
    clusters = run_clusters(train, random_state)
    hypotheses = run_hypotheses(classification["frames"]["test"])
    psi = {
        feature: {
            "psi": population_stability_index(
                train[feature].to_numpy(),
                test[feature].to_numpy(),
                bins=int(params["monitoring"]["psi_bins"]),
            ),
        }
        for feature in FEATURES + [TARGET_ALIAS]
    }
    for item in psi.values():
        item["banda"] = psi_band(item["psi"])

    def slice_profile(frame: pd.DataFrame) -> dict:
        return {
            "inicio": str(frame["Time"].min()),
            "fin": str(frame["Time"].max()),
            "vacio": float(frame[TARGET_ALIAS].median()),
            "temp_agua_mar": float(frame["temp_agua_mar"].median()),
            "potencia_tv": float(frame["potencia_tv"].median()),
        }

    test_prediction = regression["modelo_comparacion"].predict(test[FEATURES])
    curves = _learning_curves(train, TARGET_ALIAS, random_state)
    vif_rows = _vif(train)

    summary = {
        "filas_limpias": int(len(frame)),
        "filas_operacion": int(len(operating)),
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
        },
        "regresion": regression["tabla"].to_dict(orient="records"),
        "ganador_regresion": regression["ganador_validacion"],
        "modelo_flexible": regression["mejor_modelo_flexible"],
        "flexible_supera_mediana": regression["flexible_supera_mediana"],
        "modelo_servido": (
            regression["mejor_modelo_flexible"]
            if regression["flexible_supera_mediana"]
            else "lineal_agua_y_carga"
        ),
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
    }

    metrics_dir = root / "reports" / "metrics"
    figures_dir = root / "reports" / "figures"
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
    test_frame["ttd"] = terminal_difference(test_frame)
    save_figures(summary, test_frame, forecast["preview_test"], figures_dir)
    write_report(summary, root / "reports" / "INFORME.md")

    if regression["flexible_supera_mediana"]:
        served_pipeline = regression["modelo_desplegado"]
        served_name = regression["mejor_modelo_flexible"]
    else:
        served_pipeline = baseline
        served_name = "lineal_agua_y_carga"
    bundle = {
        "pipeline": served_pipeline,
        "baseline": baseline,
        "model_name": served_name,
        "anomaly_threshold": quantiles["critico"],
        "class_thresholds": quantiles,
    }
    joblib.dump(bundle, model_dir / "vacuum_bundle.joblib", compress=3)
    print(f"Listo. Ganador de validación: {regression['ganador_validacion']}")
    return summary
