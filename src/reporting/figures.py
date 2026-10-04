"""Figuras del informe. Se guardan en reports/figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.dictionary import FEATURES, TARGET_ALIAS


def _bar(table: pd.DataFrame, metric: str, title: str, path: Path, higher_is_better: bool = False):
    pivot = table.pivot(index="modelo", columns="particion", values=metric)
    order = pivot["val"].sort_values(ascending=not higher_is_better).index
    pivot = pivot.loc[order]
    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=30)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.legend(title="partición")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_figures(summary: dict, test_frame: pd.DataFrame, forecast_preview: pd.DataFrame, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    regression = pd.DataFrame(summary["regresion"])
    classification = pd.DataFrame(summary["clasificacion"])
    forecast = pd.DataFrame(summary["pronostico"])

    _bar(
        regression,
        "mae",
        "Vacío esperado: MAE en validación y test",
        directory / "regresion_mae.png",
    )
    _bar(
        classification,
        "f1_macro",
        "Degradación del vacío: F1 macro",
        directory / "clasificacion_f1.png",
        higher_is_better=True,
    )
    _bar(
        forecast,
        "mae",
        f"Pronóstico a {summary['horizonte_min']} min: MAE",
        directory / "pronostico_mae.png",
    )

    matrix = np.asarray(summary["matriz_test"], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=summary["clases"],
        yticklabels=summary["clases"],
        ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase por residual de la línea base")
    ax.set_title(f"Matriz de confusión en test ({summary.get('modelo_matriz', summary['ganador_clasificacion'])})")
    plt.tight_layout()
    plt.savefig(directory / "matriz_confusion.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(test_frame["residual_base"], bins=40, color="#4c78a8", edgecolor="white")
    threshold = summary["umbral_residual"]["critico"]
    ax.axvline(threshold, color="#e45756", linestyle="--", label=f"umbral crítico {threshold:.3f}")
    ax.set_title(summary.get("residual_title", "Residual de vacío en test (observado − línea agua/carga)"))
    ax.set_xlabel("Residual")
    ax.legend()
    plt.tight_layout()
    plt.savefig(directory / "residual_test.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    target = summary.get("target", TARGET_ALIAS)
    ax.scatter(test_frame[target], test_frame["prediccion"], s=6, alpha=0.25, color="#4c78a8")
    limits = [
        min(test_frame[target].min(), test_frame["prediccion"].min()),
        max(test_frame[target].max(), test_frame["prediccion"].max()),
    ]
    ax.plot(limits, limits, color="#e45756", linewidth=1)
    ax.set_xlabel("Vacío observado")
    ax.set_ylabel("Vacío predicho")
    ax.set_title(f"Test del mejor modelo flexible ({summary.get('modelo_flexible', summary['ganador_regresion'])})")
    plt.tight_layout()
    plt.savefig(directory / "observado_vs_predicho.png", dpi=120)
    plt.close()

    importance = pd.DataFrame(summary["importancia"])
    if not importance.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=importance, y="variable", x="importancia", color="#72b7b2", ax=ax)
        ax.set_title(f"Importancia de {summary.get('modelo_flexible', 'el modelo flexible')}")
        plt.tight_layout()
        plt.savefig(directory / "importancia.png", dpi=120)
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, curve in summary["curvas"].items():
        ax.plot(curve["muestras"], curve["mae_val"], marker="o", label=f"{name} val")
        ax.plot(curve["muestras"], curve["mae_train"], marker=".", linestyle="--", label=f"{name} train")
    ax.set_xlabel("Muestras de entrenamiento")
    ax.set_ylabel("MAE")
    ax.set_title("Curvas de aprendizaje (partición temporal)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(directory / "curvas_aprendizaje.png", dpi=120)
    plt.close()

    preview = forecast_preview.copy()
    preview["Time"] = pd.to_datetime(preview["Time"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(preview["Time"], preview["vacio_futuro"], label="observado", linewidth=1.2)
    ax.plot(preview["Time"], preview["prediccion"], label="pronóstico", linewidth=1.0, alpha=0.85)
    ax.set_title(f"Últimos días de test: {summary['ganador_pronostico']}")
    ax.set_ylabel("Vacío")
    ax.legend()
    plt.tight_layout()
    plt.savefig(directory / "pronostico_reciente.png", dpi=120)
    plt.close()

    features = summary.get("features") or FEATURES
    target = summary.get("target", TARGET_ALIAS)
    corr = test_frame[features + [target]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Spearman en el bloque de test operativo")
    plt.tight_layout()
    plt.savefig(directory / "spearman_test.png", dpi=120)
    plt.close()

    profile = pd.DataFrame(summary["clusters"]["perfil"])
    if not profile.empty:
        melted = profile.melt(id_vars=["cluster", "n"], var_name="variable", value_name="mediana")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        shown = summary.get("cluster_vars") or ["temp_agua_mar", "potencia_tv", "vacio"]
        sns.barplot(data=melted[melted["variable"].isin(shown)],
                    x="variable", y="mediana", hue="cluster", ax=ax)
        ax.set_title(f"Medianas por cluster (k={summary['clusters']['k']})")
        plt.tight_layout()
        plt.savefig(directory / "clusters.png", dpi=120)
        plt.close()
