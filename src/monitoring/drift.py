"""Índice de estabilidad poblacional, como control de data drift."""

import numpy as np


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < bins or len(current) < bins:
        return float("nan")

    cuts = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return float("nan")

    ref_counts, _ = np.histogram(reference, bins=cuts)
    cur_counts, _ = np.histogram(current, bins=cuts)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-4, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-4, None)
    ref_pct = ref_pct / ref_pct.sum()
    cur_pct = cur_pct / cur_pct.sum()
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_band(value: float) -> str:
    if not np.isfinite(value):
        return "sin_dato"
    if value < 0.10:
        return "estable"
    if value < 0.25:
        return "cambio_moderado"
    return "drift"
