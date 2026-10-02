"""Carga y limpieza del extracto PI del condensador de superficie."""

from pathlib import Path

import pandas as pd

from src.data.dictionary import FEATURE_ALIAS, FEATURE_TAGS, TARGET_ALIAS, VALIDATOR_TAGS

EXCEL_ORIGIN = "1899-12-30"
NON_NUMERIC_MARKERS = {"configure", "bad", "tag not found", "nan", "none", ""}


def parse_time(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return pd.to_datetime(numeric, unit="D", origin=EXCEL_ORIGIN)
    return pd.to_datetime(series, errors="coerce")


def to_numeric_tag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.mask(cleaned.str.lower().isin(NON_NUMERIC_MARKERS))
    return pd.to_numeric(cleaned, errors="coerce")


def clean_frame(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """Convierte tags a número, descarta textos de adquisición y ordena el tiempo."""
    frame = df.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if time_col not in frame.columns:
        raise KeyError(f"No está la columna de tiempo {time_col}")

    frame[time_col] = parse_time(frame[time_col])
    for column in frame.columns:
        if column == time_col:
            continue
        frame[column] = to_numeric_tag(frame[column])

    duplicate_names = [column for column in frame.columns if column.endswith(".1")]
    frame = frame.drop(columns=duplicate_names)

    empty = [
        column
        for column in frame.columns
        if column != time_col and frame[column].notna().sum() == 0
    ]
    frame = frame.drop(columns=empty)
    frame = frame.dropna(subset=[time_col]).sort_values(time_col)
    frame = frame.drop_duplicates(subset=[time_col], keep="first")
    return frame.reset_index(drop=True)


def modeling_frame(df: pd.DataFrame, time_col: str, target: str) -> pd.DataFrame:
    """Deja solo el tiempo, el vacío, los drivers y las variables de contraste."""
    required = [time_col, target, *FEATURE_TAGS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas para el modelo: {missing}")

    keep = required + [tag for tag in VALIDATOR_TAGS if tag in df.columns]
    frame = df[keep].rename(
        columns={
            target: TARGET_ALIAS,
            **FEATURE_ALIAS,
            **{tag: alias for tag, alias in VALIDATOR_TAGS.items() if tag in df.columns},
        }
    )
    return frame


def load_historian(excel_path: str, sheet: str, time_col: str, target: str) -> pd.DataFrame:
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No está el Excel del historiador en {path}. "
            "Colócalo en 00_Final_Project/Data_Condensador.xlsx."
        )

    cache = Path(__file__).resolve().parents[2] / "data" / "interim" / "condensador_limpio.parquet"
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        cached = pd.read_parquet(cache)
        cached["Time"] = pd.to_datetime(cached["Time"])
        return cached

    raw = pd.read_excel(path, sheet_name=sheet)
    cleaned = clean_frame(raw, time_col=time_col)
    frame = modeling_frame(cleaned, time_col=time_col, target=target)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    frame["Time"] = pd.to_datetime(frame["Time"])
    return frame


def quality_report(raw_rows: int, frame: pd.DataFrame, time_col: str) -> dict:
    step = frame[time_col].diff().dt.total_seconds().div(60)
    return {
        "filas_crudas": int(raw_rows),
        "filas_limpias": int(len(frame)),
        "inicio": str(frame[time_col].min()),
        "fin": str(frame[time_col].max()),
        "paso_mediano_min": float(step.median()) if step.notna().any() else None,
        "vacio_nulos": int(frame["vacio"].isna().sum()),
    }
