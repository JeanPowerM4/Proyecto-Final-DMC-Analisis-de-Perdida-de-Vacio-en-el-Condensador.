"""Punto de entrada: entrenar el estudio o puntuar un CSV."""

import argparse
import json

import pandas as pd

from src.acc_study import run_acc_study
from src.config import ROOT, load_params
from src.data.dictionary import API_FIELDS
from src.models.inference import load_bundle, score_record
from src.reporting.comparison import write_comparison_files
from src.training import run_training


def predict_file(path: str) -> pd.DataFrame:
    bundle = load_bundle()
    frame = pd.read_csv(path)
    missing = [column for column in API_FIELDS if column not in frame.columns]
    if missing:
        raise SystemExit(f"Al CSV le faltan columnas: {missing}")
    return pd.DataFrame([score_record(bundle, row.to_dict()) for _, row in frame.iterrows()])


def main():
    parser = argparse.ArgumentParser(description="Vacío del condensador de superficie")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train")
    sub.add_parser("train-aero")
    sub.add_parser("compare")
    predict = sub.add_parser("predict")
    predict.add_argument("--data", required=True)
    predict.add_argument("--output", default="reports/predicciones.csv")
    args = parser.parse_args()

    if args.command == "train":
        run_training(load_params())
        return
    if args.command == "train-aero":
        run_acc_study(load_params())
        write_comparison_files(ROOT)
        return
    if args.command == "compare":
        path = write_comparison_files(ROOT)
        print(path)
        return

    result = predict_file(args.data)
    result.to_csv(args.output, index=False)
    print(json.dumps(result.to_dict(orient="records"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
