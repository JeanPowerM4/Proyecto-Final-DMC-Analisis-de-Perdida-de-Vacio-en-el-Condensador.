from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params(path: Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "params.yaml"
    with path.open(encoding="utf-8") as file:
        params = yaml.safe_load(file)
    excel = Path(params["data"]["excel_path"])
    if not excel.is_absolute():
        excel = (ROOT / excel).resolve()
    params["data"]["excel_path"] = str(excel)
    params["_root"] = str(ROOT)
    return params
