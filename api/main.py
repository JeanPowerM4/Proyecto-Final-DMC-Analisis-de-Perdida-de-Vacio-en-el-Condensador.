from io import StringIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse

from api.dependencies import get_bundle
from api.schemas import CondenserSnapshot, ScoreResponse
from src.data.dictionary import API_FIELDS
from src.models.inference import score_record

app = FastAPI(
    title="Vacío del condensador de superficie",
    description=(
        "Vacío esperado, residual frente a la línea agua de mar/carga "
        "y clase de degradación para el condensador de superficie."
    ),
    version="1.0.0",
)

_calls = {"predicciones": 0, "anomalias": 0}


@app.get("/")
def root():
    return {"service": "condensador-superficie", "version": "1.0.0"}


@app.get("/health")
def health():
    bundle = get_bundle()
    return {
        "status": "healthy",
        "modelo": bundle["model_name"],
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return (
        f"condenser_predictions_total {_calls['predicciones']}\n"
        f"condenser_anomalies_total {_calls['anomalias']}\n"
    )


@app.post("/predict", response_model=ScoreResponse)
def predict(snapshot: CondenserSnapshot):
    try:
        result = score_record(get_bundle(), snapshot.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    _calls["predicciones"] += 1
    if result.get("anomalia"):
        _calls["anomalias"] += 1
    return result


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe ser CSV.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="El CSV está vacío.")
    try:
        frame = pd.read_csv(StringIO(content.decode("utf-8")))
        if frame.empty:
            raise HTTPException(status_code=400, detail="El CSV está vacío.")
        missing = [column for column in API_FIELDS if column not in frame.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")
        scored = pd.DataFrame(
            [score_record(get_bundle(), row.to_dict()) for _, row in frame.iterrows()]
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _calls["predicciones"] += len(scored)
    if "anomalia" in scored.columns:
        _calls["anomalias"] += int(scored["anomalia"].fillna(False).sum())
    output = StringIO()
    scored.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predicciones_vacio.csv"'},
    )
