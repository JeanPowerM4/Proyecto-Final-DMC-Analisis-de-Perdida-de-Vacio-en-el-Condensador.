from pydantic import BaseModel, Field


class CondenserSnapshot(BaseModel):
    temperatura_agua_mar: float = Field(ge=0, le=120)
    potencia_tv_mw: float = Field(ge=0, le=400)
    potencia_tg11_mw: float = Field(ge=0, le=400)
    potencia_tg12_mw: float = Field(ge=0, le=400)
    presion_eyectores: float = Field(ge=0, le=300)
    presion_vapor_lp: float = Field(ge=0, le=200)
    flujo_fw_lp: float = Field(ge=0, le=2000)
    vacio_observado: float | None = Field(default=None, ge=0, le=35)


class ScoreResponse(BaseModel):
    vacio_esperado: float
    vacio_linea_base_agua_carga: float
    modelo: str
    vacio_observado: float | None = None
    residual_modelo: float | None = None
    residual_linea_base: float | None = None
    clase: str | None = None
    anomalia: bool | None = None
    umbral_anomalia: float | None = None
