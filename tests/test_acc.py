import numpy as np
import pandas as pd

from src.data.acc import prepare_acc


def _row(**overrides):
    base = {
        "date": "2012-08-15 08:31",
        "CF": 0.98,
        "BP_Flow": 35.0,
        "Wind_speed": 6.4,
        "Wind_dir": "E",
        "T_ambient": 17.0,
        "TV_BP_Pout": 0.093,
        "TV_BP_Tout": 44.2,
    }
    base.update(overrides)
    return base


def test_potencia_es_cf_por_300_mw():
    prepared = prepare_acc(pd.DataFrame([_row(CF=0.5)]), rated_mw=300)
    assert prepared["potencia_mw"].iloc[0] == 150
    assert prepared["vacio"].iloc[0] == 0.093


def test_viento_del_este_queda_en_seno_uno():
    prepared = prepare_acc(pd.DataFrame([_row(Wind_dir="E")]))
    assert abs(prepared["viento_sin"].iloc[0] - 1) < 1e-9
    assert abs(prepared["viento_cos"].iloc[0]) < 1e-9


def test_direccion_sin_dato_no_entra():
    prepared = prepare_acc(pd.DataFrame([_row(Wind_dir="---")]))
    assert prepared.empty


def test_temperatura_de_salida_no_es_feature():
    prepared = prepare_acc(pd.DataFrame([_row()]))
    from src.data.acc import ACC_FEATURES

    assert "temp_salida_bp" not in ACC_FEATURES
    assert np.isfinite(prepared["temp_salida_bp"].iloc[0])
