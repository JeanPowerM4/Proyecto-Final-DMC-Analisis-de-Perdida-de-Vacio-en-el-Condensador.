"""Diccionario operativo del condensador de superficie.

Los alias cortos son los nombres que ven los modelos. Los tags PI se conservan
para trazar cada variable hasta el historiador.
"""

TAG_DICTIONARY = {
    "FNX:00TI7004.PV": "Temperatura de agua de mar",
    "FNX:10TI3005.PV": "Temperatura de condensador",
    "FNX:10PI3001.PV": "Vacío del condensador (tag nominal, sin datos)",
    "FNX:S1_EV_P.PV": "Presión de escape de TV10 (proxy de vacío)",
    "FNX:10PIC2333.PV": "Presión de vapor de eyectores",
    "FNX:S1.TT_EXH.PV": "Temperatura de descarga de turbina LP (sin datos)",
    "FNX:G1_DWATT.PV": "Potencia TG11",
    "FNX:G2_DWATT.PV": "Potencia TG12",
    "FNX:S1_DWATT.PV": "Potencia TV10",
    "FNX:12PI1801A.PV": "Presión del domo HP del HRSG 12",
    "FNX:12TI1245B.PV": "Temperatura de entrada al domo HP",
    "FNX:12PI1701A.PV": "Presión del domo IP del HRSG 12",
    "FNX:12PI1601A.PV": "Presión del domo LP del HRSG 12",
    "FNX:11PI1601A.PV": "Presión del domo LP del HRSG 11",
    "FNX:12PI2620.PV": "Presión de vapor LP",
    "FNX:12TI1305B.PV": "Temperatura de salida del sobrecalentador LP",
    "FNX:12FI1630.PV": "Flujo de agua de alimentación al domo LP",
    "FNX:12FT1630.PV": "Flujo de agua de alimentación LP",
    "FNX:12TI1682B.PV": "Temperatura de agua de alimentación al domo LP",
    "FNX:12TI1634B.PV": "Temperatura de agua de alimentación LP",
    "FNX:12PI1615.PV": "Presión de agua de alimentación LP",
    "FNX:S1_IP_P.PV": "Presión de admisión IP",
    "FNX:S1_TT_IS.PV": "Temperatura de vapor de admisión",
    "FNX:S1_TT_ES.PV": "Temperatura de escape HP",
    "FNX:S1_HRHP_P.PV": "Presión de recalentamiento",
    "FNX:S1_TT_RHS.PV": "Temperatura de vapor recalentado",
    "FNX:S1_AP_P.PV": "Presión de admisión",
    "FNX:S1_TT_EXH.PV": "Temperatura de vapor de escape LP",
    "FNX:10TI1919.PV": "Temperatura del header LP",
    "FNX:12PI2441.PV": "Presión de vapor frío de recalentamiento",
}

# Causas operativas del vacío. No incluyen la temperatura del condensador ni
# la temperatura de escape: ambas acompañan la presión de saturación y
# filtrarían el mismo fenómeno que se quiere explicar.
FEATURE_TAGS = [
    "FNX:00TI7004.PV",
    "FNX:S1_DWATT.PV",
    "FNX:G1_DWATT.PV",
    "FNX:G2_DWATT.PV",
    "FNX:10PIC2333.PV",
    "FNX:12PI2620.PV",
    "FNX:12FT1630.PV",
]

FEATURE_ALIAS = {
    "FNX:00TI7004.PV": "temp_agua_mar",
    "FNX:S1_DWATT.PV": "potencia_tv",
    "FNX:G1_DWATT.PV": "potencia_tg11",
    "FNX:G2_DWATT.PV": "potencia_tg12",
    "FNX:10PIC2333.PV": "presion_eyectores",
    "FNX:12PI2620.PV": "presion_vapor_lp",
    "FNX:12FT1630.PV": "flujo_fw_lp",
}

FEATURES = [FEATURE_ALIAS[tag] for tag in FEATURE_TAGS]

# Línea base física: el agua de mar y la carga fijan el vacío esperado.
# El residual de esta línea es la señal de ensuciamiento, aire o enfriamiento.
BASELINE_FEATURES = ["temp_agua_mar", "potencia_tv"]

TARGET_ALIAS = "vacio"
CONDENSER_TEMP_ALIAS = "temp_condensador"
EXHAUST_TEMP_ALIAS = "temp_escape"

VALIDATOR_TAGS = {
    "FNX:10TI3005.PV": CONDENSER_TEMP_ALIAS,
    "FNX:S1_TT_EXH.PV": EXHAUST_TEMP_ALIAS,
}

API_FIELDS = {
    "temperatura_agua_mar": "temp_agua_mar",
    "potencia_tv_mw": "potencia_tv",
    "potencia_tg11_mw": "potencia_tg11",
    "potencia_tg12_mw": "potencia_tg12",
    "presion_eyectores": "presion_eyectores",
    "presion_vapor_lp": "presion_vapor_lp",
    "flujo_fw_lp": "flujo_fw_lp",
}
