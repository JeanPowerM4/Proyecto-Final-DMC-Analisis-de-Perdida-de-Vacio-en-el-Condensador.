"""Comparación entre condensador de superficie y aerocondensador."""

import json
from pathlib import Path

from src.reporting.report import _fmt, _row

INHG_TO_MBAR = 33.8639


def _best(records: list[dict], partition: str, metric: str = "mae") -> dict:
    rows = [row for row in records if row["particion"] == partition]
    return min(rows, key=lambda row: row[metric])


def _relative(mae: float, level: float) -> float:
    return mae / level if level else float("nan")


def write_comparison(surface: dict, aero: dict, path: Path) -> None:
    surface_level = surface["vacio_operacion"]["mediana"]
    aero_level = aero["vacio_operacion"]["mediana"]
    surface_mbar = surface_level * INHG_TO_MBAR
    aero_mbar = aero_level * 1000

    surface_test_best = _best(surface["regresion"], "test")
    aero_test_best = _best(aero["regresion"], "test")
    surface_base = _row(surface["regresion"], "lineal_agua_y_carga", "test")
    aero_base = _row(aero["regresion"], aero["baseline_name"], "test")
    surface_median = _row(surface["regresion"], "mediana_entrenamiento", "test")
    aero_median = _row(aero["regresion"], "mediana_entrenamiento", "test")
    surface_persist = _row(surface["pronostico"], "persistencia", "test")
    aero_persist = _row(aero["pronostico"], "persistencia", "test")
    surface_forecast = _row(surface["pronostico"], surface["ganador_pronostico"], "test")
    aero_forecast = _row(aero["pronostico"], aero["ganador_pronostico"], "test")

    surface_ttd = surface["hipotesis"].get("ttd", {})
    aero_ttd = aero["hipotesis"].get("ttd", {})
    surface_psi = surface["psi"]["temp_agua_mar"]["psi"]
    aero_psi = aero["psi"]["temp_ambiente"]["psi"]

    text = f"""# Superficie y aerocondensador

Los dos estudios hacen la misma pregunta: con la turbina en servicio, qué presión de condensación cabe esperar a partir del fluido frío y de la carga, y cuándo lo observado es peor que esa referencia. No son el mismo equipo ni el mismo año. La comparación sirve para ver qué se repite de la física y qué cambia con la tecnología.

## Nivel de presión

| | Superficie | Aerocondensador |
| --- | --- | --- |
| Señal | `FNX:S1_EV_P.PV` | `TV_BP_Pout` |
| Lectura de unidad | inHg absolutos, por la magnitud | bar absolutos, porque la temperatura de salida calza con la saturación |
| Mediana en servicio | {_fmt(surface_level, 3)} inHgA, cerca de {_fmt(surface_mbar, 1)} mbar | {_fmt(aero_level, 3)} bar, {_fmt(aero_mbar, 1)} mbar |
| Percentil 95 | {_fmt(surface['vacio_operacion']['p95'], 3)} | {_fmt(aero['vacio_operacion']['p95'], 3)} |
| Periodo | {surface['inicio'][:10]} a {surface['fin'][:10]} | {aero['inicio'][:10]} a {aero['fin'][:10]} |
| Paso | 15 min | 1 min, en bloques con huecos |
| Filas de modelado | {surface['filas_operacion']:,} | {aero['filas_operacion']:,} |
| Carga | desde 50 MW, con paradas fuera del modelo | CF {_fmt(aero['cf']['min'], 3)} a {_fmt(aero['cf']['max'], 3)}, potencia = CF × 300 MW |

En números redondos, el aerocondensador condensa cerca de {_fmt(aero_mbar / surface_mbar, 2)} veces la presión absoluta del condensador de superficie. Es lo esperable: el aire ambiente enfría peor que el agua de mar. La muestra del aerocondensador, además, ya está en base. La de superficie tuvo que recortar el modo fuera de servicio, donde el vacío se iba hacia 30.

## La misma línea física

En superficie la línea es agua de mar + potencia de TV. En el aerocondensador es temperatura ambiente + potencia. En las dos, el coeficiente del fluido frío es positivo: más calor en la fuente fría, más presión de condensación.

| Test | Superficie | Aerocondensador |
| --- | --- | --- |
| MAE de la mediana | {_fmt(surface_median.get('mae'))} | {_fmt(aero_median.get('mae'))} |
| MAE de la línea fría + carga | {_fmt(surface_base.get('mae'))} | {_fmt(aero_base.get('mae'))} |
| MAPE de esa línea | {_fmt(surface_base.get('mape_pct'), 2)} % | {_fmt(aero_base.get('mape_pct'), 2)} % |
| R² de esa línea | {_fmt(surface_base.get('r2'))} | {_fmt(aero_base.get('r2'))} |
| Mejor MAE del test | {surface_test_best['modelo']} {_fmt(surface_test_best['mae'])} | {aero_test_best['modelo']} {_fmt(aero_test_best['mae'])} |
| MAE / mediana | línea {_fmt(_relative(surface_base.get('mae'), surface_level), 3)} | línea {_fmt(_relative(aero_base.get('mae'), aero_level), 3)} |
| Ganador de validación | {surface['ganador_regresion']} | {aero['ganador_regresion']} |
| Modelo que queda de referencia | {surface['modelo_servido']} | {aero['modelo_servido']} |

El MAE no se compara en crudo: las unidades no son las mismas. El MAPE y el MAE dividido por la mediana sí. En el test, la línea física de superficie queda en {_fmt(_relative(surface_base.get('mae'), surface_level) * 100, 1)} % de su mediana y la del aerocondensador en {_fmt(_relative(aero_base.get('mae'), aero_level) * 100, 1)} %.

La diferencia está en qué modelo se sostiene al salir del entrenamiento. En superficie la validación la gana la mediana y ningún ensemble se promueve: el régimen cambia y la referencia que queda es la línea de agua y carga. En el aerocondensador la validación la gana {aero['ganador_regresion']}. La muestra ya viene en base, sin el salto de parada, y la contrapresión sigue al ambiente. La línea ambiente + carga ya deja un R² de test de {_fmt(aero_base.get('r2'))} y un MAPE de {_fmt(aero_base.get('mape_pct'), 2)} %; el modelo elegido en validación llega a MAE {_fmt(aero_test_best['mae'])} en el test.

## Una hora hacia adelante

| Test | Superficie | Aerocondensador |
| --- | --- | --- |
| Horizonte | {surface['horizonte_min']} min | {aero['horizonte_min']} min |
| Ganador de validación | {surface['ganador_pronostico']} | {aero['ganador_pronostico']} |
| MAE del ganador | {_fmt(surface_forecast.get('mae'))} | {_fmt(aero_forecast.get('mae'))} |
| MAE de la persistencia | {_fmt(surface_persist.get('mae'))} | {_fmt(aero_persist.get('mae'))} |
| Persistencia / mediana | {_fmt(_relative(surface_persist.get('mae'), surface_level), 3)} | {_fmt(_relative(aero_persist.get('mae'), aero_level), 3)} |

La presión de condensación cambia lento. A una hora, la persistencia es el baseline que hay que ganarle. En superficie no se le gana. En el aerocondensador el ganador de validación es {aero['ganador_pronostico']}.

El archivo del aerocondensador no tiene un bloque continuo de 24 horas, así que la persistencia diaria que sí se pudo calcular en superficie aquí no existe.

## La alarma

En los dos casos la clase crítica es el residual por encima del percentil 95 de entrenamiento, y la temperatura caliente del condensador no entra al modelo. Sirve de contraste.

| | Superficie | Aerocondensador |
| --- | --- | --- |
| Diferencia caliente − frío, normal | {_fmt(surface_ttd.get('mediana_normal'), 2)} | {_fmt(aero_ttd.get('mediana_normal'), 2)} |
| Diferencia caliente − frío, crítico | {_fmt(surface_ttd.get('mediana_critico'), 2)} | {_fmt(aero_ttd.get('mediana_critico'), 2)} |
| p-valor | {_fmt(surface_ttd.get('p_valor'), 4)} | {_fmt(aero_ttd.get('p_valor'), 4)} |

Si el grupo crítico queda más caliente, el residual no es solo un recorte estadístico: el condensador está rechazando peor el calor.

## Qué cambia con la tecnología

El condensador de superficie se explica sobre todo por el agua de mar y la carga. Los eyectores y el vapor de baja presión agregan contexto de planta, con redundancia alta entre potencias y flujos.

El aerocondensador agrega viento. La velocidad entra al modelo y la dirección queda como seno y coseno, porque un viento que recircula aire caliente no es lo mismo que uno que barre el haz. En el test, el grupo crítico tiene viento mediano {_fmt(aero['hipotesis'].get('viento', {}).get('mediana_critico'), 1)} frente a {_fmt(aero['hipotesis'].get('viento', {}).get('mediana_normal'), 1)} en el normal, mientras la temperatura ambiente casi no se mueve ({_fmt(aero['hipotesis'].get('temp_ambiente', {}).get('mediana_normal'), 1)} contra {_fmt(aero['hipotesis'].get('temp_ambiente', {}).get('mediana_critico'), 1)}). El residual se alinea con viento fuerte.

El drift también se repite. PSI del agua de mar entre entrenamiento y test: {_fmt(surface_psi, 2)}. PSI de la temperatura ambiente: {_fmt(aero_psi, 2)}. Un modelo ajustado al primer tramo no se promueve solo porque el error de entrenamiento se vea bien.

## Lectura para el proyecto

Las dos tecnologías sostienen la misma cadena: fluido frío y carga fijan la presión esperada; el residual marca la pérdida de desempeño; a una hora manda la inercia del proceso. El aerocondensador opera a mayor presión absoluta y su muestra ya viene en base, así que el problema no es detectar la parada. Es ver la degradación dentro de una banda estrecha, con el ambiente y el viento como condiciones de frontera.

El dato que falta en superficie es el flujo de agua de mar. En el aerocondensador falta el estado de los ventiladores. Sin eso, la clase crítica sigue siendo una desviación respecto de la línea física, no una causa cerrada.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_comparison_files(root: Path) -> Path:
    surface = json.loads((root / "reports" / "metrics" / "summary.json").read_text(encoding="utf-8"))
    aero = json.loads((root / "reports" / "aero" / "metrics" / "summary.json").read_text(encoding="utf-8"))
    path = root / "reports" / "COMPARACION.md"
    write_comparison(surface, aero, path)
    return path
