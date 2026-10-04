"""Informe del aerocondensador."""

from pathlib import Path

import pandas as pd

from src.reporting.report import _fmt, _markdown, _row, _table


def write_acc_report(summary: dict, path: Path) -> None:
    winner = summary["ganador_regresion"]
    flexible = summary["modelo_flexible"]
    baseline = summary["baseline_name"]
    base_test = _row(summary["regresion"], baseline, "test")
    median_test = _row(summary["regresion"], "mediana_entrenamiento", "test")
    flexible_test = _row(summary["regresion"], flexible, "test")
    flexible_val = _row(summary["regresion"], flexible, "val")
    winner_val = _row(summary["regresion"], winner, "val")
    forecast_winner = summary["ganador_pronostico"]
    forecast_test = _row(summary["pronostico"], forecast_winner, "test")
    persist_test = _row(summary["pronostico"], "persistencia", "test")
    class_winner = summary["ganador_clasificacion"]
    class_test = _row(summary["clasificacion"], class_winner, "test")
    adf_p = summary["adf_p_valor"]
    stationary = (
        "se rechaza la raíz unitaria al 5%"
        if adf_p < 0.05
        else "no se rechaza la raíz unitaria al 5%"
    )
    vif_lines = "\n".join(
        f"| {row['variable']} | {_fmt(row['vif'], 2)} |" for row in summary["vif"]
    )
    psi_lines = "\n".join(
        f"| {name} | {_fmt(item['psi'], 3)} | {item['banda']} |"
        for name, item in summary["psi"].items()
    )
    hypothesis_lines = []
    for name, item in summary["hipotesis"].items():
        if not item or item.get("p_valor") is None and "mediana_normal" not in item:
            continue
        hypothesis_lines.append(
            f"- **{name}**: mediana normal {_fmt(item.get('mediana_normal'), 3)}, "
            f"mediana crítico {_fmt(item.get('mediana_critico'), 3)}, "
            f"p-valor {_fmt(item.get('p_valor'), 4)}."
        )
    direction = _markdown(pd.DataFrame(summary["vacio_por_direccion"]).round(4))
    clusters = _markdown(pd.DataFrame(summary["clusters"]["perfil"]).round(3))
    coef = summary["coef_linea_base_escalada"]
    service = (
        "Ningún modelo flexible supera a la mediana en validación. "
        f"La referencia que se guarda es la línea {baseline}."
        if not summary["flexible_supera_mediana"]
        else f"El modelo flexible supera a la mediana y queda como referencia: {summary['modelo_servido']}."
    )

    text = f"""# Contrapresión del aerocondensador

Misma pregunta que en el condensador de superficie: con la unidad en servicio, qué presión de escape cabe esperar y cuándo el valor observado es peor que esa referencia.

La señal objetivo es `TV_BP_Pout`, la contrapresión de baja presión. Su magnitud, junto con `TV_BP_Tout`, es consistente con bar absolutos: alrededor de 0.10 bar el agua satura cerca de 46 °C, que es la zona de la temperatura de salida. La potencia no viene en MW. El archivo trae el factor de capacidad `CF`; la potencia usada aquí es CF × {summary['potencia_nominal_mw']:.0f} MW.

## Datos

El archivo tiene {summary['filas_archivo']:,} filas y {summary['filas_limpias']:,} quedan con fecha, contrapresión y drivers completos, del {summary['inicio']} al {summary['fin']}. El muestreo válido es de 1 minuto, en bloques separados por huecos. Ningún bloque continuo llega a 24 horas, así que no hay persistencia diaria identificable.

El factor de capacidad se mueve entre {_fmt(summary['cf']['min'], 3)} y {_fmt(summary['cf']['max'], 3)}, con mediana {_fmt(summary['cf']['mediana'], 3)}. No aparece el modo fuera de servicio que en el condensador de superficie empujaba el vacío hacia 30. Toda la muestra ya es carga alta.

La contrapresión tiene mediana {_fmt(summary['vacio_operacion']['mediana'], 3)} y percentil 95 {_fmt(summary['vacio_operacion']['p95'], 3)}. Partición cronológica: {summary['train']:,} entrenamiento, {summary['val']:,} validación y {summary['test']:,} prueba.

## Qué entra y qué no

Entran temperatura ambiente, potencia, velocidad del viento, flujo de vapor de baja presión y la dirección del viento en seno y coseno. No entra `TV_BP_Tout`: acompaña la temperatura de saturación de la propia contrapresión, igual que la temperatura del condensador en el caso de superficie.

`CF` y la temperatura ambiente se mueven juntos en sentido contrario. Una correlación negativa de la carga con la contrapresión, mirándola sola, no quiere decir que generar más mejore el vacío: en este archivo la carga baja cuando el ambiente sube. El coeficiente parcial de la línea base es el que separa esos dos efectos.

Coeficientes escalados de la línea ambiente + carga: ambiente {_fmt(coef.get('temp_ambiente'))}, potencia {_fmt(coef.get('potencia_mw'))}. El ambiente empuja la contrapresión hacia arriba. El de la potencia queda levemente negativo dentro de una banda de factor de capacidad muy estrecha; no se lee como regla de diseño. El nivel lo pone el ambiente.

## Contrapresión esperada

### Validación

{_table(summary['regresion'], 'val', 'mae', True)}

### Test

{_table(summary['regresion'], 'test', 'mae', True)}

En validación el menor MAE es **{winner}** ({_fmt(winner_val.get('mae'))}). El mejor flexible es **{flexible}** (validación {_fmt(flexible_val.get('mae'))}, test {_fmt(flexible_test.get('mae'))}). En test la mediana queda en {_fmt(median_test.get('mae'))} y la línea ambiente/carga en {_fmt(base_test.get('mae'))}.

{service}

VIF:

| Variable | VIF |
| --- | --- |
{vif_lines}

## Degradación

Las clases salen del residual de la línea ambiente/carga, con los mismos percentiles 75 y 95 del entrenamiento. Conteo de entrenamiento: {summary['conteo_train']}.

### Validación

{_table(summary['clasificacion'], 'val', 'f1_macro', False)}

### Test

{_table(summary['clasificacion'], 'test', 'f1_macro', False)}

Gana la validación **{class_winner}**. En test su F1 macro es {_fmt(class_test.get('f1_macro'))}. La matriz corresponde a **{summary['modelo_matriz']}**.

## Contraste del residual

{chr(10).join(hypothesis_lines)}

El viento separa el grupo crítico con más claridad que la temperatura ambiente. La alarma no es simplemente un día más caluroso: es contrapresión peor de lo que el ambiente y la carga ya explican, y en el test eso coincide con viento más fuerte.

## Pronóstico a {summary['horizonte_min']} minutos

Lags de 15, 30, 60 y 120 minutos, solo si el reloj confirma esa distancia. Filas con ventana válida: {summary['pronostico_filas']:,}. ADF: estadístico {_fmt(summary['adf_estadistico'], 3)}, p-valor {_fmt(adf_p, 4)}; {stationary}.

### Validación

{_table(summary['pronostico'], 'val', 'mae', True)}

### Test

{_table(summary['pronostico'], 'test', 'mae', True)}

Gana la validación **{forecast_winner}**, con MAE de test {_fmt(forecast_test.get('mae'))}. La persistencia queda en {_fmt(persist_test.get('mae'))}.

## Dirección del viento

Mediana de contrapresión por dirección, en toda la muestra válida:

{direction}

## Regímenes

k elegido: **{summary['clusters']['k']}**.

{clusters}

## Estabilidad

| Variable | PSI | Banda |
| --- | --- | --- |
{psi_lines}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
