"""Informe en español a partir de las métricas ya calculadas."""

from pathlib import Path

import pandas as pd


def _markdown(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(map(str, shown.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in shown.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def _table(records: list[dict], partition: str, sort_by: str, ascending: bool) -> str:
    frame = pd.DataFrame(records)
    frame = frame[frame["particion"] == partition].sort_values(sort_by, ascending=ascending)
    columns = [column for column in frame.columns if column != "particion"]
    return _markdown(frame[columns])


def _fmt(value, digits=4) -> str:
    if value is None:
        return "s/d"
    return f"{float(value):.{digits}f}"


def _row(records: list[dict], model: str, partition: str) -> dict:
    for record in records:
        if record["modelo"] == model and record["particion"] == partition:
            return record
    return {}


def write_report(summary: dict, path: Path) -> None:
    winner = summary["ganador_regresion"]
    winner_val = _row(summary["regresion"], winner, "val")
    winner_test = _row(summary["regresion"], winner, "test")
    flexible_name = summary.get("modelo_flexible", winner)
    flexible_val = _row(summary["regresion"], flexible_name, "val")
    flexible_test = _row(summary["regresion"], flexible_name, "test")
    class_val = _row(summary["clasificacion"], summary["ganador_clasificacion"], "val")
    base_test = _row(summary["regresion"], "lineal_agua_y_carga", "test")
    median_test = _row(summary["regresion"], "mediana_entrenamiento", "test")
    class_winner = summary["ganador_clasificacion"]
    class_test = _row(summary["clasificacion"], class_winner, "test")
    dummy_test = _row(summary["clasificacion"], "dummy_frecuente", "test")
    forecast_winner = summary["ganador_pronostico"]
    forecast_test = _row(summary["pronostico"], forecast_winner, "test")
    persist_test = _row(summary["pronostico"], "persistencia", "test")
    deployed = summary["test_modelo_desplegado"]
    adf_p = summary["adf_p_valor"]
    stationary = "se rechaza la raíz unitaria al 5%" if adf_p < 0.05 else "no se rechaza la raíz unitaria al 5%"

    vif_lines = "\n".join(
        f"| {row['variable']} | {_fmt(row['vif'], 2)} |" for row in summary["vif"]
    )
    psi_lines = "\n".join(
        f"| {name} | {_fmt(item['psi'], 3)} | {item['banda']} |"
        for name, item in summary["psi"].items()
    )
    cluster_frame = pd.DataFrame(summary["clusters"]["perfil"])
    silhouette = ", ".join(
        f"k={item['k']}: {_fmt(item['silhouette'], 3)}" for item in summary["clusters"]["silhouette"]
    )
    def periodo(name: str) -> str:
        item = summary.get("particiones", {}).get(name)
        if not item:
            return name
        return (
            f"{item['inicio'][:10]} a {item['fin'][:10]}, "
            f"vacío mediano {item['vacio']:.3f}, agua de mar {item['temp_agua_mar']:.1f}"
        )

    hypothesis_lines = []
    for name, item in summary["hipotesis"].items():
        if not item:
            continue
        hypothesis_lines.append(
            f"- **{name}**: mediana normal {_fmt(item.get('mediana_normal'), 3)}, "
            f"mediana crítico {_fmt(item.get('mediana_critico'), 3)}, "
            f"p-valor {_fmt(item.get('p_valor'), 4)} "
            f"(n normal {item.get('n_normal')}, n crítico {item.get('n_critico')})."
        )

    text = f"""# Vacío del condensador de superficie

Estudio de Jean Piere Cholán para el proyecto integrador del Diploma Advanced Data Scientist.
El alcance es el **condensador de superficie enfriado por agua de mar**. El aerocondensador queda fuera de este repositorio.

## Pregunta

Con la turbina de vapor en servicio, ¿qué vacío cabe esperar a partir del agua de mar, la carga y el sistema de extracción de aire, y cuándo el vacío observado es peor que esa referencia?

Esa diferencia es la señal de pérdida de desempeño del condensador: ensuciamiento, ingreso de aire o falta de enfriamiento. No es un detector de unidad fuera de servicio.

## Datos

El extracto del historiador cubre del {summary['inicio']} al {summary['fin']}, con paso mediano de 15 minutos. Después de convertir textos de adquisición (`Configure`, `Bad`, `Tag not found`) a faltantes, quedan {summary['filas_limpias']:,} filas. El régimen de modelado exige potencia de TV10 de al menos 50 MW y vacío entre 0.4 y 5. Entran {summary['filas_operacion']:,} filas.

En ese régimen el vacío tiene mediana {_fmt(summary['vacio_operacion']['mediana'], 3)} y percentil 95 {_fmt(summary['vacio_operacion']['p95'], 3)}. La magnitud es consistente con pulgadas de mercurio absolutas: cerca de 1.3 inHgA el agua satura alrededor de 31 °C, y la temperatura de condensador del historiador se mueve en esa zona si se lee en °F. Las temperaturas del archivo también caen en rango Fahrenheit. Conviene confirmar la unidad en la descripción del tag PI antes de publicar límites de planta.

La partición es cronológica: {summary['train']:,} entrenamiento ({periodo('train')}), {summary['val']:,} validación ({periodo('val')}) y {summary['test']:,} prueba ({periodo('test')}). La validación elige el modelo. El test se reporta una vez, sin usarlo para decidir.

## Qué no entra al modelo

La temperatura del condensador y la temperatura de vapor de escape acompañan la presión de saturación. Usarlas para “predecir” el vacío sería medir el mismo fenómeno dos veces. Tampoco entra el tag nominal `FNX:10PI3001.PV`, que en este archivo no tiene valores numéricos. El objetivo sigue siendo `FNX:S1_EV_P.PV`, el proxy ya justificado en el primer entregable.

Los drivers son agua de mar, potencia de TV10, potencia de TG11 y TG12, presión de vapor de eyectores, presión de vapor LP y flujo de agua de alimentación LP.

La correlación global del primer entregable estaba dominada por el modo fuera de servicio, donde el vacío salta cerca de 30. Dentro de la operación la variación es mucho más chica, y es la que importa para performance.

## Línea base física y anomalía

Una regresión lineal con agua de mar y potencia de TV, ajustada solo en entrenamiento y con variables escaladas, define el vacío esperado por condiciones de frontera. El residual es observado menos esperado. Sobre los residuales de entrenamiento:

- hasta el cuantil {summary['umbral_residual']['alerta']:.4f} (percentil 75): **normal**
- hasta el cuantil {summary['umbral_residual']['critico']:.4f} (percentil 95): **alerta**
- por encima: **crítico** / anomalía

Esas clases no son órdenes de trabajo etiquetadas. Son un proxy reproducible mientras no exista un registro de limpiezas e ingresos de aire. El contraste externo es la diferencia de temperaturas condensador − agua de mar, que no se usó como entrada.

Coeficientes de la línea base, en unidades de desviación estándar del entrenamiento: agua de mar {_fmt(summary['coef_linea_base_escalada'].get('temp_agua_mar'))}, potencia TV {_fmt(summary['coef_linea_base_escalada'].get('potencia_tv'))}. El signo positivo del agua de mar significa vacío peor (presión más alta) cuando el mar está más caliente.

## Vacío esperado

MAE de validación elige el modelo. Todas las filas de abajo están ajustadas solo con entrenamiento.

### Validación

{_table(summary['regresion'], 'val', 'mae', True)}

### Test

{_table(summary['regresion'], 'test', 'mae', True)}

En validación el menor MAE es **{winner}** ({_fmt(winner_val.get('mae'))}). El mejor modelo flexible en esa misma ventana es **{flexible_name}**, con MAE {_fmt(flexible_val.get('mae'))}. En test, la mediana de entrenamiento queda en MAE {_fmt(median_test.get('mae'))}, la línea agua/carga en {_fmt(base_test.get('mae'))} y {flexible_name} en {_fmt(flexible_test.get('mae'))}.

{"Ningún modelo flexible supera a la mediana en validación. La API no publica esa constante: la alarma está definida sobre la línea de agua y carga, y esa es la predicción que responde el servicio (`lineal_agua_y_carga`). Ridge puede quedar primero en el test final; esa tabla se muestra y no se usa para coronar un modelo." if not summary.get("flexible_supera_mediana") else "El modelo flexible supera a la mediana en validación y es el que sirve la API, reentrenado con entrenamiento y validación. Su MAE en test es " + _fmt(deployed.get("mae")) + "."}

El GridSearch de XGBoost, con `TimeSeriesSplit`, dejó {summary['params_xgb']} y un MAE de validación cruzada {_fmt(summary['mae_cv_xgb'])}. El bloque de validación y el de test no se parecen: el PSI de más abajo está por encima de 0.25 en agua de mar, carga y vacío. Por eso un modelo puede ganar la ventana intermedia y otro la ventana final.

VIF en entrenamiento, para no ignorar la redundancia entre carga, flujo y presiones:

| Variable | VIF |
| --- | --- |
{vif_lines}

## Degradación en tres clases

La clase sale del residual de la línea base. Los clasificadores no ven el vacío: tienen que reconocer la degradación desde los drivers de proceso. El dummy de clase frecuente es la referencia. Se elige por F1 macro de validación, porque la clase crítica es la minoría.

Conteo en entrenamiento: {summary['conteo_train']}.

### Validación

{_table(summary['clasificacion'], 'val', 'f1_macro', False)}

### Test

{_table(summary['clasificacion'], 'test', 'f1_macro', False)}

En validación gana **{class_winner}** (F1 macro {_fmt(class_val.get('f1_macro'))}). En test ese resultado vale F1 macro {_fmt(class_test.get('f1_macro'))} y accuracy {_fmt(class_test.get('accuracy'))}. El dummy en test llega a F1 macro {_fmt(dummy_test.get('f1_macro'))}.

La ventana de validación casi no contiene degradación: el dummy, que siempre dice "normal", acierta casi todas las filas. Por eso su F1 macro puede quedar arriba aunque no detecte un crítico. La matriz de `reports/figures/matriz_confusion.png` corresponde a **{summary.get('modelo_matriz', class_winner)}** en el test, donde sí hay casos de alerta y crítico. Ahí el contraste útil es el F1 macro, no el accuracy.

## Contraste del residual

Prueba de Mann-Whitney en el bloque de test, normal contra crítico. No se asume normalidad: el primer análisis ya mostró colas y regímenes distintos.

{chr(10).join(hypothesis_lines)}

Si la diferencia de temperatura condensador − agua de mar es mayor en el grupo crítico, el residual no es solo un artefacto del modelo: el condensador está más caliente de lo que el agua de mar explica.

## Pronóstico a {summary['horizonte_min']} minutos

Se construyen lags de 15 min, 30 min, 1 h, 2 h y 24 h, y solo se acepta un lag si el reloj confirma esa distancia. El objetivo es el vacío futuro. La persistencia (el vacío de ahora) y la persistencia diaria son los baselines del módulo de series de tiempo. La red comparada es un MLP sobre esa ventana: el diplomado trabaja MLP, LSTM y transformers; aquí corre el MLP porque es el modelo neuronal que el entorno puede entrenar sin TensorFlow.

ADF sobre el vacío de entrenamiento: estadístico {_fmt(summary['adf_estadistico'], 3)}, p-valor {_fmt(adf_p, 4)}. Con ese resultado, {stationary}. Filas con ventana válida: {summary['pronostico_filas']:,}.

### Validación

{_table(summary['pronostico'], 'val', 'mae', True)}

### Test

{_table(summary['pronostico'], 'test', 'mae', True)}

{"Gana la persistencia: MAE de test " + _fmt(persist_test.get("mae")) + "." if forecast_winner == "persistencia" else "Ganador de validación: **" + forecast_winner + "**. MAE de test " + _fmt(forecast_test.get("mae")) + ", frente a persistencia " + _fmt(persist_test.get("mae")) + "."} En un condensador el vacío cambia lento: a una hora la persistencia es difícil de mejorar, y en este histórico no se mejora. El MLP sobre la ventana tampoco la alcanza. El aporte del estudio no es reemplazar la inercia del proceso, sino separar la parte que agua de mar y carga ya explican.

## Regímenes

K-means sobre los drivers estandarizados del entrenamiento. Silhouette en una muestra ordenada: {silhouette}. k elegido: **{summary['clusters']['k']}**.

{_markdown(cluster_frame.round(3))}

Los clusters describen modos de carga, no la degradación. El grupo grande es ciclo combinado con las dos turbinas de gas en servicio. Los grupos con una turbina de gas casi en cero tienen menos potencia de vapor y un vacío más bajo, que es lo esperable cuando hay menos vapor por condensar. El grupo más chico se separa por una presión de eyectores claramente menor.

## Estabilidad entre entrenamiento y test

PSI por variable. Por debajo de 0.10 se considera estable; entre 0.10 y 0.25, cambio moderado; por encima, drift que justificaría revisar el modelo antes de seguir usándolo.

| Variable | PSI | Banda |
| --- | --- | --- |
{psi_lines}

## Qué toma del diplomado

- Estadística: limpieza, faltantes, Spearman, VIF, Mann-Whitney y cuantiles para definir clases.
- Supervisado: árbol, Random Forest, GBM, AdaBoost, XGBoost, GridSearch y curvas de aprendizaje con partición temporal.
- No supervisado: K-means y silhouette sobre regímenes de operación.
- Series de tiempo: estacionariedad, lags, ventanas, persistencia y MLP.
- MLE: `src/`, `params.yaml`, pipeline, pytest, DVC descrito en `dvc.yaml` y Git.
- Despliegue: FastAPI para una lectura y para un CSV, contenedor Docker y PSI como control de drift.

Quedan fuera, porque no aportan a este equipo, el scraping, el análisis de texto, RFM y las reglas de asociación.

## Límite

Sin bitácora de limpieza de tubos, pruebas de ingreso de aire o estado de bombas de circulación, la clase crítica es una desviación estadística y no un diagnóstico cerrado de causa. El siguiente dato que más cambiaría el estudio es el flujo y la temperatura de entrada y salida del agua de circulación, más el evento de limpieza.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
