# Vacío del condensador de superficie

Estudio de Jean Piere Cholán para el proyecto integrador del Diploma Advanced Data Scientist.
El alcance es el **condensador de superficie enfriado por agua de mar**. El aerocondensador queda fuera de este repositorio.

## Pregunta

Con la turbina de vapor en servicio, ¿qué vacío cabe esperar a partir del agua de mar, la carga y el sistema de extracción de aire, y cuándo el vacío observado es peor que esa referencia?

Esa diferencia es la señal de pérdida de desempeño del condensador: ensuciamiento, ingreso de aire o falta de enfriamiento. No es un detector de unidad fuera de servicio.

## Datos

El extracto del historiador cubre del 2024-01-01 00:59:59.999999786 al 2026-01-01 00:59:59.999999786, con paso mediano de 15 minutos. Después de convertir textos de adquisición (`Configure`, `Bad`, `Tag not found`) a faltantes, quedan 70,176 filas. El régimen de modelado exige potencia de TV10 de al menos 50 MW y vacío entre 0.4 y 5. Entran 58,867 filas.

En ese régimen el vacío tiene mediana 1.302 y percentil 95 1.797. La magnitud es consistente con pulgadas de mercurio absolutas: cerca de 1.3 inHgA el agua satura alrededor de 31 °C, y la temperatura de condensador del historiador se mueve en esa zona si se lee en °F. Las temperaturas del archivo también caen en rango Fahrenheit. Conviene confirmar la unidad en la descripción del tag PI antes de publicar límites de planta.

La partición es cronológica: 41,206 entrenamiento (2024-01-01 a 2025-06-12, vacío mediano 1.300, agua de mar 59.8), 8,830 validación (2025-06-12 a 2025-09-13, vacío mediano 1.342, agua de mar 61.2) y 8,831 prueba (2025-09-13 a 2026-01-01, vacío mediano 1.267, agua de mar 58.0). La validación elige el modelo. El test se reporta una vez, sin usarlo para decidir.

## Qué no entra al modelo

La temperatura del condensador y la temperatura de vapor de escape acompañan la presión de saturación. Usarlas para “predecir” el vacío sería medir el mismo fenómeno dos veces. Tampoco entra el tag nominal `FNX:10PI3001.PV`, que en este archivo no tiene valores numéricos. El objetivo sigue siendo `FNX:S1_EV_P.PV`, el proxy ya justificado en el primer entregable.

Los drivers son agua de mar, potencia de TV10, potencia de TG11 y TG12, presión de vapor de eyectores, presión de vapor LP y flujo de agua de alimentación LP.

La correlación global del primer entregable estaba dominada por el modo fuera de servicio, donde el vacío salta cerca de 30. Dentro de la operación la variación es mucho más chica, y es la que importa para performance.

## Línea base física y anomalía

Una regresión lineal con agua de mar y potencia de TV, ajustada solo en entrenamiento y con variables escaladas, define el vacío esperado por condiciones de frontera. El residual es observado menos esperado. Sobre los residuales de entrenamiento:

- hasta el cuantil 0.0225 (percentil 75): **normal**
- hasta el cuantil 0.4357 (percentil 95): **alerta**
- por encima: **crítico** / anomalía

Esas clases no son órdenes de trabajo etiquetadas. Son un proxy reproducible mientras no exista un registro de limpiezas e ingresos de aire. El contraste externo es la diferencia de temperaturas condensador − agua de mar, que no se usó como entrada.

Coeficientes de la línea base, en unidades de desviación estándar del entrenamiento: agua de mar 0.1531, potencia TV 0.1368. El signo positivo del agua de mar significa vacío peor (presión más alta) cuando el mar está más caliente.

## Vacío esperado

MAE de validación elige el modelo. Todas las filas de abajo están ajustadas solo con entrenamiento.

### Validación

| modelo | mae | rmse | r2 | mape_pct |
| --- | --- | --- | --- | --- |
| mediana_entrenamiento | 0.0712 | 0.0972 | -0.0793 | 5.6794 |
| arbol | 0.0754 | 0.1491 | -1.5410 | 5.8395 |
| lineal_agua_y_carga | 0.1202 | 0.1342 | -1.0585 | 9.1371 |
| random_forest | 0.1264 | 0.1966 | -3.4182 | 9.5887 |
| xgboost | 0.1313 | 0.1737 | -2.4480 | 9.9447 |
| ridge | 0.1315 | 0.1473 | -1.4780 | 9.9699 |
| gbm | 0.1348 | 0.1744 | -2.4766 | 10.1569 |
| xgboost_gridsearch | 0.1435 | 0.1813 | -2.7550 | 10.8132 |

### Test

| modelo | mae | rmse | r2 | mape_pct |
| --- | --- | --- | --- | --- |
| ridge | 0.0525 | 0.0982 | 0.5294 | 4.1181 |
| lineal_agua_y_carga | 0.0542 | 0.1121 | 0.3872 | 4.2335 |
| arbol | 0.0843 | 0.1641 | -0.3137 | 6.7390 |
| random_forest | 0.0869 | 0.1538 | -0.1546 | 6.7400 |
| mediana_entrenamiento | 0.0911 | 0.1539 | -0.1562 | 8.6898 |
| gbm | 0.0956 | 0.1414 | 0.0248 | 7.5513 |
| xgboost | 0.1002 | 0.1484 | -0.0751 | 7.8814 |
| xgboost_gridsearch | 0.1009 | 0.1463 | -0.0451 | 7.9411 |

En validación el menor MAE es **mediana_entrenamiento** (0.0712). El mejor modelo flexible en esa misma ventana es **arbol**, con MAE 0.0754. En test, la mediana de entrenamiento queda en MAE 0.0911, la línea agua/carga en 0.0542 y arbol en 0.0843.

Ningún modelo flexible supera a la mediana en validación. La API no publica esa constante: la alarma está definida sobre la línea de agua y carga, y esa es la predicción que responde el servicio (`lineal_agua_y_carga`). Ridge puede quedar primero en el test final; esa tabla se muestra y no se usa para coronar un modelo.

El GridSearch de XGBoost, con `TimeSeriesSplit`, dejó {'model__learning_rate': 0.1, 'model__max_depth': 3} y un MAE de validación cruzada 0.1832. El bloque de validación y el de test no se parecen: el PSI de más abajo está por encima de 0.25 en agua de mar, carga y vacío. Por eso un modelo puede ganar la ventana intermedia y otro la ventana final.

VIF en entrenamiento, para no ignorar la redundancia entre carga, flujo y presiones:

| Variable | VIF |
| --- | --- |
| temp_agua_mar | 1.05 |
| potencia_tv | 8.13 |
| potencia_tg11 | 5.17 |
| potencia_tg12 | 22.88 |
| presion_eyectores | 1.04 |
| presion_vapor_lp | 4.09 |
| flujo_fw_lp | 21.72 |

## Degradación en tres clases

La clase sale del residual de la línea base. Los clasificadores no ven el vacío: tienen que reconocer la degradación desde los drivers de proceso. El dummy de clase frecuente es la referencia. Se elige por F1 macro de validación, porque la clase crítica es la minoría.

Conteo en entrenamiento: {'normal': 30904, 'alerta': 8241, 'critico': 2061}.

### Validación

| modelo | accuracy | f1_macro | f1_weighted |
| --- | --- | --- | --- |
| dummy_frecuente | 0.9964 | 0.3327 | 0.9946 |
| random_forest | 0.8087 | 0.3115 | 0.8910 |
| adaboost | 0.7437 | 0.2965 | 0.8498 |
| gbm | 0.7173 | 0.2899 | 0.8324 |
| xgboost | 0.6959 | 0.2852 | 0.8176 |
| arbol | 0.7080 | 0.2824 | 0.8260 |

### Test

| modelo | accuracy | f1_macro | f1_weighted |
| --- | --- | --- | --- |
| random_forest | 0.6130 | 0.4459 | 0.6506 |
| arbol | 0.5609 | 0.4230 | 0.6035 |
| xgboost | 0.5679 | 0.4222 | 0.6070 |
| gbm | 0.5555 | 0.4140 | 0.5988 |
| adaboost | 0.5483 | 0.4092 | 0.5848 |
| dummy_frecuente | 0.7152 | 0.2780 | 0.5965 |

En validación gana **dummy_frecuente** (F1 macro 0.3327). En test ese resultado vale F1 macro 0.2780 y accuracy 0.7152. El dummy en test llega a F1 macro 0.2780.

La ventana de validación casi no contiene degradación: el dummy, que siempre dice "normal", acierta casi todas las filas. Por eso su F1 macro puede quedar arriba aunque no detecte un crítico. La matriz de `reports/figures/matriz_confusion.png` corresponde a **random_forest** en el test, donde sí hay casos de alerta y crítico. Ahí el contraste útil es el F1 macro, no el accuracy.

## Contraste del residual

Prueba de Mann-Whitney en el bloque de test, normal contra crítico. No se asume normalidad: el primer análisis ya mostró colas y regímenes distintos.

- **ttd**: mediana normal 28.183, mediana crítico 35.602, p-valor 0.0000 (n normal 6316, n crítico 74).
- **temp_escape**: mediana normal 91.794, mediana crítico 98.640, p-valor 0.0000 (n normal 6316, n crítico 74).
- **presion_eyectores**: mediana normal 150.157, mediana crítico 149.913, p-valor 0.0397 (n normal 6316, n crítico 74).
- **temp_agua_mar**: mediana normal 58.306, mediana crítico 57.375, p-valor 0.0000 (n normal 6316, n crítico 74).

Si la diferencia de temperatura condensador − agua de mar es mayor en el grupo crítico, el residual no es solo un artefacto del modelo: el condensador está más caliente de lo que el agua de mar explica.

## Pronóstico a 60 minutos

Se construyen lags de 15 min, 30 min, 1 h, 2 h y 24 h, y solo se acepta un lag si el reloj confirma esa distancia. El objetivo es el vacío futuro. La persistencia (el vacío de ahora) y la persistencia diaria son los baselines del módulo de series de tiempo. La red comparada es un MLP sobre esa ventana: el diplomado trabaja MLP, LSTM y transformers; aquí corre el MLP porque es el modelo neuronal que el entorno puede entrenar sin TensorFlow.

ADF sobre el vacío de entrenamiento: estadístico -8.510, p-valor 0.0000. Con ese resultado, se rechaza la raíz unitaria al 5%. Filas con ventana válida: 54,509.

### Validación

| modelo | mae | rmse | r2 | mape_pct |
| --- | --- | --- | --- | --- |
| persistencia | 0.0132 | 0.0416 | 0.7702 | 1.0691 |
| random_forest_lags | 0.0166 | 0.0387 | 0.8017 | 1.3309 |
| ridge_lags | 0.0199 | 0.0420 | 0.7658 | 1.5956 |
| mlp_ventanas | 0.0225 | 0.0678 | 0.3905 | 1.8259 |
| persistencia_diaria | 0.0613 | 0.1141 | -0.7259 | 5.0105 |

### Test

| modelo | mae | rmse | r2 | mape_pct |
| --- | --- | --- | --- | --- |
| persistencia | 0.0246 | 0.0784 | 0.6937 | 2.0519 |
| random_forest_lags | 0.0306 | 0.0767 | 0.7065 | 2.5854 |
| ridge_lags | 0.0327 | 0.0790 | 0.6890 | 2.7545 |
| mlp_ventanas | 0.0351 | 0.0924 | 0.5743 | 2.9544 |
| persistencia_diaria | 0.1035 | 0.1849 | -0.7040 | 9.3574 |

Gana la persistencia: MAE de test 0.0246. En un condensador el vacío cambia lento: a una hora la persistencia es difícil de mejorar, y en este histórico no se mejora. El MLP sobre la ventana tampoco la alcanza. El aporte del estudio no es reemplazar la inercia del proceso, sino separar la parte que agua de mar y carga ya explican.

## Regímenes

K-means sobre los drivers estandarizados del entrenamiento. Silhouette en una muestra ordenada: k=2: 0.482, k=3: 0.501, k=4: 0.526, k=5: 0.536, k=6: 0.506. k elegido: **5**.

| cluster | temp_agua_mar | potencia_tv | potencia_tg11 | potencia_tg12 | presion_eyectores | presion_vapor_lp | flujo_fw_lp | vacio | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 59.4900 | 185.2290 | 181.7940 | 181.8170 | 149.9730 | 53.1630 | 236.8890 | 1.3150 | 23324 |
| 1 | 60.4760 | 75.9830 | 0.1380 | 126.4740 | 149.9330 | 44.8960 | 172.0070 | 0.7720 | 2927 |
| 2 | 60.4550 | 153.6430 | 124.6550 | 124.5150 | 149.9630 | 46.2430 | 166.8910 | 1.2120 | 13757 |
| 3 | 61.0750 | 74.0290 | 0.1560 | 123.3490 | 114.4460 | 44.6050 | 163.5090 | 0.9720 | 152 |
| 4 | 61.1210 | 87.6390 | 174.9560 | 0.1830 | 149.5810 | 32.9190 | 0.0090 | 0.8250 | 1046 |

Los clusters describen modos de carga, no la degradación. El grupo grande es ciclo combinado con las dos turbinas de gas en servicio. Los grupos con una turbina de gas casi en cero tienen menos potencia de vapor y un vacío más bajo, que es lo esperable cuando hay menos vapor por condensar. El grupo más chico se separa por una presión de eyectores claramente menor.

## Estabilidad entre entrenamiento y test

PSI por variable. Por debajo de 0.10 se considera estable; entre 0.10 y 0.25, cambio moderado; por encima, drift que justificaría revisar el modelo antes de seguir usándolo.

| Variable | PSI | Banda |
| --- | --- | --- |
| temp_agua_mar | 1.520 | drift |
| potencia_tv | 0.591 | drift |
| potencia_tg11 | 0.401 | drift |
| potencia_tg12 | 0.405 | drift |
| presion_eyectores | 0.103 | cambio_moderado |
| presion_vapor_lp | 1.358 | drift |
| flujo_fw_lp | 0.460 | drift |
| vacio | 0.799 | drift |

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
