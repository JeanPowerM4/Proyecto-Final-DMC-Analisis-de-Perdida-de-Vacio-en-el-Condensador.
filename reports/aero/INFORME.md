# Contrapresión del aerocondensador

Misma pregunta que en el condensador de superficie: con la unidad en servicio, qué presión de escape cabe esperar y cuándo el valor observado es peor que esa referencia.

La señal objetivo es `TV_BP_Pout`, la contrapresión de baja presión. Su magnitud, junto con `TV_BP_Tout`, es consistente con bar absolutos: alrededor de 0.10 bar el agua satura cerca de 46 °C, que es la zona de la temperatura de salida. La potencia no viene en MW. El archivo trae el factor de capacidad `CF`; la potencia usada aquí es CF × 300 MW.

## Datos

El archivo tiene 51,542 filas y 35,507 quedan con fecha, contrapresión y drivers completos, del 2012-08-15 08:31:00 al 2013-10-03 16:49:00. El muestreo válido es de 1 minuto, en bloques separados por huecos. Ningún bloque continuo llega a 24 horas, así que no hay persistencia diaria identificable.

El factor de capacidad se mueve entre 0.895 y 0.996, con mediana 0.976. No aparece el modo fuera de servicio que en el condensador de superficie empujaba el vacío hacia 30. Toda la muestra ya es carga alta.

La contrapresión tiene mediana 0.098 y percentil 95 0.119. Partición cronológica: 24,854 entrenamiento, 5,326 validación y 5,327 prueba.

## Qué entra y qué no

Entran temperatura ambiente, potencia, velocidad del viento, flujo de vapor de baja presión y la dirección del viento en seno y coseno. No entra `TV_BP_Tout`: acompaña la temperatura de saturación de la propia contrapresión, igual que la temperatura del condensador en el caso de superficie.

`CF` y la temperatura ambiente se mueven juntos en sentido contrario. Una correlación negativa de la carga con la contrapresión, mirándola sola, no quiere decir que generar más mejore el vacío: en este archivo la carga baja cuando el ambiente sube. El coeficiente parcial de la línea base es el que separa esos dos efectos.

Coeficientes escalados de la línea ambiente + carga: ambiente 0.0051, potencia -0.0050. El ambiente empuja la contrapresión hacia arriba. El de la potencia queda levemente negativo dentro de una banda de factor de capacidad muy estrecha; no se lee como regla de diseño. El nivel lo pone el ambiente.

## Contrapresión esperada

### Validación

| modelo | mae | rmse | r2 | mape_pct |
| --- | --- | --- | --- | --- |
| xgboost | 0.0030 | 0.0037 | 0.8020 | 3.1081 |
| xgboost_gridsearch | 0.0031 | 0.0038 | 0.7950 | 3.2062 |
| random_forest | 0.0031 | 0.0040 | 0.7762 | 3.2905 |
| gbm | 0.0032 | 0.0039 | 0.7847 | 3.3000 |
| lineal_ambiente_y_carga | 0.0032 | 0.0045 | 0.7186 | 3.3374 |
| ridge | 0.0032 | 0.0041 | 0.7631 | 3.3832 |
| arbol | 0.0035 | 0.0043 | 0.7400 | 3.7245 |
| mediana_entrenamiento | 0.0085 | 0.0100 | -0.4112 | 9.2692 |

### Test

| modelo | mae | rmse | r2 | mape_pct |
| --- | --- | --- | --- | --- |
| xgboost | 0.0026 | 0.0033 | 0.8425 | 2.8026 |
| xgboost_gridsearch | 0.0027 | 0.0033 | 0.8392 | 2.8834 |
| gbm | 0.0028 | 0.0034 | 0.8345 | 2.9447 |
| random_forest | 0.0033 | 0.0040 | 0.7629 | 3.4769 |
| arbol | 0.0033 | 0.0041 | 0.7505 | 3.5175 |
| ridge | 0.0034 | 0.0047 | 0.6834 | 3.6051 |
| lineal_ambiente_y_carga | 0.0035 | 0.0059 | 0.4961 | 3.6147 |
| mediana_entrenamiento | 0.0086 | 0.0102 | -0.5193 | 9.3972 |

En validación el menor MAE es **xgboost** (0.0030). El mejor flexible es **xgboost** (validación 0.0030, test 0.0026). En test la mediana queda en 0.0086 y la línea ambiente/carga en 0.0035.

El modelo flexible supera a la mediana y queda como referencia: xgboost.

VIF:

| Variable | VIF |
| --- | --- |
| temp_ambiente | 1.73 |
| potencia_mw | 2.39 |
| viento | 2.19 |
| flujo_bp | 1.39 |
| viento_sin | 1.53 |
| viento_cos | 1.23 |

## Degradación

Las clases salen del residual de la línea ambiente/carga, con los mismos percentiles 75 y 95 del entrenamiento. Conteo de entrenamiento: {'normal': 18640, 'alerta': 4971, 'critico': 1243}.

### Validación

| modelo | accuracy | f1_macro | f1_weighted |
| --- | --- | --- | --- |
| arbol | 0.6714 | 0.5417 | 0.6742 |
| random_forest | 0.6996 | 0.5018 | 0.6885 |
| xgboost | 0.6827 | 0.4930 | 0.6811 |
| adaboost | 0.6564 | 0.4927 | 0.6672 |
| gbm | 0.6671 | 0.4656 | 0.6664 |
| dummy_frecuente | 0.6879 | 0.2717 | 0.5608 |

### Test

| modelo | accuracy | f1_macro | f1_weighted |
| --- | --- | --- | --- |
| xgboost | 0.6893 | 0.6546 | 0.7174 |
| random_forest | 0.7085 | 0.6476 | 0.7246 |
| gbm | 0.6903 | 0.6309 | 0.7149 |
| adaboost | 0.6523 | 0.6197 | 0.6865 |
| arbol | 0.6843 | 0.6085 | 0.7068 |
| dummy_frecuente | 0.7699 | 0.2900 | 0.6697 |

Gana la validación **arbol**. En test su F1 macro es 0.6085. La matriz corresponde a **arbol**.

## Contraste del residual

- **ttd**: mediana normal 26.831, mediana crítico 30.706, p-valor 0.0000.
- **temp_salida_bp**: mediana normal 44.928, mediana crítico 49.494, p-valor 0.0000.
- **temp_ambiente**: mediana normal 18.294, mediana crítico 18.545, p-valor 0.0002.
- **viento**: mediana normal 4.800, mediana crítico 22.500, p-valor 0.0000.
- **potencia_mw**: mediana normal 294.213, mediana crítico 290.391, p-valor 0.0000.

El viento separa el grupo crítico con más claridad que la temperatura ambiente. La alarma no es simplemente un día más caluroso: es contrapresión peor de lo que el ambiente y la carga ya explican, y en el test eso coincide con viento más fuerte.

## Pronóstico a 60 minutos

Lags de 15, 30, 60 y 120 minutos, solo si el reloj confirma esa distancia. Filas con ventana válida: 22,875. ADF: estadístico -5.752, p-valor 0.0000; se rechaza la raíz unitaria al 5%.

### Validación

| modelo | mae | rmse | r2 | mape_pct |
| --- | --- | --- | --- | --- |
| persistencia | 0.0030 | 0.0041 | 0.6761 | 3.0670 |
| ridge_lags | 0.0031 | 0.0041 | 0.6727 | 3.2009 |
| random_forest_lags | 0.0048 | 0.0058 | 0.3637 | 4.9530 |
| mlp_ventanas | 0.0081 | 0.0110 | -1.3380 | 8.3986 |

### Test

| modelo | mae | rmse | r2 | mape_pct |
| --- | --- | --- | --- | --- |
| persistencia | 0.0030 | 0.0042 | 0.7659 | 3.0498 |
| ridge_lags | 0.0031 | 0.0041 | 0.7744 | 3.1951 |
| random_forest_lags | 0.0042 | 0.0053 | 0.6298 | 4.3969 |
| mlp_ventanas | 0.0095 | 0.0133 | -1.3250 | 10.0344 |

Gana la validación **persistencia**, con MAE de test 0.0030. La persistencia queda en 0.0030.

## Dirección del viento

Mediana de contrapresión por dirección, en toda la muestra válida:

| direccion | vacio |
| --- | --- |
| WSW | 0.0920 |
| SW | 0.0920 |
| ENE | 0.0940 |
| NE | 0.0940 |
| WNW | 0.0960 |
| NNE | 0.0970 |
| W | 0.0990 |
| ESE | 0.0990 |
| N | 0.1000 |
| SSW | 0.1050 |
| NNW | 0.1050 |
| E | 0.1070 |
| NW | 0.1070 |
| SSE | 0.1200 |

## Regímenes

k elegido: **4**.

| cluster | temp_ambiente | potencia_mw | viento | flujo_bp | viento_sin | viento_cos | vacio | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 18.7970 | 294.2430 | 3.2000 | 34.3040 | 0.3830 | 0.9240 | 0.0930 | 8060 |
| 1 | 19.9910 | 290.4970 | 16.1000 | 33.9600 | 1.0000 | 0.0000 | 0.1070 | 3787 |
| 2 | 19.5510 | 293.5940 | 3.2000 | 34.3980 | -0.9240 | 0.3830 | 0.0980 | 5192 |
| 3 | 22.3820 | 289.4700 | 8.0000 | 33.8680 | 0.0000 | 0.7070 | 0.1110 | 7815 |

## Estabilidad

| Variable | PSI | Banda |
| --- | --- | --- |
| temp_ambiente | 1.125 | drift |
| potencia_mw | 0.586 | drift |
| viento | 0.151 | cambio_moderado |
| flujo_bp | 1.097 | drift |
| viento_sin | 0.241 | cambio_moderado |
| viento_cos | 0.280 | drift |
| vacio | 0.915 | drift |
