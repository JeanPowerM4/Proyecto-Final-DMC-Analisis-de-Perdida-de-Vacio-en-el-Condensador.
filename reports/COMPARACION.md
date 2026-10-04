# Superficie y aerocondensador

Los dos estudios hacen la misma pregunta: con la turbina en servicio, qué presión de condensación cabe esperar a partir del fluido frío y de la carga, y cuándo lo observado es peor que esa referencia. No son el mismo equipo ni el mismo año. La comparación sirve para ver qué se repite de la física y qué cambia con la tecnología.

## Nivel de presión

| | Superficie | Aerocondensador |
| --- | --- | --- |
| Señal | `FNX:S1_EV_P.PV` | `TV_BP_Pout` |
| Lectura de unidad | inHg absolutos, por la magnitud | bar absolutos, porque la temperatura de salida calza con la saturación |
| Mediana en servicio | 1.302 inHgA, cerca de 44.1 mbar | 0.098 bar, 98.0 mbar |
| Percentil 95 | 1.797 | 0.119 |
| Periodo | 2024-01-01 a 2026-01-01 | 2012-08-15 a 2013-10-03 |
| Paso | 15 min | 1 min, en bloques con huecos |
| Filas de modelado | 58,867 | 35,507 |
| Carga | desde 50 MW, con paradas fuera del modelo | CF 0.895 a 0.996, potencia = CF × 300 MW |

En números redondos, el aerocondensador condensa cerca de 2.22 veces la presión absoluta del condensador de superficie. Es lo esperable: el aire ambiente enfría peor que el agua de mar. La muestra del aerocondensador, además, ya está en base. La de superficie tuvo que recortar el modo fuera de servicio, donde el vacío se iba hacia 30.

## La misma línea física

En superficie la línea es agua de mar + potencia de TV. En el aerocondensador es temperatura ambiente + potencia. En las dos, el coeficiente del fluido frío es positivo: más calor en la fuente fría, más presión de condensación.

| Test | Superficie | Aerocondensador |
| --- | --- | --- |
| MAE de la mediana | 0.0911 | 0.0086 |
| MAE de la línea fría + carga | 0.0542 | 0.0035 |
| MAPE de esa línea | 4.23 % | 3.61 % |
| R² de esa línea | 0.3872 | 0.4961 |
| Mejor MAE del test | ridge 0.0525 | xgboost 0.0026 |
| MAE / mediana | línea 0.042 | línea 0.036 |
| Ganador de validación | mediana_entrenamiento | xgboost |
| Modelo que queda de referencia | lineal_agua_y_carga | xgboost |

El MAE no se compara en crudo: las unidades no son las mismas. El MAPE y el MAE dividido por la mediana sí. En el test, la línea física de superficie queda en 4.2 % de su mediana y la del aerocondensador en 3.6 %.

La diferencia está en qué modelo se sostiene al salir del entrenamiento. En superficie la validación la gana la mediana y ningún ensemble se promueve: el régimen cambia y la referencia que queda es la línea de agua y carga. En el aerocondensador la validación la gana xgboost. La muestra ya viene en base, sin el salto de parada, y la contrapresión sigue al ambiente. La línea ambiente + carga ya deja un R² de test de 0.4961 y un MAPE de 3.61 %; el modelo elegido en validación llega a MAE 0.0026 en el test.

## Una hora hacia adelante

| Test | Superficie | Aerocondensador |
| --- | --- | --- |
| Horizonte | 60 min | 60 min |
| Ganador de validación | persistencia | persistencia |
| MAE del ganador | 0.0246 | 0.0030 |
| MAE de la persistencia | 0.0246 | 0.0030 |
| Persistencia / mediana | 0.019 | 0.030 |

La presión de condensación cambia lento. A una hora, la persistencia es el baseline que hay que ganarle. En superficie no se le gana. En el aerocondensador el ganador de validación es persistencia.

El archivo del aerocondensador no tiene un bloque continuo de 24 horas, así que la persistencia diaria que sí se pudo calcular en superficie aquí no existe.

## La alarma

En los dos casos la clase crítica es el residual por encima del percentil 95 de entrenamiento, y la temperatura caliente del condensador no entra al modelo. Sirve de contraste.

| | Superficie | Aerocondensador |
| --- | --- | --- |
| Diferencia caliente − frío, normal | 28.18 | 26.83 |
| Diferencia caliente − frío, crítico | 35.60 | 30.71 |
| p-valor | 0.0000 | 0.0000 |

Si el grupo crítico queda más caliente, el residual no es solo un recorte estadístico: el condensador está rechazando peor el calor.

## Qué cambia con la tecnología

El condensador de superficie se explica sobre todo por el agua de mar y la carga. Los eyectores y el vapor de baja presión agregan contexto de planta, con redundancia alta entre potencias y flujos.

El aerocondensador agrega viento. La velocidad entra al modelo y la dirección queda como seno y coseno, porque un viento que recircula aire caliente no es lo mismo que uno que barre el haz. En el test, el grupo crítico tiene viento mediano 22.5 frente a 4.8 en el normal, mientras la temperatura ambiente casi no se mueve (18.3 contra 18.5). El residual se alinea con viento fuerte.

El drift también se repite. PSI del agua de mar entre entrenamiento y test: 1.52. PSI de la temperatura ambiente: 1.12. Un modelo ajustado al primer tramo no se promueve solo porque el error de entrenamiento se vea bien.

## Lectura para el proyecto

Las dos tecnologías sostienen la misma cadena: fluido frío y carga fijan la presión esperada; el residual marca la pérdida de desempeño; a una hora manda la inercia del proceso. El aerocondensador opera a mayor presión absoluta y su muestra ya viene en base, así que el problema no es detectar la parada. Es ver la degradación dentro de una banda estrecha, con el ambiente y el viento como condiciones de frontera.

El dato que falta en superficie es el flujo de agua de mar. En el aerocondensador falta el estado de los ventiladores. Sin eso, la clase crítica sigue siendo una desviación respecto de la línea física, no una causa cerrada.
