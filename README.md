# Vacío del condensador de superficie

Proyecto de Jean Piere Cholán para el Diploma Advanced Data Scientist de DMC.
El caso es la pérdida de vacío en un ciclo combinado. Este repositorio cubre el **condensador de superficie enfriado por agua de mar**. El aerocondensador es el alcance del otro integrante.

El historiador es de proceso real. No se sube el Excel a Git. El código, las métricas, las figuras y el modelo sí.

## Resultado

Sobre 58,867 puntos con la TV en servicio, de enero 2024 a enero 2026:

- A una hora, la persistencia gana. MAE de test 0.025, por debajo del random forest con lags (0.031) y del MLP.
- Ningún modelo flexible supera a la mediana en la validación de mitad de 2025. En el test final, Ridge (MAE 0.053) y la línea de agua de mar + carga (MAE 0.054) sí quedan por debajo de la mediana (0.091). El test no se usa para elegir. La API sirve la línea física, que es también la referencia de la alarma.
- Esa alarma, en test, coincide con un condensador más caliente: diferencia condensador − agua de mar de 28.2 en operación normal y 35.6 en la clase crítica (Mann-Whitney, p ≈ 0).
- Hay drift entre entrenamiento y test (PSI del agua de mar 1.52). Conviene no promover un ensemble solo porque ajusta el primer año.

El detalle, las tablas y la lectura de clusters están en `reports/INFORME.md`.

## Qué responde

Con la turbina de vapor en servicio:

1. Qué vacío cabe esperar a partir del agua de mar, la carga y el sistema de eyectores.
2. Cuándo el vacío observado es peor que esa referencia (normal, alerta, crítico).
3. Si un pronóstico a 1 hora mejora la persistencia.
4. Si el régimen de operación cambió entre el entrenamiento y el test (PSI).

La temperatura del condensador no entra como variable explicativa: sigue de cerca la presión de saturación del propio vacío.

## Cómo correrlo

Desde esta carpeta, con el Excel en `../00_Final_Project/Data_Condensador.xlsx`:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python main.py train
.venv\Scripts\pytest -q
.venv\Scripts\uvicorn api.main:app --port 8000
```

Un instante:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @data/external/snapshot_ejemplo.json
```

Un lote:

```bash
.venv\Scripts\python main.py predict --data data/external/lote_ejemplo.csv
```

Contenedor, después de entrenar:

```bash
docker build -t condensador-superficie .
docker run --rm -p 8000:8000 condensador-superficie
```

## Dónde está cada resultado

| Pieza | Ruta |
| --- | --- |
| Informe | `reports/INFORME.md` |
| Métricas | `reports/metrics/` |
| Figuras | `reports/figures/` |
| Modelo servido | `artifacts/models/vacuum_bundle.joblib` |
| Parámetros | `params.yaml` |

El informe deja el mapeo con los módulos del diplomado: estadística, ensembles y tuning, clusters, series de tiempo, estructura MLE, API y drift.

## Estructura

```text
src/data          carga, limpieza y diccionario de tags
src/features      régimen operativo, split temporal, lags
src/models        métricas e inferencia
src/monitoring    PSI
src/reporting     figuras e informe
src/training.py   comparación de modelos
api               FastAPI
tests             pytest
```

`dvc.yaml` describe la etapa de entrenamiento. El dato crudo se queda fuera de Git por el mismo motivo del taller de DVC: no versionar el historiador dentro del repositorio de código.
