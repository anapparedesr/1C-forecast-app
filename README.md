# Modelo ML-predsales

Repositorio de un modelo de Machine Learning para predecir ventas mensuales por tienda y producto, desarrollado como parte del MGE.

## Descripción del Proyecto

El objetivo es predecir la cantidad de unidades vendidas (`item_cnt_month`) para cada combinación de tienda y producto en el mes 34, usando como features los lags históricos de ventas (1, 3, 6 y 12 meses).

El modelo base es un **Random Forest Regressor** entrenado sobre un grid completo de combinaciones tienda-producto-mes, con optimización de hiperparámetros mediante **RandomizedSearchCV**.

**Dataset:** [Predict Future Sales — Kaggle](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data)

---

## Estructura del Repositorio

```
ModeloML-predsales/
├── data/
│   ├── raw/                  # Datos crudos (no versionados)
│   ├── prep/                 # Datos preparados para entrenamiento
│   ├── inference/            # Datos preparados para inferencia
│   └── predictions/          # Predicciones generadas
├── artifacts/                # Modelos entrenados (.pkl)
│   └── logs/                 # Logs de ejecución
├── notebook/                 # Notebooks exploratorios
└── src/
    ├── preprocessing/
    │   ├── prep.py           # Preparación de datos de entrenamiento
    │   ├── prep_inference.py # Preparación de datos de inferencia
    │   ├── __main__.py       # Entry point con argparse
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   ├── utils/
    │   │   ├── logger.py
    │   │   ├── metrics.py
    │   │   └── data_validation.py
    │   └── test/
    │       └── test_prep.py
    ├── training/
    │   ├── train.py          # Entrenamiento con RandomizedSearchCV
    │   ├── __main__.py       # Entry point con argparse
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   ├── utils/
    │   └── test/
    │       └── test_train.py
    └── inference/
        ├── inference.py      # Generación de predicciones
        ├── __main__.py       # Entry point con argparse
        ├── Dockerfile
        ├── requirements.txt
        ├── utils/
        └── test/
            └── test_inference.py
```

---

## Git Workflow

Este repositorio sigue una estrategia de branching profesional para MLOps:

- **`main`** — rama de producción, solo recibe cambios via PR desde `development`
- **`development`** — rama de integración, recibe cambios via PR desde feature branches
- **`feature/*`** — una rama por cada entregable, con commits usando [Conventional Commits](https://www.conventionalcommits.org/)

Flujo de trabajo:
```
feature/preprocessing → development → main
feature/training      → development → main
feature/inference     → development → main
feature/tests         → development → main
feature/docker        → development → main
feature/model-improvement → development → main
feature/readme        → development → main
```

---

## Instalación y Setup

### Requisitos
- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/anapparedesr/ModeloML-predsales.git
cd ModeloML-predsales

# 2. Instalar dependencias
uv sync

# 3. Descargar los datos desde Kaggle y colocarlos en data/raw/
# https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data
# Archivos necesarios:
#   data/raw/sales_train.csv
#   data/raw/items.csv
#   data/raw/test.csv
```

---

## Ejecución del Pipeline Completo

### Con Python directamente

```bash
# Paso 1 — Preprocessing
uv run python -m src.preprocessing \
    --raw-dir data/raw \
    --prep-dir data/prep \
    --inference-dir data/inference

# Paso 2 — Training
uv run python -m src.training \
    --prep-dir data/prep \
    --artifacts-dir artifacts

# Paso 3 — Inference
uv run python -m src.inference \
    --inference-dir data/inference \
    --artifacts-dir artifacts \
    --predictions-dir data/predictions
```

### Con Docker

```bash
# Construir imágenes
docker build -t ml-preprocessing:latest ./src/preprocessing/
docker build -t ml-training:latest ./src/training/
docker build -t ml-inference:latest ./src/inference/
```

---

## Ejecución de Contenedores

Cada contenedor acepta argumentos de entrada, salida e hiperparámetros por CLI:

```bash
# Preprocessing
docker run \
    -v $(pwd)/data:/app/data \
    ml-preprocessing:latest \
    --raw-dir data/raw \
    --prep-dir data/prep \
    --inference-dir data/inference

# Training (con hiperparámetros personalizados)
docker run \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/artifacts:/app/artifacts \
    ml-training:latest \
    --prep-dir data/prep \
    --artifacts-dir artifacts \
    --n-estimators 200 \
    --max-depth 8

# Inference
docker run \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/artifacts:/app/artifacts \
    ml-inference:latest \
    --inference-dir data/inference \
    --artifacts-dir artifacts \
    --predictions-dir data/predictions
```

---

## Mejora del Modelo

Se implementó **RandomizedSearchCV** para optimizar los hiperparámetros del Random Forest antes del entrenamiento final.

**Espacio de búsqueda:**

| Hiperparámetro | Valores explorados |
|---|---|
| `n_estimators` | 50, 100, 200, 300 |
| `max_depth` | 5, 8, 10, 12, 15, None |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |
| `max_features` | sqrt, log2 |

**Configuración:** 20 iteraciones, 3-fold cross-validation, métrica: RMSE.

El modelo con hiperparámetros optimizados se compara contra el baseline en el set de validación (mes 33). Los resultados se registran en los logs de ejecución en `artifacts/logs/`.

---

## Pruebas Unitarias

El proyecto incluye 23 pruebas unitarias organizadas por step:

```bash
uv run pytest src/ -v
```

**Resultado esperado:**
```
23 passed in 5.23s

**Captura de pantalla**
![pruebaspytest](https://github.com/user-attachments/assets/6e36c633-3e09-484c-920e-aef9a9fa887c)


```

| Step | Archivo | Tests |
|---|---|---|
| Preprocessing | `src/preprocessing/test/test_prep.py` | 9 |
| Training | `src/training/test/test_train.py` | 7 |
| Inference | `src/inference/test/test_inference.py` | 7 |
