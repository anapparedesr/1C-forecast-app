# Proyecto para predecir ventas mensuales 

Este repositorio contiene un pipeline de machine learning para la predicción de ventas mensuales a nivel tienda-item, a partir de datos históricos de ventas. 
El objetivo principal del repositorio fue refactorizar un notebook que iba desde el análisis exploratorio hasta la creación de un modelo para predecir la demanda mensual y convertirlo en un repositorio estructurado con scripts que puedan ejecutarse de forma automática. 


## Estructura del repositorio 
```text
ModeloML-predsales/
├── data/
│   ├── raw/                # Datos originales
│   ├── prep/               # Datos preparados para modelado
│   ├── inference/          # Datos listos para inferencia
│   └── predictions/        # Predicciones creadas
│
├── artifacts/              # Modelos entrenados y otros objetos
│
├── notebooks/              # Notebooks (todo el proceso en los scripts)
│   └── Tarea1_FINAL.ipynb
│
├── src/                    # Scripts del pipeline
│   ├── __init__.py
│   ├── prep.py           
|   ├── prep_inference.py
│   ├── train.py
│   └── inference.py
│
├── .gitignore
└── README.md
```

## Datos originales 

Para replicar los resultados obtenidos, se sugiera que se descarguen los datos del siguiente [link](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales). 
Se necesita que se descarguen los datos de (https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales) 

Dentro del link se necesitan las siguientes bases de datos: 
- sales_train.csv
- test.csv
- items.csv

## Descripción de los scripts 

### prep.py

**Entrada**
* data/raw/sales_train.csv
* data/raw/items.csv

**Salida**
* data/prep/grid_model.csv

**Funcionalidad**
* Limpieza básica de datos
* Agregación mensual de vental
* Construcción del grid completo (tienda-item-mes)
* Creación de variables de rezago (`lag_1`, `lag_3`, `lag_6`, `lag_12`)

### prep_inference.py

Se recomienda utilizar este script si los datos para predecir no tienen las mismas columnas que la base: grid_model.csv 

**Entrada**
* data/raw/test.csv

**Salida**
* data/inference/test_with_lags.csv

**Funcionalidad**
* Crea rezagos a los datos para la predicción 
* Prepara los datos para la inferencia 

### train.py

**Entrada**
* data/prep/grid_model.csv

**Salida**
* artifacts/random_forest_lags.pkl

**Funcionalidad**
* Split temporal (train / validation)
* Entrenamiento de un RandomForestRegressor
* Evaluación con RMSE
* Persistencia del modelo entrenado 
 
### inference.py

**Entrada**
* data/inference/test_with_lags.csv
* artifacts/random_forest_lags.pkl

**Salida**
* data/predictions/predictions.csv

**Funcionalidad**
* Carga de datos para inferencia en batch
* Generación de predicciones con el modelo entrenado
* Guardado de resultados

### Ejecución de scripts 
Todos los scripts se ejecutan desde el root del repositorio, usando `uv`

```
uv run python src/prep.py
uv run python src/prep_inference.py
uv run python src/train.py
uv run python src/inference.py
```

Si se sigue correctamente los scripts y con los datos indicados, se obtendrá la siguiente organización
## Estructura del repositorio al correr los scripts y descargarlos datos 

```text
ModeloML-predsales/
├── data/
│   ├── raw/                # Datos originales
│   │   ├── sales_train.csv
│   │   ├── items.csv
│   │   └── test.csv
│   ├── prep/               # Datos preparados para modelado
│   │   └── grid_model.csv
│   ├── inference/          # Datos listos para inferencia
│   │   └── test_with_lags.csv
│   └── predictions/        # Predicciones en batch
│       └── predictions.csv
│
├── artifacts/              # Modelos entrenados y otros objetos
│   └── random_forest_lags.pkl
│
├── notebooks/              # Notebooks exploratorios
│   └── Tarea1_FINAL.ipynb
│
├── src/                    # Scripts del pipeline
│   ├── __init__.py
│   ├── prep.py
│   ├── train.py
│   └── inference.py
│
├── .gitignore
└── README.md
```

