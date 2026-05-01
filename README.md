# 1C Company - Producto de Datos para el pronóstico de las ventas 

Repositorio de la aplicación web de pronóstico de ventas desarrollada para 1C Company. 
Permite a los equipos de planeación, finanzas y operaciones consultar predicciones de ventas mensuales por tienda, categoría y producto, sin necesidad de acceder a AWS o ejecutar código.


**App con URL pública:** [http://forecast-alb-1188022330.us-east-1.elb.amazonaws.com](http://forecast-alb-1188022330.us-east-1.elb.amazonaws.com)

## Arquitectura de la solución

![Arquitectura](docs/Diagrama.png)

El sistema se divide en dos capas:

**Capa offline (local):** El pipeline de ML procesa los datos históricos de ventas, entrena un modelo Random Forest y genera pronósticos para el mes 34. Los resultados se cargan en RDS y S3 mediante scripts ETL.

**Capa productiva (AWS):** Una aplicación Streamlit desplegada en ECS Fargate lee los pronósticos desde RDS PostgreSQL y los presenta al usuario final a través de una URL pública. La infraestructura se despliega con CloudFormation.

---

## Modelo de datos

![ERD](docs/erd.png)

La base de datos RDS PostgreSQL contiene 7 tablas:

| Tabla | Descripción |
|---|---|
| `predictions` | Pronósticos del mes 34 por tienda y producto |
| `validation` | Predicciones del mes 33 vs. ground truth para evaluación |
| `sales_monthly` | Histórico mensual de ventas para visualización |
| `feedback` | Comentarios del negocio sobre predicciones problemáticas |
| `shops` | Catálogo de tiendas |
| `items` | Catálogo de productos |
| `item_categories` | Catálogo de categorías |

---

## Servicios de AWS

| Servicio | Uso |
|---|---|
| **Amazon S3** | Almacenamiento de datos crudos, modelo `.pkl` y predicciones CSV |
| **AWS Glue** | Catálogo de metadata sobre los CSV de predicciones en S3 |
| **Amazon Athena** | Queries ad-hoc sobre los datos en S3 para el equipo de BI |
| **Amazon RDS** | Base de datos PostgreSQL con tablas operacionales de la app |
| **AWS Secrets Manager** | Gestión segura de credenciales de RDS |
| **Amazon ECR** | Registro de la imagen Docker del Streamlit |
| **Amazon ECS Fargate** | Ejecución del contenedor de la app sin gestión de servidores |
| **AWS CloudFormation** | Despliegue de infraestructura como código |

---

## Vistas de la aplicación

| Tab | Descripción |
|---|---|
| Pronósticos | Filtro por tienda, categoría y producto con histórico de ventas |
| Batch Export | Descarga de pronósticos por tienda o catálogo completo |
| Feedback | Registro de productos con predicciones problemáticas |
| Evaluación | RMSE por categoría y tienda, predicción vs. ground truth |
| Catálogos | Referencia de tiendas, productos y categorías |

---

## Cómo correr el pipeline local

```bash
# 1. Clonar el repo
git clone https://github.com/anapparedesr/1C-forecast-app.git
cd 1C-forecast-app

# 2. Instalar dependencias
uv sync

# 3. Colocar los datos en data/raw/
# Descargar desde: https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data

# 4. Correr el pipeline
cd src/preprocessing && uv run python __main__.py --raw-dir ../../data/raw --prep-dir ../../data/prep --inference-dir ../../data/inference
cd ../training && uv run python __main__.py --prep-dir ../../data/prep --artifacts-dir ../../artifacts --no-random-search
cd ../inference && uv run python __main__.py --inference-dir ../../data/inference --artifacts-dir ../../artifacts --predictions-dir ../../data/predictions

# 5. Correr la app localmente
DB_USER=postgres DB_PASSWORD= DB_HOST= uv run streamlit run app/app.py
```
---


## Estructura del Repositorio

```
1C-forescast-app/
├── app/
│   ├── app.py                # Aplicación streamlit
│   ├── Dockerfile            # Imagen Docker
│   ├── requirements.txt      # Dependencias que se utilizan
├── data/
│   ├── raw/                  # Datos crudos 
│   ├── prep/                 # Datos preparados para entrenamiento
│   ├── inference/            # Datos preparados para inferencia
│   └── predictions/          # Predicciones generadas
├── scripts/               
│   └── setup_rds.py          # ETL: carga predictions y validations en RDS
│   └── setup_rds_sales.py    # ETL: carga sales_monthly en RDS
│   └── setup_rds_catalogs.py # ETL: carga catálogos en RDS
├── docs/ 
│   └── Diagrama.png          # Diagrama de flujo
│   └── ERD.png               # Diagrama entidad-relación                
└── src/
│   └── preprocessing/        # Pipeline de preprocesamiento
│   └── training/             # Pipeline de entrenamiento
│   └── inference/            # Pipeline de inferencia
└── ecs-fargate-app.yaml
└── Reporte del POC.pdf
└── README.md
```

---

## Costo operativo estimado

| Recurso | Costo mensual aproximado |
|---|---|
| RDS db.t3.micro | ~$12.40 |
| ECS Fargate (0.25 vCPU, 1GB) | ~$7.20 |
| Application Load Balancer | ~$5.76 |
| S3, ECR, Secrets Manager | ~$1.00 |
| **Total** | **~$26.36/mes** |

---

## Uso de herramientas de IA

Sí se utilizó IA durante el proceso de creación del producto de datos, en específico se utilizó Claude.ai como herramienta de consulta para debuggear los problemas de código, resolución de problemas en el despliegue de servicios de AWS y en el cálculo de costos aproximados. 

---

## Autora

Ana P. Paredes — ITAM MGE 2026

