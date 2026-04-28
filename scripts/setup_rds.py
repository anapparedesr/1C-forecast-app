"""
Crea las tablas en RDS y las puebla con los datos de predicciones y validación.
"""
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path

# ── Conexión ───────────────────────────────────────────────────────────────
conn = psycopg2.connect(
    host="forecasting-1c-db.cw5yuskyg5vm.us-east-1.rds.amazonaws.com",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="forecasting1cc#"
)
cur = conn.cursor()

# ── Crear tablas ───────────────────────────────────────────────────────────
cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        shop_id INTEGER,
        item_id INTEGER,
        item_category_id INTEGER,
        item_category_name VARCHAR(255),
        item_cnt_pred FLOAT
    );
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS validation (
        id SERIAL PRIMARY KEY,
        shop_id INTEGER,
        item_id INTEGER,
        item_cnt_actual FLOAT,
        item_cnt_pred FLOAT
    );
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id SERIAL PRIMARY KEY,
        shop_id INTEGER,
        item_id INTEGER,
        comment TEXT,
        username VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

conn.commit()
print("Tablas creadas")

# ── Cargar datos ───────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent

# Predictions
predictions  = pd.read_csv(BASE / "data/predictions/submissions.csv")
test         = pd.read_csv(BASE / "data/inference/test_with_lags.csv")
items        = pd.read_csv(BASE / "data/raw/items.csv")
categories   = pd.read_csv(BASE / "data/raw/item_categories.csv")

df = test[["ID", "shop_id", "item_id"]].merge(predictions, on="ID")
df = df.rename(columns={"item_cnt_month": "item_cnt_pred"})
df = df.merge(items[["item_id", "item_category_id"]], on="item_id", how="left")
df = df.merge(categories, on="item_category_id", how="left")

pred_rows = list(df[["shop_id", "item_id", "item_category_id", "item_category_name", "item_cnt_pred"]].itertuples(index=False, name=None))
execute_values(cur, """
    INSERT INTO predictions (shop_id, item_id, item_category_id, item_category_name, item_cnt_pred)
    VALUES %s
""", pred_rows)
conn.commit()
print(f"Predictions cargadas: {len(pred_rows):,} registros")

# Validation
val = pd.read_csv(BASE / "data/predictions/validation_preds.csv")
val_rows = list(val[["shop_id", "item_id", "item_cnt_month", "item_cnt_pred"]].itertuples(index=False, name=None))
execute_values(cur, """
    INSERT INTO validation (shop_id, item_id, item_cnt_actual, item_cnt_pred)
    VALUES %s
""", val_rows)
conn.commit()
print(f"Validation cargada: {len(val_rows):,} registros")

cur.close()
conn.close()
print("Conexión cerrada")