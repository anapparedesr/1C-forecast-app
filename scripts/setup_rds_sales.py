"""
Carga la tabla sales_monthly en RDS.
"""
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path

conn = psycopg2.connect(
    host="forecasting-1c-db.cw5yuskyg5vm.us-east-1.rds.amazonaws.com",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="forecasting1cc#"
)
cur = conn.cursor()

# Crear tabla
cur.execute("""
    CREATE TABLE IF NOT EXISTS sales_monthly (
        id SERIAL PRIMARY KEY,
        shop_id INTEGER,
        item_id INTEGER,
        date_block_num INTEGER,
        item_cnt_month FLOAT
    );
""")
conn.commit()

# Cargar datos
BASE = Path(__file__).parent.parent
sales = pd.read_csv(BASE / "data/raw/sales_train.csv")
sales_monthly = (
    sales.groupby(["shop_id", "item_id", "date_block_num"])["item_cnt_day"]
    .sum()
    .reset_index()
    .rename(columns={"item_cnt_day": "item_cnt_month"})
)

sales_rows = list(sales_monthly.itertuples(index=False, name=None))
execute_values(cur, """
    INSERT INTO sales_monthly (shop_id, item_id, date_block_num, item_cnt_month)
    VALUES %s
""", sales_rows)
conn.commit()
print(f"Sales monthly cargadas: {len(sales_rows):,} registros")

cur.close()
conn.close()