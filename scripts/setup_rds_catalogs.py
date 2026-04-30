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

BASE = Path(__file__).parent.parent

# Shops
cur.execute("""
    CREATE TABLE IF NOT EXISTS shops (
        shop_id INTEGER PRIMARY KEY,
        shop_name VARCHAR(255)
    );
""")
conn.commit()

shops = pd.read_csv(BASE / "data/raw/shops.csv")
shop_rows = list(shops[["shop_id", "shop_name"]].itertuples(index=False, name=None))
execute_values(cur, "INSERT INTO shops (shop_id, shop_name) VALUES %s ON CONFLICT DO NOTHING", shop_rows)
conn.commit()
print(f"Shops cargados: {len(shop_rows)} registros")

# Items
cur.execute("""
    CREATE TABLE IF NOT EXISTS items (
        item_id INTEGER PRIMARY KEY,
        item_name VARCHAR(500),
        item_category_id INTEGER
    );
""")
conn.commit()

items = pd.read_csv(BASE / "data/raw/items.csv")
item_rows = list(items[["item_id", "item_name", "item_category_id"]].itertuples(index=False, name=None))
execute_values(cur, "INSERT INTO items (item_id, item_name, item_category_id) VALUES %s ON CONFLICT DO NOTHING", item_rows)
conn.commit()
print(f"Items cargados: {len(item_rows)} registros")

# Item categories
cur.execute("""
    CREATE TABLE IF NOT EXISTS item_categories (
        item_category_id INTEGER PRIMARY KEY,
        item_category_name VARCHAR(255)
    );
""")
conn.commit()

cats = pd.read_csv(BASE / "data/raw/item_categories.csv")
cat_rows = list(cats[["item_category_id", "item_category_name"]].itertuples(index=False, name=None))
execute_values(cur, "INSERT INTO item_categories (item_category_id, item_category_name) VALUES %s ON CONFLICT DO NOTHING", cat_rows)
conn.commit()
print(f"Categorías cargadas: {len(cat_rows)} registros")

cur.close()
conn.close()