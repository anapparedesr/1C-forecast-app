"""
Genera predicciones para el mes 33 (validación) y las guarda en
data/predictions/validation_preds.csv para comparar contra ground truth.

Para poder hacer la gráfica de predicción vs ground truth
"""
import joblib
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent

# Cargar modelo
model = joblib.load(BASE / "artifacts/random_forest_lags.pkl")

# Cargar grid y filtrar mes 33
grid = pd.read_csv(BASE / "data/prep/grid_model.csv")
val = grid[grid["date_block_num"] == 33].copy()

# Features
FEATURES = ["lag_1", "lag_3", "lag_6", "lag_12"]
X_val = val[FEATURES]

# Predecir
val["item_cnt_pred"] = model.predict(X_val).clip(0, 20)

# Guardar
out = val[["shop_id", "item_id", "item_cnt_month", "item_cnt_pred"]]
out.to_csv(BASE / "data/predictions/validation_preds.csv", index=False)
print(f"Guardado: {len(out):,} registros")