import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from sqlalchemy import create_engine
import os

# ── Conexión a RDS ─────────────────────────────────────────────────────────
DB_URL = (
    f"postgresql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
    f"@{os.environ['DB_HOST']}:5432/postgres"
)

@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

@st.cache_data
def load_data():
    engine = get_engine()
    df        = pd.read_sql("SELECT * FROM predictions", engine)
    val_preds = pd.read_sql("SELECT * FROM validation", engine)
    sales     = pd.read_csv(Path(__file__).parent.parent / "data/raw/sales_train.csv")
    sales_monthly = (
        sales.groupby(["shop_id", "item_id", "date_block_num"])["item_cnt_day"]
        .sum()
        .reset_index()
        .rename(columns={"item_cnt_day": "item_cnt_month"})
    )
    return df, val_preds, sales_monthly

df, val_preds, sales_monthly = load_data()

# ── Configuración ──────────────────────────────────────────────────────────
st.set_page_config(page_title="1C Company — Pronóstico de Ventas", layout="wide")
st.title("Pronóstico de Ventas — 1C Company")

tab1, tab2, tab3, tab4 = st.tabs([
    "Pronósticos",
    "Batch Export",
    "Feedback",
    "Evaluación",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pronósticos
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Pronósticos por tienda y producto")

    col1, col2, col3 = st.columns(3)
    with col1:
        shop_ids = sorted(df["shop_id"].unique())
        selected_shop = st.selectbox("Tienda", shop_ids)

    df_shop = df[df["shop_id"] == selected_shop]

    with col2:
        categories_list = sorted(df_shop["item_category_name"].dropna().unique())
        selected_category = st.selectbox("Categoría", ["Todas"] + categories_list)

    if selected_category != "Todas":
        df_shop = df_shop[df_shop["item_category_name"] == selected_category]

    with col3:
        item_ids = sorted(df_shop["item_id"].unique())
        selected_item = st.selectbox("Producto", item_ids)

    # ── KPIs ──
    df_item_pred = df_shop[df_shop["item_id"] == selected_item]
    pred_value = df_item_pred["item_cnt_pred"].values[0] if len(df_item_pred) > 0 else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Predicción (Mes 34)", f"{pred_value:.2f} unidades")
    k2.metric("Tienda seleccionada", f"Shop {selected_shop}")
    k3.metric("Categoría", f"{selected_category}")
    st.divider()

    # ── Gráfica histórico + predicción ──
    st.subheader(f"Histórico de ventas — Producto {selected_item}")

    historico = sales_monthly[
        (sales_monthly["shop_id"] == selected_shop) &
        (sales_monthly["item_id"] == selected_item)
    ].sort_values("date_block_num")

    fig = go.Figure()

    if len(historico) > 0:
        fig.add_scatter(
            x=historico["date_block_num"],
            y=historico["item_cnt_month"],
            mode="lines+markers",
            name="Ventas reales",
            line=dict(color="#1f77b4")
        )

    fig.add_scatter(
        x=[34],
        y=[pred_value],
        mode="markers",
        name="Predicción Mes 34",
        marker=dict(color="#ff7f0e", size=12, symbol="star")
    )

    fig.update_layout(
        xaxis_title="Mes",
        yaxis_title="Unidades vendidas",
        yaxis=dict(rangemode="nonnegative"),
        legend=dict(orientation="h"),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("El histórico muestra únicamente los meses en que el producto registró ventas en esta tienda. Los meses sin datos indican que el producto no tuvo actividad comercial en ese período.")

    st.divider()

    # ── Tabla de pronósticos de la tienda ──
    st.subheader(f"Todos los pronósticos — Tienda {selected_shop}")
    st.dataframe(
        df_shop[["shop_id", "item_id", "item_category_name", "item_cnt_pred"]].reset_index(drop=True),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Export
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Exportar pronósticos")
    st.write("Selecciona un grupo y descarga el archivo de pronósticos.")

    export_option = st.radio(
        "¿Qué quieres exportar?",
        ["Una tienda completa", "Todo el catálogo"]
    )

    if export_option == "Una tienda completa":
        shop_export = st.selectbox("Tienda a exportar", sorted(df["shop_id"].unique()), key="export_shop")
        df_export = df[df["shop_id"] == shop_export][["shop_id", "item_id", "item_category_name", "item_cnt_pred"]]
    else:
        df_export = df[["shop_id", "item_id", "item_category_name", "item_cnt_pred"]]

    st.metric("Registros a exportar", f"{len(df_export):,}")

    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇Descargar CSV",
        data=csv,
        file_name="pronosticos_mes34.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Feedback
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Feedback del negocio")
    st.write("Registra productos cuyas predicciones no reflejan la realidad del negocio.")

    col1, col2 = st.columns(2)
    with col1:
        fb_shop = st.selectbox("Tienda", sorted(df["shop_id"].unique()), key="fb_shop")
    with col2:
        fb_item = st.selectbox("Producto", sorted(df[df["shop_id"] == fb_shop]["item_id"].unique()), key="fb_item")

    fb_username = st.text_input("Tu nombre", placeholder="Ej: Ana Paredes")
    fb_comment = st.text_area("¿Qué problema tiene esta predicción?", placeholder="Ej: El producto fue descontinuado, hay estacionalidad no capturada...")

    if st.button("Registrar feedback"):
        if fb_comment.strip() and fb_username.strip():
            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO feedback (shop_id, item_id, comment, username)
                        VALUES (:shop_id, :item_id, :comment, :username)
                    """),
                    {"shop_id": int(fb_shop), "item_id": int(fb_item),
                     "comment": fb_comment, "username": fb_username}
                )
                conn.commit()
            st.success(f"Feedback registrado para Tienda {fb_shop} — Producto {fb_item}")
        else:
            st.warning("Por favor completa tu nombre y el comentario antes de registrar.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Evaluación del modelo
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Evaluación del modelo — Mes 33")
    st.write("Comparación entre las ventas reales del mes 33 y las predicciones del modelo para ese mismo mes.")

    col1, col2 = st.columns(2)
    with col1:
        shop_eval = st.selectbox("Tienda", sorted(val_preds["shop_id"].unique()), key="eval_shop")

    df_eval = val_preds[val_preds["shop_id"] == shop_eval]

    with col2:
        item_eval = st.selectbox("Producto", sorted(df_eval["item_id"].unique()), key="eval_item")

    df_eval_item = df_eval[df_eval["item_id"] == item_eval]

    if len(df_eval_item) > 0:
        real  = df_eval_item["item_cnt_actual"].values[0]
        pred  = df_eval_item["item_cnt_pred"].values[0]
        error = abs(real - pred)

        k1, k2, k3 = st.columns(3)
        k1.metric("Ventas reales", f"{real:.2f}")
        k2.metric("Predicción", f"{pred:.2f}")
        k3.metric("Error absoluto", f"{error:.2f}")

        fig_eval = go.Figure()
        fig_eval.add_bar(name="Real", x=["Mes 33"], y=[real], marker_color="#1f77b4")
        fig_eval.add_bar(name="Predicción", x=["Mes 33"], y=[pred], marker_color="#ff7f0e")
        fig_eval.update_layout(
            barmode="group",
            yaxis=dict(rangemode="nonnegative"),
            height=400
        )
        st.plotly_chart(fig_eval, use_container_width=True)

    st.divider()

    st.subheader(f"Resumen por producto — Tienda {shop_eval}")
    df_eval_summary = df_eval.copy()
    df_eval_summary["error_absoluto"] = (df_eval_summary["item_cnt_actual"] - df_eval_summary["item_cnt_pred"]).abs()
    st.dataframe(
        df_eval_summary[["shop_id", "item_id", "item_cnt_actual", "item_cnt_pred", "error_absoluto"]].reset_index(drop=True),
        use_container_width=True
    )