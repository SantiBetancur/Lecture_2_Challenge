from pathlib import Path

import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

import mege


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@st.cache_data
def load_data():
    feedback_df, inventory_df, transactions_df = mege.load_datasets()
    merged_df = mege.merge_datasets(feedback_df, inventory_df, transactions_df)
    merged_df["Ultima_Revision"] = pd.to_datetime(merged_df["Ultima_Revision"], errors="coerce")
    merged_df["Antiguedad_Dias"] = (pd.Timestamp("today") - merged_df["Ultima_Revision"]).dt.days
    merged_df["Ticket_Soporte_Abierto"] = merged_df["Ticket_Soporte_Abierto"].astype(str).str.strip().str.lower()
    merged_df["Ticket_Soporte_Abierto"] = merged_df["Ticket_Soporte_Abierto"].replace({"sí": "sí", "si": "sí", "yes": "sí", "true": "sí"})
    merged_df["Ticket_Soporte_Abierto"] = merged_df["Ticket_Soporte_Abierto"].replace({"no": "no", "n": "no", "false": "no", "nan": "no"})
    return merged_df


st.set_page_config(page_title="DSS TechLogistics", page_icon="📊", layout="wide")
st.title("DSS TechLogistics S.A.S. - Dashboard de Recuperación")
st.caption("Análisis operativo, financiero y de fidelización para detectar riesgos y priorizar acciones")

merged_df = load_data()

st.sidebar.header("Controles")
st.sidebar.text_input("API Key de Groq", type="password", key="groq_api_key")
canal = st.sidebar.selectbox("Canal de venta", ["Todos", *sorted(merged_df["Canal_Venta"].dropna().astype(str).unique())])
ciudad = st.sidebar.selectbox("Ciudad", ["Todas", *sorted(merged_df["Ciudad_Destino"].dropna().astype(str).unique())])
bodega = st.sidebar.selectbox("Bodega", ["Todas", *sorted(merged_df["Bodega_Origen"].dropna().astype(str).unique())])

if canal != "Todos":
    merged_df = merged_df[merged_df["Canal_Venta"].astype(str) == canal]
if ciudad != "Todas":
    merged_df = merged_df[merged_df["Ciudad_Destino"].astype(str) == ciudad]
if bodega != "Todas":
    merged_df = merged_df[merged_df["Bodega_Origen"].astype(str) == bodega]

st.sidebar.download_button(
    label="Descargar reporte limpio",
    data=merged_df.to_csv(index=False).encode("utf-8"),
    file_name="reporte_dss_techlogistics.csv",
    mime="text/csv",
)

if merged_df.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

income_column = "Ingreso_Total_USD" if "Ingreso_Total_USD" in merged_df.columns else "Precio_Venta_Final"
ingreso_total = round(float(pd.to_numeric(merged_df[income_column], errors="coerce").fillna(0).sum()), 2)

metrics = {
    "Margen total USD": round(float(merged_df["Margen_Contribucion_USD"].sum()), 2),
    "Ingreso total USD": ingreso_total,
    "Promedio NPS": round(float(pd.to_numeric(merged_df["Satisfaccion_NPS"], errors="coerce").fillna(0).mean()), 2),
    "Tickets de soporte": int(merged_df["Ticket_Soporte_Abierto"].eq("sí").sum()),
}

cols = st.columns(4)
cols[0].metric("Margen total USD", f"{metrics['Margen total USD']:,.2f}")
cols[1].metric("Ingreso total USD", f"{metrics['Ingreso total USD']:,.2f}")
cols[2].metric("Promedio NPS", f"{metrics['Promedio NPS']:.2f}")
cols[3].metric("Tickets de soporte", f"{metrics['Tickets de soporte']}")

st.markdown("---")


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Fuga de capital",
    "2. Crisis logística",
    "3. Venta invisible",
    "4. Fidelidad",
    "5. Riesgo operativo",
])

with tab1:
    st.subheader("1. Fuga de Capital y Rentabilidad")
    col_a, col_b = st.columns(2)
    with col_a:
        margen_df = (
            merged_df.groupby(["SKU_ID", "Canal_Venta"], as_index=False)["Margen_Contribucion_USD"]
            .sum()
            .sort_values("Margen_Contribucion_USD")
        )
        fig1 = px.bar(
            margen_df.head(20),
            x="SKU_ID",
            y="Margen_Contribucion_USD",
            color="Canal_Venta",
            title="Top 20 SKUs por margen negativo o bajo",
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        loss_df = merged_df.groupby("Canal_Venta").agg(
            margen_total=("Margen_Contribucion_USD", "sum"),
            ventas=("Transaccion_ID", "nunique"),
        ).reset_index()
        fig2 = px.bar(
            loss_df,
            x="Canal_Venta",
            y="margen_total",
            color="Canal_Venta",
            title="Margen por canal",
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("2. Crisis Logística y Cuellos de Botella")
    col_c, col_d = st.columns(2)
    with col_c:
        corr_series = (
            merged_df.groupby(["Ciudad_Destino", "Bodega_Origen"])
            .apply(lambda x: x["Tiempo_Entrega_Real"].corr(x["Satisfaccion_NPS"]))
        )
        corr_df = corr_series.reset_index()
        corr_df = corr_df.rename(columns={0: "corr"})
        corr_df["corr_abs"] = corr_df["corr"].abs()
        fig3 = px.scatter(
            corr_df,
            x="corr",
            y="Ciudad_Destino",
            size="corr_abs",
            color="Bodega_Origen",
            title="Correlación tiempo de entrega vs. NPS por ciudad y bodega",
        )
        st.plotly_chart(fig3, use_container_width=True)
    with col_d:
        bubble_df = merged_df.groupby(["Ciudad_Destino", "Bodega_Origen"], as_index=False).agg(
            promedio_tiempo=("Tiempo_Entrega_Real", "mean"),
            nps_promedio=("Satisfaccion_NPS", "mean"),
            riesgo=("Riesgo_Operativo", "mean"),
        )
        fig4 = px.scatter(
            bubble_df,
            x="promedio_tiempo",
            y="nps_promedio",
            size="riesgo",
            color="Bodega_Origen",
            hover_name="Ciudad_Destino",
            title="Zona crítica: tiempo promedio vs. NPS",
        )
        st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.subheader("3. Análisis de la Venta Invisible")
    missing_inventory = merged_df[merged_df["SKU_ID"].isna() | merged_df["SKU_ID"].astype(str).str.contains("nan", na=False)]
    if missing_inventory.empty:
        st.info("No hay ventas con SKU faltante en el maestro de inventario en el filtro aplicado.")
    else:
        impact_usd = float(pd.to_numeric(missing_inventory["Precio_Venta_Final"], errors="coerce").fillna(0).sum())
        pct = (
            impact_usd / float(pd.to_numeric(merged_df["Precio_Venta_Final"], errors="coerce").fillna(0).sum()) * 100
        ) if merged_df["Precio_Venta_Final"].notna().any() else 0
        st.metric("Impacto financiero USD", f"{impact_usd:,.2f}")
        st.metric("Porcentaje del ingreso en riesgo", f"{pct:.2f}%")
        fig5 = px.histogram(
            missing_inventory,
            x="Ciudad_Destino",
            color="Canal_Venta",
            title="Ventas sin SKU en inventario por ciudad",
        )
        st.plotly_chart(fig5, use_container_width=True)

with tab4:
    st.subheader("4. Diagnóstico de Fidelidad")
    fidelity_plot_df = merged_df.copy()
    fidelity_plot_df["margen_abs"] = fidelity_plot_df["Margen_Contribucion_USD"].abs()
    fidelity_plot_df["size_scaled"] = (fidelity_plot_df["margen_abs"] / max(fidelity_plot_df["margen_abs"].max(), 1) * 30).clip(5, 40)
    fig6 = px.scatter(
        fidelity_plot_df,
        x="Stock_Actual",
        y="Satisfaccion_NPS",
        color="Categoria",
        size="size_scaled",
        hover_name="SKU_ID",
        title="Disponibilidad vs. sentimiento del cliente por categoría",
    )
    st.plotly_chart(fig6, use_container_width=True)

    fidelity_summary = (
        merged_df.groupby("Categoria", as_index=False)
        .agg(
            stock_promedio=("Stock_Actual", "mean"),
            nps_promedio=("Satisfaccion_NPS", "mean"),
            margen_promedio=("Margen_Contribucion_USD", "mean"),
        )
        .sort_values("nps_promedio")
    )

    fig7 = plt.figure(figsize=(8, 4))
    sns.barplot(data=fidelity_summary, x="Categoria", y="nps_promedio", palette="mako")
    plt.xticks(rotation=30)
    plt.title("NPS promedio por categoría")
    st.pyplot(fig7)

with tab5:
    st.subheader("5. Storytelling de Riesgo Operativo")
    risk_plot_df = (
        merged_df.groupby(["Bodega_Origen", "Categoria"], as_index=False)
        .agg(
            antiguedad_promedio=("Antiguedad_Dias", "mean"),
            tickets=("Ticket_Soporte_Abierto", lambda x: (x == "sí").sum()),
            nps_promedio=("Satisfaccion_NPS", "mean"),
        )
    )
    risk_plot_df["size_scaled"] = (
        (risk_plot_df["nps_promedio"].abs() / max(risk_plot_df["nps_promedio"].abs().max(), 1) * 30).clip(5, 40)
    )
    fig8 = px.scatter(
        risk_plot_df,
        x="antiguedad_promedio",
        y="tickets",
        size="size_scaled",
        color="Bodega_Origen",
        hover_name="Categoria",
        title="Antigüedad del stock vs. tickets de soporte",
    )
    st.plotly_chart(fig8, use_container_width=True)

st.markdown("---")
st.subheader("Resumen ejecutivo")
summary = pd.DataFrame(
    {
        "Pregunta": [
            "Margen negativo por SKU",
            "Riesgo logístico por ciudad/bodega",
            "Impacto de ventas sin inventario",
            "Paradoja stock alto vs. sentimiento negativo",
            "Antigüedad de stock y soporte",
        ],
        "Observación": [
            "Revisar SKUs con margen negativo en canales online y físico.",
            "Priorizar ciudades y bodegas con alta correlación entre retraso y NPS bajo.",
            "Medir la pérdida por ventas sin SKU controlado en inventario.",
            "Investigar si la causa es calidad o sobrecosto en categorías con stock alto y NPS bajo.",
            "Detectar bodegas operando con revisiones antiguas y alta carga de soporte.",
        ],
    }
)
st.dataframe(summary, use_container_width=True)
