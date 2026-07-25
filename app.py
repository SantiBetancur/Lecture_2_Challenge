"""
app.py
------
Aplicación Streamlit - Sistema de Soporte a la Decisión (DSS) para
TechLogistics S.A.S. Integra Transacciones Logísticas, Inventario Central
y Feedback de Clientes en una Sola Fuente de Verdad, responde las 5
preguntas de alta gerencia del Challenge 02 y genera recomendaciones
estratégicas con IA (Groq / Llama 3.3).

Autor: Santiago Betancur
Challenge 02 - Fundamentos en Ciencia de Datos (Maestría) - EAFIT 2026-1
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Agregar ruta de processing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSING_DIR = os.path.join(SCRIPT_DIR, "processing")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")
sys.path.insert(0, PROCESSING_DIR)

# Importar funciones de processing
from lg_transactions import procesar_transacciones
from inventory import procesar_inventario
from feedback import procesar_feedback
from integracion import construir_fuente_unica, SLA_ENTREGA_DIAS


# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="TechLogistics - Data Hub & DSS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNCIONES AUXILIARES - CARGA Y MÉTRICAS
# ============================================================================

@st.cache_data(show_spinner=False)
def cargar_datos():
    """Ejecuta los 3 pipelines de curación y construye la Fuente Única de Verdad."""
    try:
        with st.spinner("⏳ Cargando, curando e integrando los datasets..."):
            df_transacciones = procesar_transacciones()
            df_inventario = procesar_inventario()
            df_feedback = procesar_feedback()
            df_maestro = construir_fuente_unica()

        return {
            "transacciones": df_transacciones,
            "inventario": df_inventario,
            "feedback": df_feedback,
            "maestro": df_maestro,
        }
    except Exception as e:
        st.error(f"❌ Error al cargar o integrar los datos: {e}")
        return None


def calcular_metricas(datasets):
    """Calcula métricas generales de los datasets."""
    metricas = {
        "transacciones": {
            "registros": len(datasets["transacciones"]),
            "columnas": len(datasets["transacciones"].columns),
        },
        "inventario": {
            "registros": len(datasets["inventario"]),
            "columnas": len(datasets["inventario"].columns),
        },
        "feedback": {
            "registros": len(datasets["feedback"]),
            "columnas": len(datasets["feedback"].columns),
        },
    }
    return metricas


def obtener_columnas_numericas(df):
    """Retorna columnas numéricas de un DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def obtener_columnas_categoricas(df):
    """Retorna columnas categóricas de un DataFrame."""
    return df.select_dtypes(include=["object"]).columns.tolist()


def leer_health_score(nombre_dataset):
    """Lee el comparativo de Health Score ya exportado por processing/*.py."""
    ruta = os.path.join(REPORTS_DIR, f"health_score_{nombre_dataset}.csv")
    if os.path.exists(ruta):
        return pd.read_csv(ruta)
    return None


def construir_reporte_limpieza_consolidado():
    """Concatena los 3 logs de limpieza en un único CSV descargable."""
    patrones = {
        "transacciones": "log_limpieza_transacciones.csv",
        "inventario": "log_limpieza_inventario.csv",
        "feedback": "log_limpieza_feedback.csv",
    }
    partes = []
    for dataset, archivo in patrones.items():
        ruta = os.path.join(REPORTS_DIR, archivo)
        if os.path.exists(ruta):
            df_log = pd.read_csv(ruta)
            df_log.insert(0, "dataset", dataset)
            partes.append(df_log)
    if not partes:
        return None
    return pd.concat(partes, ignore_index=True)


def construir_health_score_consolidado():
    """Concatena los 3 comparativos de Health Score en un único CSV descargable."""
    partes = []
    for dataset in ["transacciones", "inventario", "feedback"]:
        df_health = leer_health_score(dataset)
        if df_health is not None:
            df_health.insert(0, "dataset", dataset)
            partes.append(df_health)
    if not partes:
        return None
    return pd.concat(partes, ignore_index=True)


# ============================================================================
# FILTROS GLOBALES SOBRE LA FUENTE ÚNICA DE VERDAD
# ============================================================================

def aplicar_filtros_maestro(df):
    """Renderiza los filtros globales en el sidebar y retorna el df filtrado."""
    st.sidebar.markdown("### 🔍 Filtros - Fuente Única de Verdad")

    categorias = sorted(df["Categoria"].dropna().unique().tolist())
    bodegas = sorted(df["Bodega_Origen"].dropna().unique().tolist())
    ciudades = sorted(df["Ciudad_Destino"].dropna().unique().tolist())
    canales = sorted(df["Canal_Venta"].dropna().unique().tolist())

    f_categorias = st.sidebar.multiselect("Categoría:", categorias, default=categorias)
    f_bodegas = st.sidebar.multiselect("Bodega de origen:", bodegas, default=bodegas)
    f_ciudades = st.sidebar.multiselect("Ciudad destino:", ciudades, default=ciudades)
    f_canales = st.sidebar.multiselect("Canal de venta:", canales, default=canales)
    incluir_sku_fantasma = st.sidebar.checkbox(
        "Incluir ventas de SKU fantasma (sin inventario)", value=True
    )

    df_filtrado = df.copy()
    # Los SKU fantasma no tienen Categoria/Bodega; se preservan si el
    # usuario los incluye explícitamente, para no ocultar el fenómeno.
    mask_categoria = df_filtrado["Categoria"].isin(f_categorias) | (
        incluir_sku_fantasma & df_filtrado["SKU_Fantasma"]
    )
    mask_bodega = df_filtrado["Bodega_Origen"].isin(f_bodegas) | (
        incluir_sku_fantasma & df_filtrado["SKU_Fantasma"]
    )
    df_filtrado = df_filtrado[
        mask_categoria
        & mask_bodega
        & df_filtrado["Ciudad_Destino"].isin(f_ciudades)
        & df_filtrado["Canal_Venta"].isin(f_canales)
    ]
    if not incluir_sku_fantasma:
        df_filtrado = df_filtrado[~df_filtrado["SKU_Fantasma"]]

    st.sidebar.caption(f"Filas tras filtrar: {len(df_filtrado):,} / {len(df):,}")
    return df_filtrado


# ============================================================================
# MÓDULO DE IA (GROQ / LLAMA 3.3)
# ============================================================================

def generar_resumen_estadistico(df):
    """Arma el resumen estadístico del subconjunto filtrado para el prompt de IA."""
    ingreso_total = df["Ingreso_Bruto"].sum()
    ingreso_riesgo = df["Ingreso_En_Riesgo"].sum()
    pct_riesgo = (ingreso_riesgo / ingreso_total * 100) if ingreso_total else 0
    margen_pct_prom = df["Margen_Utilidad_Pct"].mean()
    n_margen_negativo = int((df["Margen_Utilidad"] < 0).sum())
    entrega_prom = df["Tiempo_Entrega_Real"].mean()
    pct_tardias = df["Entrega_Tardia"].mean() * 100
    nps_prom = df["Satisfaccion_NPS"].mean()
    pct_sku_fantasma = df["SKU_Fantasma"].mean() * 100

    top_categorias_margen = (
        df.dropna(subset=["Categoria"])
        .groupby("Categoria")["Margen_Utilidad_Pct"]
        .mean()
        .sort_values()
        .head(3)
    )
    top_ciudades_entrega = (
        df.groupby("Ciudad_Destino")["Tiempo_Entrega_Real"].mean().sort_values(ascending=False).head(3)
    )

    resumen = f"""
RESUMEN ESTADÍSTICO - TECHLOGISTICS S.A.S. (datos filtrados por el usuario)

Volumen: {len(df):,} transacciones analizadas.
Ingreso bruto total: USD {ingreso_total:,.2f}
Ingreso en riesgo (SKU fantasma, sin respaldo de inventario): USD {ingreso_riesgo:,.2f} ({pct_riesgo:.1f}% del ingreso)
SKUs sin registro en inventario: {pct_sku_fantasma:.1f}% de las ventas filtradas.

Rentabilidad: margen promedio {margen_pct_prom:.1f}% sobre precio de venta.
Transacciones con margen negativo: {n_margen_negativo:,}.
Categorías con menor margen promedio:
{top_categorias_margen.to_string()}

Logística: tiempo de entrega promedio {entrega_prom:.1f} días (SLA {SLA_ENTREGA_DIAS} días).
Porcentaje de entregas tardías: {pct_tardias:.1f}%.
Ciudades con mayor tiempo de entrega promedio:
{top_ciudades_entrega.to_string()}

Experiencia de cliente: NPS promedio {nps_prom:.1f} (escala -100 a 100).
"""
    return resumen.strip()


def generar_recomendaciones_ia(resumen_estadistico, api_key, temperature=0.5):
    """Llama a Groq (Llama 3.3 70B) pidiendo 3 párrafos de recomendación estratégica."""
    from groq import Groq

    client = Groq(api_key=api_key)
    prompt = f"""Eres un consultor senior de datos presentando hallazgos a la junta
directiva de TechLogistics S.A.S., un retailer tecnológico que sufre erosión
de margen y caída de lealtad de clientes por invisibilidad operativa entre
sus sistemas de Inventario, Logística y Feedback.

{resumen_estadistico}

Con base ÚNICAMENTE en estas cifras, escribe EXACTAMENTE 3 párrafos de
recomendación estratégica dirigidos a la junta directiva:
1. Diagnóstico: qué revelan estas cifras sobre el estado del negocio.
2. Riesgo prioritario: cuál es el problema más urgente a resolver y por qué.
3. Acción recomendada: qué debería hacer la junta en los próximos 90 días.

No inventes cifras que no estén en el resumen. Sé directo y ejecutivo."""

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Eres un consultor senior de ciencia de datos especializado "
                "en retail y logística. Respondes de forma ejecutiva y basada en evidencia.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=1200,
        stream=True,
    )
    return stream


# ============================================================================
# CARGAR DATOS
# ============================================================================

datasets = cargar_datos()

if datasets is None:
    st.stop()

df_maestro_completo = datasets["maestro"]

# ============================================================================
# SIDEBAR - NAVEGACIÓN
# ============================================================================

st.sidebar.header("🔍 Navegación")
pagina = st.sidebar.radio(
    "Selecciona una vista:",
    [
        "📈 Dashboard Principal",
        "📌 Preguntas Estratégicas",
        "🤖 Recomendaciones IA",
        "📦 Transacciones",
        "🏭 Inventario",
        "💬 Feedback",
        "🔗 Comparativa",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### 📊 Datasets Cargados")
metricas = calcular_metricas(datasets)
for dataset, metrica in metricas.items():
    st.sidebar.metric(
        f"{dataset.upper()}",
        f"{metrica['registros']:,} registros",
        f"{metrica['columnas']} columnas"
    )

st.sidebar.divider()
st.sidebar.markdown("### 🔑 Configuración IA (Groq)")
groq_api_key = st.sidebar.text_input(
    "Groq API Key:", type="password", help="https://console.groq.com/"
)

# Filtros globales: solo se muestran (y aplican) en las páginas que
# consumen la Fuente Única de Verdad.
paginas_con_filtro = {"📌 Preguntas Estratégicas", "🤖 Recomendaciones IA"}
if pagina in paginas_con_filtro:
    st.sidebar.divider()
    df_maestro = aplicar_filtros_maestro(df_maestro_completo)
else:
    df_maestro = df_maestro_completo


# ============================================================================
# PÁGINA 1: DASHBOARD PRINCIPAL
# ============================================================================

if pagina == "📈 Dashboard Principal":
    st.title("📊 TechLogistics - Data Hub & Sistema de Soporte a la Decisión")
    st.markdown("""
    **Dashboard integrado para análisis de:**
    - 📦 Transacciones logísticas · 🏭 Inventario central · 💬 Feedback de clientes
    - 🔗 Fuente Única de Verdad (merge estratégico) · 📌 5 preguntas de alta gerencia · 🤖 IA
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Transacciones", f"{metricas['transacciones']['registros']:,}")
    with col2:
        st.metric("🏭 Inventario", f"{metricas['inventario']['registros']:,}", "productos")
    with col3:
        st.metric("💬 Feedback", f"{metricas['feedback']['registros']:,}", "opiniones")
    with col4:
        pct_fantasma = df_maestro_completo["SKU_Fantasma"].mean() * 100
        st.metric("👻 SKU Fantasma", f"{pct_fantasma:.1f}%", "del total de ventas")

    st.divider()

    # -------------------------------------------------------------
    # Auditoría de Calidad y Transparencia (Fase 1 del Challenge)
    # -------------------------------------------------------------
    st.subheader("🩺 Auditoría de Calidad - Health Score Antes vs. Después")
    st.caption(
        "Cada pipeline (`processing/*.py`) documenta su propio Health Score "
        "compuesto (Completitud 40% + Unicidad 25% + Consistencia 20% + Validez 15%)."
    )

    tab_tx, tab_inv, tab_fb = st.tabs(["📦 Transacciones", "🏭 Inventario", "💬 Feedback"])
    for tab, nombre in zip([tab_tx, tab_inv, tab_fb], ["transacciones", "inventario", "feedback"]):
        with tab:
            df_health = leer_health_score(nombre)
            if df_health is not None:
                fila_total = df_health[df_health["metrica"] == "health_score_total"].iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Antes", f"{fila_total['antes']:.1f}/100")
                c2.metric("Después", f"{fila_total['despues']:.1f}/100")
                c3.metric("Mejora", f"{fila_total['delta']:+.1f} pts")
                st.dataframe(df_health, use_container_width=True, hide_index=True)
            else:
                st.info("Ejecuta processing/*.py para generar el Health Score.")

    st.divider()
    st.subheader("📥 Reporte de Limpieza Descargable")
    col1, col2 = st.columns(2)
    with col1:
        df_log_consolidado = construir_reporte_limpieza_consolidado()
        if df_log_consolidado is not None:
            st.download_button(
                "💾 Descargar log de limpieza (los 3 datasets)",
                data=df_log_consolidado.to_csv(index=False).encode("utf-8"),
                file_name="log_limpieza_consolidado.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with col2:
        df_health_consolidado = construir_health_score_consolidado()
        if df_health_consolidado is not None:
            st.download_button(
                "💾 Descargar comparativo Health Score",
                data=df_health_consolidado.to_csv(index=False).encode("utf-8"),
                file_name="health_score_consolidado.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.divider()
    st.subheader("👁️ Vista Previa de Datasets")
    tab1, tab2, tab3, tab4 = st.tabs(["Transacciones", "Inventario", "Feedback", "Fuente Única de Verdad"])
    with tab1:
        st.dataframe(datasets["transacciones"].head(10), use_container_width=True)
    with tab2:
        st.dataframe(datasets["inventario"].head(10), use_container_width=True)
    with tab3:
        st.dataframe(datasets["feedback"].head(10), use_container_width=True)
    with tab4:
        st.dataframe(df_maestro_completo.head(10), use_container_width=True)


# ============================================================================
# PÁGINA 2: PREGUNTAS ESTRATÉGICAS (5 preguntas obligatorias del Challenge)
# ============================================================================

elif pagina == "📌 Preguntas Estratégicas":
    st.title("📌 Preguntas Estratégicas de Alta Gerencia")
    st.caption(
        f"Análisis sobre {len(df_maestro):,} de {len(df_maestro_completo):,} "
        "transacciones (según filtros del sidebar)."
    )

    if df_maestro.empty:
        st.warning("⚠️ No hay registros con los filtros actuales.")
        st.stop()

    tabs = st.tabs([
        "1️⃣ Fuga de Capital",
        "2️⃣ Crisis Logística",
        "3️⃣ Venta Invisible",
        "4️⃣ Diagnóstico de Fidelidad",
        "5️⃣ Riesgo Operativo",
    ])

    # --- Pregunta 1: Fuga de Capital y Rentabilidad ------------------------
    with tabs[0]:
        st.subheader("¿Qué SKUs se venden con margen negativo?")
        df_neg = df_maestro[df_maestro["Margen_Utilidad"] < 0]

        if df_neg.empty:
            st.success("✅ No hay transacciones con margen negativo en el subconjunto filtrado.")
        else:
            perdida_total = df_neg["Margen_Utilidad"].sum()
            pct_canal_online = (
                (df_neg["Canal_Venta"] == "Online").mean() * 100
                if "Online" in df_maestro["Canal_Venta"].unique()
                else 0
            )
            pct_online_base = (df_maestro["Canal_Venta"] == "Online").mean() * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("SKUs con margen negativo", f"{df_neg['SKU_ID'].nunique():,}")
            c2.metric("Pérdida acumulada", f"USD {perdida_total:,.2f}")
            c3.metric("% de esas ventas en canal Online", f"{pct_canal_online:.1f}%")

            col1, col2 = st.columns(2)
            with col1:
                por_canal = (
                    df_neg.groupby("Canal_Venta")["Margen_Utilidad"]
                    .sum()
                    .sort_values()
                    .reset_index()
                )
                fig = px.bar(
                    por_canal, x="Margen_Utilidad", y="Canal_Venta", orientation="h",
                    title="Pérdida acumulada por canal de venta",
                    labels={"Margen_Utilidad": "Margen acumulado (USD)"},
                    color="Margen_Utilidad", color_continuous_scale="Reds_r",
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                top_skus = (
                    df_neg.groupby("SKU_ID")["Margen_Utilidad"].sum()
                    .sort_values().head(15).reset_index()
                )
                fig2 = px.bar(
                    top_skus, x="Margen_Utilidad", y="SKU_ID", orientation="h",
                    title="Top 15 SKUs con mayor pérdida acumulada",
                    labels={"Margen_Utilidad": "Margen acumulado (USD)"},
                )
                st.plotly_chart(fig2, use_container_width=True)

            if pct_canal_online > pct_online_base * 1.3:
                veredicto = (
                    f"El canal **Online** concentra {pct_canal_online:.1f}% de las ventas con "
                    f"margen negativo, muy por encima de su participación general "
                    f"({pct_online_base:.1f}%): esto apunta a una **falla crítica de precios "
                    "en ese canal**, no a una pérdida aceptable por volumen."
                )
            else:
                veredicto = (
                    "Las pérdidas se distribuyen de forma similar entre canales "
                    f"(Online: {pct_canal_online:.1f}% vs. base {pct_online_base:.1f}%), "
                    "lo que sugiere que es más una **pérdida aceptable por volumen** en SKUs "
                    "puntuales que una falla sistemática de precios en un canal."
                )
            st.markdown(f"**Hallazgo:** {veredicto}")

    # --- Pregunta 2: Crisis Logística y Cuellos de Botella ------------------
    with tabs[1]:
        st.subheader("¿Dónde la correlación Tiempo de Entrega vs. NPS bajo es más fuerte?")
        df_fb = df_maestro[df_maestro["Tiene_Feedback"]].copy()
        df_fb["Zona"] = df_fb["Ciudad_Destino"] + " · " + df_fb["Bodega_Origen"]

        filas = []
        for zona, g in df_fb.groupby("Zona"):
            if len(g) >= 8 and g["Tiempo_Entrega_Real"].nunique() > 1:
                corr = g["Tiempo_Entrega_Real"].corr(g["Satisfaccion_NPS"])
                if pd.notna(corr):
                    filas.append({"Zona": zona, "Correlacion": corr, "n": len(g)})

        if not filas:
            st.info("No hay suficientes datos con feedback para calcular correlaciones por zona.")
        else:
            df_corr = pd.DataFrame(filas).sort_values("Correlacion")
            fig = px.bar(
                df_corr, x="Correlacion", y="Zona", orientation="h",
                title="Correlación Tiempo de Entrega vs. NPS por Ciudad · Bodega",
                color="Correlacion", color_continuous_scale="RdYlGn",
                hover_data=["n"],
            )
            st.plotly_chart(fig, use_container_width=True)

            peor = df_corr.iloc[0]
            st.markdown(
                f"**Hallazgo:** la zona **{peor['Zona']}** presenta la correlación más "
                f"negativa ({peor['Correlacion']:.2f}, n={peor['n']}): a mayor tiempo de "
                "entrega, más cae la satisfacción. Es la zona prioritaria para un "
                "**cambio inmediato de operador logístico**."
            )

            col1, col2 = st.columns(2)
            with col1:
                fig2 = px.scatter(
                    df_fb, x="Tiempo_Entrega_Real", y="Satisfaccion_NPS", color="Ciudad_Destino",
                    title="Tiempo de entrega vs. NPS individual", trendline="ols",
                    labels={"Tiempo_Entrega_Real": "Tiempo de entrega (días)", "Satisfaccion_NPS": "NPS"},
                )
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                por_bodega = df_fb.groupby("Bodega_Origen")["Entrega_Tardia"].mean().sort_values(ascending=False).reset_index()
                fig3 = px.bar(
                    por_bodega, x="Entrega_Tardia", y="Bodega_Origen", orientation="h",
                    title=f"% de entregas tardías (> SLA {SLA_ENTREGA_DIAS}d) por bodega",
                    labels={"Entrega_Tardia": "% tardías"},
                )
                fig3.update_xaxes(tickformat=".0%")
                st.plotly_chart(fig3, use_container_width=True)

    # --- Pregunta 3: Análisis de la Venta Invisible -------------------------
    with tabs[2]:
        st.subheader("¿Cuál es el impacto financiero de los SKU fantasma?")
        ingreso_total = df_maestro["Ingreso_Bruto"].sum()
        ingreso_riesgo = df_maestro["Ingreso_En_Riesgo"].sum()
        pct_riesgo = (ingreso_riesgo / ingreso_total * 100) if ingreso_total else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Ingreso total", f"USD {ingreso_total:,.0f}")
        c2.metric("Ingreso en riesgo (SKU fantasma)", f"USD {ingreso_riesgo:,.0f}")
        c3.metric("% del ingreso en riesgo", f"{pct_riesgo:.2f}%")

        col1, col2 = st.columns(2)
        with col1:
            por_mes = df_maestro.groupby("Mes_Venta")["Ingreso_En_Riesgo"].sum().reset_index()
            fig = px.line(
                por_mes, x="Mes_Venta", y="Ingreso_En_Riesgo", markers=True,
                title="Ingreso en riesgo por mes",
                labels={"Ingreso_En_Riesgo": "USD en riesgo"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            por_canal = df_maestro.groupby("Canal_Venta")["Ingreso_En_Riesgo"].sum().sort_values().reset_index()
            fig2 = px.bar(
                por_canal, x="Ingreso_En_Riesgo", y="Canal_Venta", orientation="h",
                title="Ingreso en riesgo por canal de venta",
                labels={"Ingreso_En_Riesgo": "USD en riesgo"},
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            f"**Hallazgo:** el **{pct_riesgo:.2f}%** del ingreso (USD {ingreso_riesgo:,.0f}) "
            "corresponde a ventas de SKUs sin respaldo en el maestro de inventario. "
            "Sin trazabilidad de costo para esas líneas, TechLogistics no puede saber "
            "si son rentables ni auditar su origen — es dinero que hoy opera sin control."
        )

    # --- Pregunta 4: Diagnóstico de Fidelidad -------------------------------
    with tabs[3]:
        st.subheader("¿Hay categorías con alto stock pero feedback negativo?")
        resumen_cat = (
            df_maestro.dropna(subset=["Categoria"])
            .groupby("Categoria")
            .agg(
                Ratio_Reorden_Prom=("Ratio_Stock_Reorden", "mean"),
                NPS_Promedio=("Satisfaccion_NPS", "mean"),
                Margen_Pct_Promedio=("Margen_Utilidad_Pct", "mean"),
                N=("Transaccion_ID", "count"),
            )
            .reset_index()
        )

        if resumen_cat.empty or resumen_cat["NPS_Promedio"].isna().all():
            st.info("No hay suficientes datos de feedback por categoría en el subconjunto filtrado.")
        else:
            fig = px.scatter(
                resumen_cat, x="Ratio_Reorden_Prom", y="NPS_Promedio", size="N",
                color="Categoria", text="Categoria",
                title="Disponibilidad de stock vs. Satisfacción por categoría",
                labels={"Ratio_Reorden_Prom": "Stock / Punto de reorden (promedio)", "NPS_Promedio": "NPS promedio"},
            )
            fig.update_traces(textposition="top center")
            mediana_stock = resumen_cat["Ratio_Reorden_Prom"].median()
            mediana_nps = resumen_cat["NPS_Promedio"].median()
            fig.add_vline(x=mediana_stock, line_dash="dash", line_color="gray")
            fig.add_hline(y=mediana_nps, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

            paradoja = resumen_cat[
                (resumen_cat["Ratio_Reorden_Prom"] >= mediana_stock)
                & (resumen_cat["NPS_Promedio"] < mediana_nps)
            ]
            if not paradoja.empty:
                nombres = ", ".join(paradoja["Categoria"].tolist())
                margen_paradoja = paradoja["Margen_Pct_Promedio"].mean()
                margen_resto = resumen_cat.loc[~resumen_cat["Categoria"].isin(paradoja["Categoria"]), "Margen_Pct_Promedio"].mean()
                if margen_paradoja > margen_resto:
                    explicacion = (
                        f"su margen promedio ({margen_paradoja:.1f}%) es más alto que el resto "
                        f"({margen_resto:.1f}%), lo que sugiere **sobrecosto** frente al valor percibido."
                    )
                else:
                    explicacion = (
                        f"su margen promedio ({margen_paradoja:.1f}%) no es mayor que el resto "
                        f"({margen_resto:.1f}%), lo que apunta más a **mala calidad de producto** "
                        "que a un problema de precio."
                    )
                st.markdown(
                    f"**Hallazgo:** {nombres} tienen alta disponibilidad de stock pero NPS "
                    f"por debajo de la mediana — {explicacion}"
                )
            else:
                st.success("✅ No se detecta la paradoja de alto stock + bajo NPS en este subconjunto.")

    # --- Pregunta 5: Storytelling de Riesgo Operativo -----------------------
    with tabs[4]:
        st.subheader("¿Qué bodegas operan a ciegas sobre su propio stock?")
        df_fb = df_maestro[df_maestro["Tiene_Feedback"]]

        antiguedad_bodega = df_maestro.groupby("Bodega_Origen")["Antiguedad_Revision_Dias"].mean()
        soporte_bodega = df_fb.groupby("Bodega_Origen")["Ticket_Soporte_Abierto"].mean()
        n_bodega = df_maestro.groupby("Bodega_Origen").size()

        resumen_bodega = pd.DataFrame({
            "Antiguedad_Revision_Prom": antiguedad_bodega,
            "Tasa_Ticket_Soporte": soporte_bodega,
            "N": n_bodega,
        }).dropna().reset_index()

        if resumen_bodega.empty:
            st.info("No hay suficientes datos cruzados de inventario y soporte por bodega.")
        else:
            fig = px.scatter(
                resumen_bodega, x="Antiguedad_Revision_Prom", y="Tasa_Ticket_Soporte",
                size="N", text="Bodega_Origen", color="Bodega_Origen",
                title="Antigüedad de la última revisión vs. tasa de tickets de soporte",
                labels={
                    "Antiguedad_Revision_Prom": "Antigüedad promedio de revisión (días)",
                    "Tasa_Ticket_Soporte": "Tasa de tickets de soporte",
                },
            )
            fig.update_traces(textposition="top center")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            peor = resumen_bodega.sort_values(
                ["Antiguedad_Revision_Prom", "Tasa_Ticket_Soporte"], ascending=False
            ).iloc[0]
            st.markdown(
                f"**Hallazgo:** la bodega **{peor['Bodega_Origen']}** combina la mayor "
                f"antigüedad de revisión ({peor['Antiguedad_Revision_Prom']:.0f} días) con una "
                f"alta tasa de tickets de soporte ({peor['Tasa_Ticket_Soporte']:.1%}): es la "
                "bodega que más 'opera a ciegas' y la que más está erosionando la "
                "satisfacción final del cliente."
            )


# ============================================================================
# PÁGINA 3: RECOMENDACIONES IA (Groq / Llama 3.3)
# ============================================================================

elif pagina == "🤖 Recomendaciones IA":
    st.title("🤖 Recomendaciones Estratégicas con IA")
    st.markdown(
        "Genera 3 párrafos de recomendación para la junta directiva a partir "
        "del resumen estadístico del subconjunto **filtrado** en el sidebar."
    )
    st.caption(f"Analizando {len(df_maestro):,} de {len(df_maestro_completo):,} transacciones.")

    if not groq_api_key:
        st.warning("⚠️ Ingresa tu **Groq API Key** en la barra lateral para usar esta función.")
        st.info("Consíguela gratis en https://console.groq.com/ → API Keys.")
    elif df_maestro.empty:
        st.warning("⚠️ No hay registros con los filtros actuales.")
    else:
        resumen = generar_resumen_estadistico(df_maestro)
        with st.expander("📄 Ver resumen estadístico enviado al modelo"):
            st.code(resumen, language="text")

        temperature = st.select_slider(
            "🌡️ Creatividad de la respuesta:", options=[0.2, 0.4, 0.5, 0.7, 0.9], value=0.5
        )

        if st.button("✨ Generar recomendaciones estratégicas", type="primary"):
            try:
                placeholder = st.empty()
                texto_completo = ""
                stream = generar_recomendaciones_ia(resumen, groq_api_key, temperature)
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        texto_completo += delta
                        placeholder.markdown(texto_completo + "▌")
                placeholder.markdown(texto_completo)
                st.session_state["ultima_recomendacion_ia"] = texto_completo
            except Exception as e:
                st.error(f"❌ Error al conectar con Groq: {e}")
                if "invalid" in str(e).lower() or "authentication" in str(e).lower():
                    st.warning("Verifica que tu API Key sea correcta.")
        elif "ultima_recomendacion_ia" in st.session_state:
            st.markdown(st.session_state["ultima_recomendacion_ia"])


# ============================================================================
# PÁGINA 4: TRANSACCIONES
# ============================================================================

elif pagina == "📦 Transacciones":
    st.title("📦 Análisis de Transacciones Logísticas")
    df = datasets["transacciones"]

    cols_num = obtener_columnas_numericas(df)
    cols_cat = obtener_columnas_categoricas(df)

    st.subheader("📊 Estadísticas Descriptivas")
    st.dataframe(df.describe(), use_container_width=True)

    if cols_num:
        st.subheader("📈 Distribuciones Numéricas")
        col1, col2 = st.columns(2)
        with col1:
            col_selected = st.selectbox("Selecciona columna numérica:", cols_num, key="trans_num")
            fig = px.histogram(df, x=col_selected, nbins=30, title=f"Distribución de {col_selected}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if len(cols_num) > 1:
                col_y = st.selectbox("Columna Y:", cols_num, index=1, key="trans_scatter_y")
                fig = px.scatter(df, x=col_selected, y=col_y, title=f"{col_selected} vs {col_y}")
                st.plotly_chart(fig, use_container_width=True)

    if cols_cat:
        st.subheader("🏷️ Análisis Categórico")
        col_cat = st.selectbox("Selecciona columna categórica:", cols_cat)
        fig = px.bar(
            df[col_cat].value_counts().reset_index(), x=col_cat, y="count",
            title=f"Conteo: {col_cat}"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Ver Datos Completos")
    st.dataframe(df, use_container_width=True)


# ============================================================================
# PÁGINA 5: INVENTARIO
# ============================================================================

elif pagina == "🏭 Inventario":
    st.title("🏭 Análisis de Inventario Central")
    df = datasets["inventario"]

    cols_num = obtener_columnas_numericas(df)
    cols_cat = obtener_columnas_categoricas(df)

    st.subheader("📊 Estadísticas Descriptivas")
    st.dataframe(df.describe(), use_container_width=True)

    if cols_num:
        st.subheader("📈 Distribuciones Numéricas")
        col1, col2 = st.columns(2)
        with col1:
            col_selected = st.selectbox("Selecciona columna numérica:", cols_num, key="inv_num")
            fig = px.histogram(df, x=col_selected, nbins=30, title=f"Distribución de {col_selected}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if len(cols_num) > 1:
                col_y = st.selectbox("Columna Y:", cols_num, index=1, key="inv_scatter_y")
                fig = px.scatter(df, x=col_selected, y=col_y, title=f"{col_selected} vs {col_y}")
                st.plotly_chart(fig, use_container_width=True)

    if cols_cat:
        st.subheader("🏷️ Análisis Categórico")
        col_cat = st.selectbox("Selecciona columna categórica:", cols_cat, key="inv_cat")
        fig = px.bar(
            df[col_cat].value_counts().reset_index(), x=col_cat, y="count",
            title=f"Conteo: {col_cat}"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Ver Datos Completos")
    st.dataframe(df, use_container_width=True)


# ============================================================================
# PÁGINA 6: FEEDBACK
# ============================================================================

elif pagina == "💬 Feedback":
    st.title("💬 Análisis de Feedback de Clientes")
    df = datasets["feedback"]

    cols_num = obtener_columnas_numericas(df)
    cols_cat = obtener_columnas_categoricas(df)

    st.subheader("📊 Estadísticas Descriptivas")
    st.dataframe(df.describe(), use_container_width=True)

    if cols_num:
        st.subheader("📈 Distribuciones Numéricas")
        col1, col2 = st.columns(2)
        with col1:
            col_selected = st.selectbox("Selecciona columna numérica:", cols_num, key="feed_num")
            fig = px.histogram(df, x=col_selected, nbins=30, title=f"Distribución de {col_selected}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if len(cols_num) > 1:
                col_y = st.selectbox("Columna Y:", cols_num, index=1, key="feed_scatter_y")
                fig = px.scatter(df, x=col_selected, y=col_y, title=f"{col_selected} vs {col_y}")
                st.plotly_chart(fig, use_container_width=True)

    if cols_cat:
        st.subheader("🏷️ Análisis Categórico")
        col_cat = st.selectbox("Selecciona columna categórica:", cols_cat, key="feed_cat")
        fig = px.bar(
            df[col_cat].value_counts().reset_index(), x=col_cat, y="count",
            title=f"Conteo: {col_cat}"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Ver Datos Completos")
    st.dataframe(df, use_container_width=True)


# ============================================================================
# PÁGINA 7: COMPARATIVA
# ============================================================================

elif pagina == "🔗 Comparativa":
    st.title("🔗 Análisis Comparativo de Datasets")

    st.subheader("📋 Dimensiones Generales")
    comparativa = pd.DataFrame({
        "Dataset": ["Transacciones", "Inventario", "Feedback"],
        "Registros": [len(datasets["transacciones"]), len(datasets["inventario"]), len(datasets["feedback"])],
        "Columnas": [len(datasets["transacciones"].columns), len(datasets["inventario"].columns), len(datasets["feedback"].columns)],
        "Duplicados": [
            int(datasets["transacciones"].duplicated().sum()),
            int(datasets["inventario"].duplicated().sum()),
            int(datasets["feedback"].duplicated().sum()),
        ],
    })
    st.dataframe(comparativa, use_container_width=True)

    st.subheader("📊 Comparativa Visual")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(comparativa, x="Dataset", y="Registros", title="Registros por Dataset", color="Dataset")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(comparativa, x="Dataset", y="Columnas", title="Columnas por Dataset", color="Dataset")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📝 Estructura de Columnas")
    tab1, tab2, tab3 = st.tabs(["Transacciones", "Inventario", "Feedback"])
    with tab1:
        st.write("**Columnas:**", ", ".join(datasets["transacciones"].columns))
        st.dataframe(datasets["transacciones"].dtypes, use_container_width=True)
    with tab2:
        st.write("**Columnas:**", ", ".join(datasets["inventario"].columns))
        st.dataframe(datasets["inventario"].dtypes, use_container_width=True)
    with tab3:
        st.write("**Columnas:**", ", ".join(datasets["feedback"].columns))
        st.dataframe(datasets["feedback"].dtypes, use_container_width=True)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #888;'>
    <small>
    📊 TechLogistics Data Hub v2.0 | Maestría Fundamentos en Ciencia de Datos - EAFIT 2026-1
    </small>
</div>
""", unsafe_allow_html=True)
