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

import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from techlogistics.config import (
    HEALTH_SCORE_POR_DATASET,
    LOG_LIMPIEZA_POR_DATASET,
    RAW_CSV_POR_DATASET,
    REPORT_PDF,
    ROOT,
    SLA_ENTREGA_DIAS,
)
from techlogistics.pipelines import (
    construir_fuente_unica,
    procesar_feedback,
    procesar_inventario,
    procesar_transacciones,
)


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

    /* --- Sidebar brand & navigation --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 55%, #0f172a 100%);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.25rem;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em;
    }
    .nav-brand {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.25rem;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.35);
    }
    .nav-brand-title {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .nav-brand-sub {
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.78rem;
        margin-top: 0.15rem;
    }
    .nav-section {
        color: #94a3b8 !important;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 1rem 0 0.35rem 0;
        padding-left: 0.15rem;
    }
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        background: rgba(255, 255, 255, 0.04) !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        padding: 0.55rem 0.75rem !important;
        transition: all 0.15s ease;
        text-align: left !important;
        justify-content: flex-start !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(96, 165, 250, 0.55) !important;
        background: rgba(59, 130, 246, 0.12) !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
        border-color: transparent !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.45);
    }
    .nav-active-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #22c55e;
        margin-right: 6px;
        vertical-align: middle;
    }
    .sidebar-stats {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        margin-top: 0.5rem;
    }
    .sidebar-stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        font-size: 0.82rem;
        color: #cbd5e1;
    }
    .sidebar-stat-row:last-child { border-bottom: none; }
    .sidebar-stat-val {
        color: #f8fafc;
        font-weight: 600;
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
        with st.spinner("Cargando, curando e integrando los datasets..."):
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
    except FileNotFoundError as exc:
        st.error(f"No se encontró un archivo de datos requerido: {exc}")
        st.info("Verifique que `data/raw/` contenga los CSV del reto y ejecute `make pipeline`.")
    except (ValueError, PermissionError, pd.errors.ParserError) as exc:
        st.error(f"Error al procesar los datasets: {exc}")
    except Exception as exc:
        st.error(f"Error inesperado al cargar los datos: {exc}")
        with st.expander("Detalle técnico"):
            st.exception(exc)
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
    """Lee el comparativo de Health Score ya exportado por los pipelines."""
    ruta = HEALTH_SCORE_POR_DATASET.get(nombre_dataset)
    if ruta is not None and ruta.exists():
        return pd.read_csv(ruta)
    return None


@st.cache_data(show_spinner=False)
def resumen_nulidad_original(nombre_dataset):
    """% de nulidad por columna del CSV crudo, tal como exige la Fase 1 del reto."""
    ruta = RAW_CSV_POR_DATASET.get(nombre_dataset)
    if ruta is None or not ruta.exists():
        return None
    df_raw = pd.read_csv(ruta)
    nulidad = (df_raw.isna().sum() / len(df_raw) * 100).round(2)
    resumen = nulidad[nulidad > 0].sort_values(ascending=False).reset_index()
    resumen.columns = ["columna", "pct_nulos"]
    duplicados = int(df_raw.duplicated().sum())
    return resumen, duplicados, len(df_raw)


def construir_reporte_limpieza_consolidado():
    """Concatena los 3 logs de limpieza en un único CSV descargable."""
    partes = []
    for dataset, ruta in LOG_LIMPIEZA_POR_DATASET.items():
        if ruta.exists():
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


def _filas_log_por_tipo(nombre_dataset):
    """Lee el log de limpieza y clasifica eliminaciones, outliers e imputaciones."""
    ruta = LOG_LIMPIEZA_POR_DATASET.get(nombre_dataset)
    if ruta is None or not ruta.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_log = pd.read_csv(ruta)
    eliminados = df_log[
        df_log["accion"].str.contains("drop_duplicates", case=False, na=False)
    ]
    outliers = df_log[
        df_log["accion"].str.contains(
            r"IQR|clip|Winsor|Centinela|Fuera de|\> 100|negativos|999",
            case=False, na=False, regex=True,
        )
    ]
    imputados = df_log[
        df_log["accion"].str.contains(
            r"mediana|moda|NaN ->|Imputación|imputa",
            case=False, na=False, regex=True,
        )
        & ~df_log.index.isin(outliers.index)
    ]
    return eliminados, outliers, imputados


def construir_informe_trazabilidad_txt():
    """
    Informe TXT: Health Score, nulidad, duplicados, outliers y decisiones éticas.
    Incluye el contenido de docs/limpieza_datasets.md como anexo.
    """
    from datetime import datetime

    lineas = [
        "TECHLOGISTICS S.A.S. — INFORME DE TRAZABILIDAD DE LIMPIEZA DE DATOS",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "Challenge 02 — Fundamentos en Ciencia de Datos (EAFIT)",
        "",
        "Un consultor senior no limpia datos sin dejar rastro. Este informe documenta",
        "Health Score antes/después, métricas de calidad y decisiones éticas (eliminar vs imputar).",
        "",
        "=" * 72,
        "1. HEALTH SCORE ANTES Y DESPUÉS DEL PROCESAMIENTO",
        "=" * 72,
        "Fórmula compuesta: Completitud 40% + Unicidad 25% + Consistencia 20% + Validez 15%",
        "",
    ]

    for nombre in ["inventario", "transacciones", "feedback"]:
        lineas.append(f"\n--- {nombre.upper()} ---")
        df_health = leer_health_score(nombre)
        if df_health is not None:
            total = df_health[df_health["metrica"] == "health_score_total"].iloc[0]
            lineas.append(
                f"Health Score total: {total['antes']:.2f} → {total['despues']:.2f} "
                f"(Δ {total['delta']:+.2f} puntos)"
            )
            for _, row in df_health[df_health["metrica"].str.startswith("score_")].iterrows():
                dim = row["metrica"].replace("score_", "").capitalize()
                lineas.append(
                    f"  · {dim}: {row['antes']:.2f} → {row['despues']:.2f} "
                    f"(Δ {row['delta']:+.2f})"
                )
        else:
            lineas.append("  (Ejecute python scripts/run_pipeline.py para generar Health Score)")

        resultado = resumen_nulidad_original(nombre)
        if resultado is not None:
            resumen_nulos, duplicados_crudo, n_original = resultado
            lineas.append(f"\nMÉTRICAS DE CALIDAD — dataset original ({n_original:,} filas):")
            lineas.append(f"  · Duplicados exactos detectados: {duplicados_crudo:,}")
            if not resumen_nulos.empty:
                lineas.append("  · Porcentaje de nulidad por columna:")
                for _, r in resumen_nulos.iterrows():
                    lineas.append(f"      - {r['columna']}: {r['pct_nulos']:.2f}%")
            else:
                lineas.append("  · Sin columnas con nulos en el crudo.")

        eliminados, outliers, imputados = _filas_log_por_tipo(nombre)
        if not eliminados.empty:
            lineas.append("\nREGISTROS ELIMINADOS (decisión ética — solo duplicados reales):")
            for _, r in eliminados.iterrows():
                lineas.append(
                    f"  · {r['columna']}: {int(r['registros_afectados']):,} filas — {r['justificacion']}"
                )
        if not outliers.empty:
            lineas.append("\nOUTLIERS / VALORES EXTREMOS DETECTADOS Y TRATADOS:")
            for _, r in outliers.iterrows():
                lineas.append(
                    f"  · {r['columna']} | {r['accion']} | {int(r['registros_afectados']):,} filas"
                )
                lineas.append(f"    Justificación: {r['justificacion']}")
        if not imputados.empty:
            lineas.append("\nIMPUTACIONES (media / mediana / moda / categoría explícita):")
            for _, r in imputados.iterrows():
                lineas.append(
                    f"  · {r['columna']} | {r['accion']} | {int(r['registros_afectados']):,} filas"
                )
                lineas.append(f"    Justificación: {r['justificacion']}")

    lineas.extend([
        "",
        "=" * 72,
        "2. GUÍA DE DECISIÓN ÉTICA — CUÁNDO ELIMINAR, IMPUTAR O CONSERVAR",
        "=" * 72,
        "· ELIMINAR: solo duplicados exactos o por clave primaria (Feedback: 500 opiniones repetidas).",
        "  Inventario y transacciones: 0 filas eliminadas — perder filas ocultaría ingresos y SKU fantasma.",
        "· IMPUTAR con MEDIANA: variables numéricas con outliers o centinelas (-5, 999, ratings 45/99).",
        "  Robusta cuando la distribución es asimétrica o hay errores de captura extremos.",
        "· IMPUTAR con MODA: categorías corruptas en bajo volumen (<15 %), p.ej. Categoria ??? en inventario.",
        "· CATEGORÍA EXPLÍCITA (no moda): nulos masivos (>15–20 %) — Estado_Envio Sin_Informacion,",
        "  Recomienda_Marca Sin_Respuesta. Imputar moda inventaría estados u opiniones inexistentes.",
        "· WINSORIZACIÓN IQR: costos unitarios extremos por categoría — acota sin eliminar el SKU.",
        "· CONSERVAR + BANDERA: centinelas y ciudades inválidas se marcan (Registro_Confiable) para auditoría.",
        "",
        "=" * 72,
        "3. PRINCIPIOS TRANSVERSALES",
        "=" * 72,
        "· No eliminar filas salvo duplicados — extremos se winsorizan, imputan o categorizan.",
        "· Imputación contextual por categoría, ciudad o canal según la variable.",
        "· Trazabilidad: banderas *_Imputado y Registro_Confiable en datasets limpios.",
        "· SKU_ID normalizado (strip + upper) en inventario y transacciones para integración honesta.",
    ])

    doc_path = ROOT / "docs" / "limpieza_datasets.md"
    if doc_path.exists():
        lineas.extend([
            "",
            "=" * 72,
            "4. ANEXO — DOCUMENTACIÓN COMPLETA (docs/limpieza_datasets.md)",
            "=" * 72,
            "",
            doc_path.read_text(encoding="utf-8"),
        ])

    return "\n".join(lineas)


def render_trazabilidad_limpieza():
    """Sección del dashboard: transparencia, Health Score y decisiones éticas."""
    st.subheader("🧾 Trazabilidad de la limpieza — sin dejar rastro")
    st.markdown(
        "Un consultor senior **documenta cada decisión**: Health Score antes y después, "
        "**nulidad por columna**, **duplicados eliminados**, **outliers tratados** y "
        "criterio explícito de **eliminar vs imputar** (media, mediana o moda según la distribución)."
    )

    # Health Score comparativo
    filas_hs = []
    for nombre, etiqueta in [
        ("inventario", "Inventario"),
        ("transacciones", "Transacciones"),
        ("feedback", "Feedback"),
    ]:
        df_h = leer_health_score(nombre)
        if df_h is not None:
            t = df_h[df_h["metrica"] == "health_score_total"].iloc[0]
            filas_hs.append({
                "Dataset": etiqueta,
                "Antes": f"{t['antes']:.1f}",
                "Después": f"{t['despues']:.1f}",
                "Mejora (pts)": f"{t['delta']:+.1f}",
            })
    if filas_hs:
        st.markdown("#### Health Score por dataset")
        st.dataframe(pd.DataFrame(filas_hs), use_container_width=True, hide_index=True)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        with st.container(border=True):
            st.markdown("**¿Qué se eliminó?**")
            st.markdown(
                "- **Feedback:** 500 registros duplicados por `Feedback_ID` "
                "(opinión repetida inflaba el NPS).\n"
                "- **Inventario y transacciones:** **0 filas** eliminadas — "
                "borrar ventas ocultaría ingreso en riesgo y SKU fantasma."
            )
    with col_e2:
        with st.container(border=True):
            st.markdown("**¿Qué se imputó y por qué?**")
            st.markdown(
                "- **Mediana** (robusta a outliers): stock negativo, costo envío, "
                "tiempo entrega 999, ratings fuera de escala.\n"
                "- **Moda:** categorías corruptas (`???`) en bajo %.\n"
                "- **Categoría explícita** (no moda): 16.8 % `Estado_Envio` → "
                "`Sin_Informacion`; 25 % `Recomienda_Marca` → `Sin_Respuesta`."
            )

    with st.expander("📖 Resumen ejecutivo (basado en `docs/limpieza_datasets.md`)", expanded=True):
        st.markdown("""
**Metodología común en los 3 pipelines**

1. Auditoría inicial → Health Score (Completitud, Unicidad, Consistencia, Validez).
2. Limpieza con **log CSV** por transformación (`reports/quality/log_limpieza_*.csv`).
3. Feature engineering para el dashboard y las 5 preguntas de gerencia.
4. Auditoría final → comparativo antes/después.

**Outliers detectados y magnitud**

| Dataset | Tratamiento | Magnitud |
|---------|-------------|----------|
| Inventario | Winsorización IQR en `Costo_Unitario_USD` | 1 SKU acotado |
| Inventario | Stock negativo → mediana por categoría | 160 filas |
| Transacciones | Centinela −5 en cantidad → mediana | 100 filas |
| Transacciones | Centinela 999 en tiempo entrega → mediana por ciudad | 50 filas |
| Feedback | Ratings 45/99 fuera de [1,5] → mediana | 27 filas |
| Feedback | Edad > 100 → mediana | 20 filas |

**Decisión ética clave:** cuando el % de nulos es alto, **no se inventa** un valor modal —
se crea una categoría auditable (`Sin_Informacion`, `Sin_Respuesta`) porque la ausencia de dato
es en sí un hallazgo de negocio (invisibilidad operativa, desenganche del cliente).

**Integración:** merge `left` conserva ventas de SKU fantasma; se cuantifica `Ingreso_En_Riesgo`
sin ocultar el fenómeno. Ver anexo completo en la descarga TXT.
        """)

    informe_txt = construir_informe_trazabilidad_txt()
    st.download_button(
        "📄 Descargar informe de trazabilidad (.txt)",
        data=informe_txt.encode("utf-8"),
        file_name="informe_trazabilidad_limpieza_techlogistics.txt",
        mime="text/plain",
        use_container_width=True,
        help="Health Score, nulidad, duplicados, outliers, decisiones éticas y anexo limpieza_datasets.md",
    )


def render_sidebar_descargas():
    """Descargas rápidas desde el sidebar (reporte de limpieza y PDF)."""
    st.sidebar.markdown("### 📥 Descargas")
    df_log = construir_reporte_limpieza_consolidado()
    if df_log is not None:
        st.sidebar.download_button(
            "Log de limpieza (CSV)",
            data=df_log.to_csv(index=False).encode("utf-8"),
            file_name="log_limpieza_consolidado.csv",
            mime="text/csv",
            use_container_width=True,
            key="sidebar_log_limpieza",
        )
    else:
        st.sidebar.caption("Log de limpieza: ejecute `make pipeline`.")

    st.sidebar.download_button(
        "Trazabilidad ética (TXT)",
        data=construir_informe_trazabilidad_txt().encode("utf-8"),
        file_name="informe_trazabilidad_limpieza.txt",
        mime="text/plain",
        use_container_width=True,
        key="sidebar_trazabilidad_txt",
    )

    if REPORT_PDF.exists():
        st.sidebar.download_button(
            "Informe PDF junta directiva",
            data=REPORT_PDF.read_bytes(),
            file_name=REPORT_PDF.name,
            mime="application/pdf",
            use_container_width=True,
            key="sidebar_pdf_junta",
        )
    else:
        st.sidebar.caption("PDF: ejecute `make report`.")


def _banner_fuentes_cruzadas():
    st.caption(
        "Las conclusiones de esta página provienen solo de este dataset. "
        "El cruce con inventario y feedback está en **📌 Preguntas Estratégicas**."
    )


def _conclusion(titulo, dato, significado, accion=None):
    """Item de conclusión: hecho observable + interpretación + acción opcional."""
    return {"titulo": titulo, "dato": dato, "significado": significado, "accion": accion}


def _mostrar_conclusiones(titulo, conclusiones):
    """Panel de conclusiones legible para audiencia no técnica."""
    if not conclusiones:
        return
    st.divider()
    st.subheader(titulo)
    st.markdown(
        "Cada punto resume **qué muestran los gráficos**, **por qué debería importarte** "
        "y, cuando aplica, **qué conviene revisar a continuación**."
    )
    for item in conclusiones:
        if isinstance(item, str):
            with st.container(border=True):
                st.markdown(item)
            continue
        with st.container(border=True):
            st.markdown(f"#### {item['titulo']}")
            st.markdown(f"**Qué dicen los números:** {item['dato']}")
            st.markdown(f"**Por qué importa:** {item['significado']}")
            if item.get("accion"):
                st.markdown(f"**Qué revisar:** {item['accion']}")


def _multiselect_todos(label, opciones, key):
    """Multiselect con todas las opciones seleccionadas por defecto."""
    opciones = sorted({o for o in opciones if pd.notna(o)})
    if not opciones:
        return []
    return st.multiselect(label, opciones, default=opciones, key=key)


NAV_SECCIONES = [
    ("Visión general", ["📈 Dashboard Principal"]),
    ("Decisiones", ["📌 Preguntas Estratégicas", "🤖 Recomendaciones IA"]),
    ("Exploración de datos", ["📦 Transacciones", "🏭 Inventario", "💬 Feedback", "🔗 Comparativa"]),
]


def render_sidebar_navigation():
    """Navbar lateral con secciones, marca y estado activo."""
    if "pagina_activa" not in st.session_state:
        st.session_state.pagina_activa = "📈 Dashboard Principal"

    st.sidebar.markdown(
        """
        <div class="nav-brand">
            <div class="nav-brand-title">TechLogistics</div>
            <div class="nav-brand-sub">Data Hub · Soporte a la Decisión</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for titulo_seccion, paginas in NAV_SECCIONES:
        st.sidebar.markdown(f'<p class="nav-section">{titulo_seccion}</p>', unsafe_allow_html=True)
        for pagina in paginas:
            activa = st.session_state.pagina_activa == pagina
            etiqueta = f"● {pagina}" if activa else pagina
            if st.sidebar.button(
                etiqueta,
                key=f"nav_{pagina}",
                use_container_width=True,
                type="primary" if activa else "secondary",
            ):
                st.session_state.pagina_activa = pagina

    return st.session_state.pagina_activa


def render_sidebar_stats(metricas):
    """Resumen compacto de datasets en el sidebar."""
    st.sidebar.markdown("### 📊 Datos cargados")
    filas = [
        ("Transacciones", metricas["transacciones"]["registros"]),
        ("Inventario", metricas["inventario"]["registros"]),
        ("Feedback", metricas["feedback"]["registros"]),
    ]
    html = '<div class="sidebar-stats">'
    for nombre, n in filas:
        html += (
            f'<div class="sidebar-stat-row"><span>{nombre}</span>'
            f'<span class="sidebar-stat-val">{n:,}</span></div>'
        )
    html += "</div>"
    st.sidebar.markdown(html, unsafe_allow_html=True)


def _nota_grafico(texto):
    """Texto breve que conecta el gráfico con la respuesta ejecutiva."""
    st.caption(f"📎 **Evidencia:** {texto}")


def _grafico_comparacion_canales(df_base, df_sub, etiqueta_sub="Ventas en pérdida"):
    """Compara participación por canal: universo total vs subconjunto crítico."""
    canales = sorted(df_base["Canal_Venta"].dropna().unique())
    filas = []
    for canal in canales:
        filas.append({
            "Canal": canal,
            "Grupo": "Todas las ventas",
            "Participación": (df_base["Canal_Venta"] == canal).mean(),
        })
        filas.append({
            "Canal": canal,
            "Grupo": etiqueta_sub,
            "Participación": (df_sub["Canal_Venta"] == canal).mean() if len(df_sub) else 0,
        })
    df_plot = pd.DataFrame(filas)
    fig = px.bar(
        df_plot, x="Canal", y="Participación", color="Grupo", barmode="group",
        title="Participación por canal: ventas totales vs. ventas en pérdida",
        labels={"Participación": "Proporción", "Canal": "Canal de venta"},
        color_discrete_map={"Todas las ventas": "#94a3b8", etiqueta_sub: "#ef4444"},
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def _grafico_ingreso_riesgo_total(ingreso_seguro, ingreso_riesgo):
    """Donut del split ingreso con respaldo vs. en riesgo."""
    df_pie = pd.DataFrame({
        "Tipo": ["Con respaldo en inventario", "SKU fantasma (en riesgo)"],
        "USD": [max(ingreso_seguro, 0), max(ingreso_riesgo, 0)],
    })
    fig = px.pie(
        df_pie, names="Tipo", values="USD", hole=0.45,
        title="¿Qué parte del ingreso no tiene respaldo en bodega?",
        color="Tipo",
        color_discrete_map={
            "Con respaldo en inventario": "#22c55e",
            "SKU fantasma (en riesgo)": "#ef4444",
        },
    )
    return fig


def render_analisis_transacciones(df):
    """Visualizaciones de negocio para el dataset de transacciones."""
    _banner_fuentes_cruzadas()
    conclusiones = []

    pct_tardia = df["Entrega_Tardia"].mean() if "Entrega_Tardia" in df.columns else 0
    tiempo_med = df["Tiempo_Entrega_Real"].median() if "Tiempo_Entrega_Real" in df.columns else 0
    pct_sin_info = (
        (df["Estado_Envio"] == "Sin_Informacion").mean()
        if "Estado_Envio" in df.columns else 0
    )
    ingreso_total = df["Ingreso_Bruto"].sum() if "Ingreso_Bruto" in df.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ingreso bruto total", f"USD {ingreso_total:,.0f}",
              help="Suma de precio × cantidad antes de descuentos, impuestos o costos.")
    k2.metric("Entregas tardías", f"{pct_tardia:.1%}", f"SLA {SLA_ENTREGA_DIAS} días",
              help=f"Porcentaje de ventas que superaron {SLA_ENTREGA_DIAS} días desde la compra.")
    k3.metric("Tiempo entrega (mediana)", f"{tiempo_med:.0f} días",
              help="La mitad de los pedidos llegó en este tiempo o menos.")
    k4.metric("Estado sin registrar", f"{pct_sin_info:.1%}",
              help="Envíos sin información de si salieron, están en camino o se entregaron.")

    tab_log, tab_ing, tab_cal = st.tabs(
        ["🚚 Desempeño logístico", "💰 Ingresos y costo de flete", "🩺 Confiabilidad del registro"]
    )

    with tab_log:
        filt1, filt2 = st.columns(2)
        with filt1:
            if "Canal_Venta" in df.columns:
                canales_log = _multiselect_todos(
                    "Canal de venta (filtra los gráficos de abajo)",
                    df["Canal_Venta"].unique(),
                    key="tx_filtro_canal_log",
                )
                df_log = df[df["Canal_Venta"].isin(canales_log)] if canales_log else df.iloc[0:0]
            else:
                df_log = df
        with filt2:
            min_tx_ciudad = st.slider(
                "Mínimo de ventas por ciudad para mostrarla",
                min_value=1, max_value=30, value=5,
                help="Evita que ciudades con muy pocas ventas distorsionen el ranking.",
                key="tx_min_ciudad",
            )

        df_geo = df_log[df_log["Ciudad_Destino"].notna()].copy()
        if df_geo.empty:
            st.warning("No hay ciudades válidas con los filtros seleccionados.")
        else:
            por_ciudad = (
                df_geo.groupby("Ciudad_Destino", as_index=False)
                .agg(
                    Pct_Tardias=("Entrega_Tardia", "mean"),
                    Tiempo_Mediano=("Tiempo_Entrega_Real", "median"),
                    N=("Transaccion_ID", "count"),
                )
                .query("N >= @min_tx_ciudad")
                .sort_values("Pct_Tardias", ascending=False)
            )
            if por_ciudad.empty:
                st.info("Ninguna ciudad cumple el mínimo de ventas. Baja el umbral del filtro.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(
                        por_ciudad,
                        x="Pct_Tardias",
                        y="Ciudad_Destino",
                        orientation="h",
                        title=f"% entregas tardías por ciudad (SLA {SLA_ENTREGA_DIAS} días)",
                        labels={"Pct_Tardias": "% tardías", "Ciudad_Destino": "Ciudad"},
                        color="Pct_Tardias",
                        color_continuous_scale="Reds",
                        hover_data={"Tiempo_Mediano": ":.0f", "N": True},
                    )
                    fig.update_xaxes(tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig2 = px.bar(
                        por_ciudad.sort_values("Tiempo_Mediano", ascending=False),
                        x="Tiempo_Mediano",
                        y="Ciudad_Destino",
                        orientation="h",
                        title="Días típicos de entrega por ciudad (mediana)",
                        labels={"Tiempo_Mediano": "Días", "Ciudad_Destino": "Ciudad"},
                        color="Tiempo_Mediano",
                        color_continuous_scale="Oranges",
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                peor = por_ciudad.iloc[0]
                n_tardias_peor = int(peor["Pct_Tardias"] * peor["N"])
                conclusiones.append(_conclusion(
                    "Hay ciudades donde la entrega llega tarde con frecuencia",
                    f"En **{peor['Ciudad_Destino']}**, {peor['Pct_Tardias']:.1%} de las ventas "
                    f"({n_tardias_peor} de {int(peor['N'])}) superó el plazo de {SLA_ENTREGA_DIAS} días. "
                    f"El tiempo típico de entrega allí es de {peor['Tiempo_Mediano']:.0f} días.",
                    "Cuando muchos pedidos llegan tarde en una misma ciudad, el problema suele estar "
                    "en el operador o en la ruta, no en un cliente aislado.",
                    "Comparar proveedores logísticos en esa ciudad y revisar si hay cuellos de botella "
                    "en bodega de salida o última milla.",
                ))

        pct_tardia_log = df_log["Entrega_Tardia"].mean() if "Entrega_Tardia" in df_log.columns and len(df_log) else pct_tardia
        if pct_tardia_log > 0 and len(df_log):
            n_tardias = int(pct_tardia_log * len(df_log))
            conclusiones.append(_conclusion(
                "Una parte importante de las ventas incumple el plazo prometido",
                f"Con los filtros actuales, {pct_tardia_log:.1%} de las ventas "
                f"({n_tardias:,} de {len(df_log):,}) llegó después de {SLA_ENTREGA_DIAS} días.",
                "No se trata solo de un par de ciudades: el retraso aparece de forma repetida, "
                "lo que presiona la satisfacción del cliente y puede aumentar devoluciones o reclamos.",
                "Definir un plan de mejora logística por canal y medir cada mes si baja el % de tardías.",
            ))

        if "Estado_Envio" in df_log.columns and len(df_log):
            estado = df_log["Estado_Envio"].value_counts().reset_index()
            estado.columns = ["Estado_Envio", "Transacciones"]
            fig3 = px.bar(
                estado, x="Estado_Envio", y="Transacciones",
                title="¿En qué estado quedó registrado cada envío?",
                color="Estado_Envio",
            )
            fig3.update_layout(showlegend=False, xaxis_tickangle=-25)
            st.plotly_chart(fig3, use_container_width=True)
            pct_sin_info_log = (df_log["Estado_Envio"] == "Sin_Informacion").mean()
            if pct_sin_info_log > 0.05:
                n_sin = int(pct_sin_info_log * len(df_log))
                conclusiones.append(_conclusion(
                    "Muchos envíos no tienen estado registrado",
                    f"{pct_sin_info_log:.1%} de los envíos ({n_sin:,}) aparece como "
                    "'Sin información' en el sistema.",
                    "Sin saber si un paquete salió, está en tránsito o se entregó, el equipo "
                    "opera a ciegas y el cliente no recibe respuestas claras.",
                    "Exigir actualización de estado en cada hito logístico antes de escalar marketing.",
                ))

    with tab_ing:
        if "Canal_Venta" in df.columns and "Ingreso_Bruto" in df.columns:
            canales_ing = _multiselect_todos(
                "Canales a incluir en ingresos y flete",
                df["Canal_Venta"].unique(),
                key="tx_filtro_canal_ing",
            )
            df_ing = df[df["Canal_Venta"].isin(canales_ing)] if canales_ing else df.iloc[0:0]
            por_canal = (
                df_ing.groupby("Canal_Venta", as_index=False)
                .agg(
                    Ingreso_Bruto=("Ingreso_Bruto", "sum"),
                    Ratio_Envio_Med=("Ratio_Envio_Venta", "median"),
                    N=("Transaccion_ID", "count"),
                )
                .sort_values("Ingreso_Bruto", ascending=False)
            )
            if por_canal.empty:
                st.info("Selecciona al menos un canal de venta.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(
                        por_canal, x="Canal_Venta", y="Ingreso_Bruto",
                        title="Dinero total vendido por canal",
                        labels={"Ingreso_Bruto": "USD", "Canal_Venta": "Canal"},
                        color="Canal_Venta",
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig2 = px.bar(
                        por_canal.sort_values("Ratio_Envio_Med", ascending=False),
                        x="Canal_Venta", y="Ratio_Envio_Med",
                        title="Cuánto del valor de venta se va en envío (%, mediana por canal)",
                        labels={"Ratio_Envio_Med": "% del ingreso en flete", "Canal_Venta": "Canal"},
                        color="Ratio_Envio_Med",
                        color_continuous_scale="Reds",
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                canal_flete_row = por_canal.loc[por_canal["Ratio_Envio_Med"].idxmax()]
                canal_top = por_canal.iloc[0]
                ingreso_filtrado = por_canal["Ingreso_Bruto"].sum()
                conclusiones.append(_conclusion(
                    "El negocio depende de pocos canales de venta",
                    f"**{canal_top['Canal_Venta']}** concentra el mayor ingreso "
                    f"(USD {canal_top['Ingreso_Bruto']:,.0f}), "
                    f"es decir {canal_top['Ingreso_Bruto'] / ingreso_filtrado:.1%} "
                    f"del total filtrado ({int(canal_top['N']):,} ventas).",
                    "Si ese canal falla en precio, stock o entrega, el impacto en caja es inmediato.",
                    "Monitorear margen y tasa de entrega a tiempo específicamente en ese canal.",
                ))
                conclusiones.append(_conclusion(
                    "En algunos canales el envío se come una parte grande de la venta",
                    f"En **{canal_flete_row['Canal_Venta']}**, el envío se come en promedio "
                    f"el **{canal_flete_row['Ratio_Envio_Med']:.1f}%** del valor de cada venta "
                    f"(mediana del canal).",
                    "Un flete alto reduce el margen aunque las ventas crezcan; puede haber "
                    "precios demasiado bajos o rutas logísticas ineficientes.",
                    "Revisar política de envío gratis, zonas de entrega y si el precio cubre costo + flete.",
                ))
                if "Online" in por_canal["Canal_Venta"].values and ingreso_filtrado > 0:
                    ing_online = por_canal.loc[
                        por_canal["Canal_Venta"] == "Online", "Ingreso_Bruto"
                    ].iloc[0]
                    conclusiones.append(_conclusion(
                        "El canal Online pesa mucho en la facturación",
                        f"Online representa {ing_online / ingreso_filtrado:.1%} del ingreso "
                        f"en los canales seleccionados (USD {ing_online:,.0f}).",
                        "En digital, un error de precio o un SKU mal costeado se multiplica rápido "
                        "por el volumen de pedidos.",
                        "Cruzar ventas Online con costos de inventario en **Preguntas Estratégicas**.",
                    ))

        if "Mes_Venta" in df.columns and "Ingreso_Bruto" in df.columns and "Canal_Venta" in df.columns:
            canales_mes = _multiselect_todos(
                "Canales para la tendencia mensual",
                df["Canal_Venta"].unique(),
                key="tx_filtro_canal_mes",
            )
            df_mes = df[df["Canal_Venta"].isin(canales_mes)] if canales_mes else df.iloc[0:0]
            if not df_mes.empty:
                por_mes = df_mes.groupby("Mes_Venta", as_index=False)["Ingreso_Bruto"].sum()
                fig3 = px.line(
                    por_mes, x="Mes_Venta", y="Ingreso_Bruto", markers=True,
                    title="Cómo evolucionó el ingreso mes a mes (canales seleccionados)",
                    labels={"Ingreso_Bruto": "USD", "Mes_Venta": "Mes"},
                )
                fig3.update_xaxes(tickangle=-45)
                st.plotly_chart(fig3, use_container_width=True)

    with tab_cal:
        pct_confiable = df["Registro_Confiable"].mean() if "Registro_Confiable" in df.columns else 1
        pct_ciudad_inv = (
            df["Ciudad_Invalida_Origen"].mean()
            if "Ciudad_Invalida_Origen" in df.columns else 0
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros confiables", f"{pct_confiable:.1%}")
        c2.metric("Ciudad inválida (canal en destino)", f"{pct_ciudad_inv:.1%}")
        c3.metric("Total transacciones", f"{len(df):,}")

        banderas = []
        for col, label in [
            ("Cantidad_Sospechosa_Origen", "Cantidad centinela (-5)"),
            ("Costo_Envio_Imputado", "Costo envío imputado"),
            ("Tiempo_Entrega_Imputado", "Tiempo entrega imputado"),
            ("Ciudad_Invalida_Origen", "Ciudad inválida"),
        ]:
            if col in df.columns:
                banderas.append({"Alerta": label, "Registros": int(df[col].sum())})
        if banderas:
            st.dataframe(pd.DataFrame(banderas), use_container_width=True, hide_index=True)

        if pct_confiable < 1:
            n_conf = int(pct_confiable * len(df))
            conclusiones.append(_conclusion(
                "No todas las filas son igual de confiables",
                f"Solo {pct_confiable:.1%} de las transacciones ({n_conf:,}) quedó sin "
                "valores rellenados por limpieza de datos.",
                "Decisiones de dinero o SLA basadas en filas imputadas pueden estar "
                "suavizando problemas reales.",
                "Usar la columna `Registro_Confiable` cuando presentes cifras a gerencia.",
            ))

    conclusiones.append(_conclusion(
        "Aquí no vemos ventas de productos fuera de catálogo",
        "Este archivo de transacciones no trae el costo del producto ni si el SKU existe en inventario.",
        "Para saber cuánto dinero entra por artículos 'fantasma' hay que cruzar con el maestro de inventario.",
        "Ir a **📌 Preguntas Estratégicas** para ver ingreso en riesgo por SKU no catalogado.",
    ))
    _mostrar_conclusiones("📋 Qué nos dicen las transacciones", conclusiones)


def render_analisis_inventario(df):
    """Visualizaciones de negocio para el dataset de inventario."""
    _banner_fuentes_cruzadas()
    conclusiones = []

    valor_total = df["Valor_Inventario"].sum() if "Valor_Inventario" in df.columns else 0
    pct_alta_disp = df["Alta_Disponibilidad"].mean() if "Alta_Disponibilidad" in df.columns else 0
    pct_confiable = df["Registro_Confiable"].mean() if "Registro_Confiable" in df.columns else 1

    if "Antiguedad_Revision_Dias" in df.columns and "Bodega_Origen" in df.columns:
        bodega_ant = df.groupby("Bodega_Origen")["Antiguedad_Revision_Dias"].mean()
        peor_bodega = bodega_ant.idxmax()
        peor_dias = bodega_ant.max()
    else:
        peor_bodega, peor_dias = "—", 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Capital inmovilizado", f"USD {valor_total:,.0f}",
              help="Dinero atado en stock = unidades × costo unitario.")
    k2.metric("SKUs alta disponibilidad", f"{pct_alta_disp:.1%}", "stock ≥ 2× reorden",
              help="Productos con el doble o más del mínimo de stock deseado.")
    k3.metric("Registros confiables", f"{pct_confiable:.1%}",
              help="Filas sin valores rellenados artificialmente en la limpieza.")
    k4.metric("Bodega más desactualizada", peor_bodega, f"{peor_dias:.0f} días",
              help="Promedio de días desde la última revisión física del inventario.")

    tab_stock, tab_bodega, tab_costos = st.tabs(
        ["📦 Stock y disponibilidad", "🏭 Bodegas y revisión", "💵 Costos"]
    )

    with tab_stock:
        if "Categoria" in df.columns and "Stock_Actual" in df.columns:
            filt_bod, filt_cat = st.columns(2)
            with filt_bod:
                if "Bodega_Origen" in df.columns:
                    bodegas_sel = _multiselect_todos(
                        "Bodegas a analizar",
                        df["Bodega_Origen"].unique(),
                        key="inv_filtro_bodega_stock",
                    )
                    df_stock = df[df["Bodega_Origen"].isin(bodegas_sel)] if bodegas_sel else df.iloc[0:0]
                else:
                    df_stock = df
            with filt_cat:
                cats_sel = _multiselect_todos(
                    "Categorías a analizar",
                    df_stock["Categoria"].unique() if len(df_stock) else [],
                    key="inv_filtro_cat_stock",
                )
                if cats_sel:
                    df_stock = df_stock[df_stock["Categoria"].isin(cats_sel)]

            if df_stock.empty:
                st.info("Ajusta los filtros de bodega o categoría para ver el análisis.")
            else:
                stock_por_cat = (
                    df_stock.groupby("Categoria", as_index=False)
                    .agg(
                        Stock_Total=("Stock_Actual", "sum"),
                        Ratio_Reorden_Prom=("Ratio_Stock_Reorden", "mean"),
                        Valor_Total=("Valor_Inventario", "sum"),
                        Num_SKUs=("SKU_ID", "nunique"),
                    )
                    .sort_values("Stock_Total", ascending=False)
                )
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(
                        stock_por_cat, x="Categoria", y="Stock_Total",
                        color="Categoria",
                        title="Unidades guardadas en bodega por categoría",
                        labels={"Stock_Total": "Unidades", "Categoria": "Categoría"},
                        hover_data={"Ratio_Reorden_Prom": ":.2f", "Num_SKUs": True},
                    )
                    fig.update_layout(showlegend=False, xaxis_tickangle=-25)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig2 = px.scatter(
                        stock_por_cat, x="Ratio_Reorden_Prom", y="Valor_Total",
                        size="Num_SKUs", color="Categoria", text="Categoria",
                        title="¿Hay mucho stock respecto al mínimo? vs. dinero inmovilizado",
                        labels={
                            "Ratio_Reorden_Prom": "Veces el punto de reorden (promedio)",
                            "Valor_Total": "USD en inventario",
                        },
                    )
                    fig2.update_traces(textposition="top center")
                    fig2.add_vline(x=2, line_dash="dash", line_color="gray",
                                   annotation_text="Umbral 2× reorden")
                    st.plotly_chart(fig2, use_container_width=True)

                alta = stock_por_cat.loc[stock_por_cat["Ratio_Reorden_Prom"].idxmax()]
                pct_alta_filtrada = (
                    df_stock["Alta_Disponibilidad"].mean()
                    if "Alta_Disponibilidad" in df_stock.columns else pct_alta_disp
                )
                conclusiones.append(_conclusion(
                    "Hay categorías con mucho más stock del necesario",
                    f"**{alta['Categoria']}** tiene en promedio {alta['Ratio_Reorden_Prom']:.1f} veces "
                    f"el punto de reorden (mínimo deseado), con USD {alta['Valor_Total']:,.0f} "
                    f"inmovilizados en {int(alta['Num_SKUs'])} SKUs.",
                    "Tener el triple o cuádruple del mínimo significa dinero quieto en estantería "
                    "que podría usarse en productos que sí rotan.",
                    "Evaluar promociones o transferencias entre bodegas antes de comprar más a proveedores.",
                ))
                if pct_alta_filtrada > 0:
                    n_alta = int(pct_alta_filtrada * len(df_stock))
                    conclusiones.append(_conclusion(
                        "Muchos productos tienen el doble (o más) del stock mínimo",
                        f"{pct_alta_filtrada:.1%} de los SKUs filtrados ({n_alta:,}) supera "
                        "2× su punto de reorden.",
                        "Cuando ocurre en muchos artículos a la vez, suele ser política de compra "
                        "excesiva, no un caso aislado.",
                        "Revisar reglas de reabastecimiento con compras y planeación de demanda.",
                    ))

    with tab_bodega:
        if "Bodega_Origen" in df.columns and "Antiguedad_Revision_Dias" in df.columns:
            cats_bod = _multiselect_todos(
                "Categorías (opcional, filtra bodegas)",
                df["Categoria"].unique() if "Categoria" in df.columns else [],
                key="inv_filtro_cat_bodega",
            )
            df_bod = df[df["Categoria"].isin(cats_bod)] if cats_bod and "Categoria" in df.columns else df
            por_bodega = (
                df_bod.groupby("Bodega_Origen", as_index=False)
                .agg(
                    Antiguedad_Prom=("Antiguedad_Revision_Dias", "mean"),
                    Valor_Inventario=("Valor_Inventario", "sum"),
                    SKUs=("SKU_ID", "nunique"),
                )
                .sort_values("Antiguedad_Prom", ascending=False)
            )
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(
                    por_bodega, x="Antiguedad_Prom", y="Bodega_Origen",
                    orientation="h",
                    title="Días desde la última revisión física de stock",
                    labels={"Antiguedad_Prom": "Días sin revisar", "Bodega_Origen": "Bodega"},
                    color="Antiguedad_Prom",
                    color_continuous_scale="Reds",
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.bar(
                    por_bodega.sort_values("Valor_Inventario", ascending=True),
                    x="Valor_Inventario", y="Bodega_Origen",
                    orientation="h",
                    title="Valor del inventario en cada bodega (USD)",
                    labels={"Valor_Inventario": "USD", "Bodega_Origen": "Bodega"},
                    color="Valor_Inventario",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig2, use_container_width=True)

            peor = por_bodega.iloc[0]
            valor_filtrado = por_bodega["Valor_Inventario"].sum()
            conclusiones.append(_conclusion(
                "Algunas bodegas llevan mucho tiempo sin contar el stock",
                f"**{peor['Bodega_Origen']}** lleva un promedio de {peor['Antiguedad_Prom']:.0f} días "
                f"sin revisión y guarda USD {peor['Valor_Inventario']:,.0f} "
                f"en {int(peor['SKUs'])} SKUs.",
                "Si no se cuenta el inventario, pueden existir faltantes, sobrantes o productos "
                "vencidos que el sistema no refleja.",
                "Programar conteo cíclico urgente en esa bodega y comparar con lo que dice el sistema.",
            ))
            if valor_filtrado > 0:
                pct_peor = peor["Valor_Inventario"] / valor_filtrado
                conclusiones.append(_conclusion(
                    "El capital más expuesto está en la bodega menos controlada",
                    f"La bodega más desactualizada concentra {pct_peor:.1%} del valor "
                    f"inventariado en el filtro actual.",
                    "Mezclar mucho dinero en bodega con poca supervisión aumenta riesgo de "
                    "pérdidas no detectadas y pedidos fallidos por stock fantasma.",
                    "Priorizar auditoría física donde hay más USD en juego.",
                ))

    with tab_costos:
        if "Categoria" in df.columns and "Costo_Unitario_USD" in df.columns:
            cats_costo = _multiselect_todos(
                "Categorías para comparar costos",
                df["Categoria"].unique(),
                key="inv_filtro_cat_costo",
            )
            df_costo = df[df["Categoria"].isin(cats_costo)] if cats_costo else df.iloc[0:0]
            if df_costo.empty:
                st.info("Selecciona al menos una categoría.")
            else:
                fig = px.box(
                    df_costo, x="Categoria", y="Costo_Unitario_USD",
                    title="Rango de costo unitario por categoría (cada punto es un SKU)",
                    labels={"Costo_Unitario_USD": "USD por unidad", "Categoria": "Categoría"},
                    color="Categoria",
                )
                fig.update_layout(showlegend=False, xaxis_tickangle=-25)
                st.plotly_chart(fig, use_container_width=True)
                cat_cara = df_costo.groupby("Categoria")["Costo_Unitario_USD"].median().idxmax()
                mediana_cara = df_costo.groupby("Categoria")["Costo_Unitario_USD"].median().max()
                conclusiones.append(_conclusion(
                    "El costo de compra varía mucho según la categoría",
                    f"La mediana más alta está en **{cat_cara}** (USD {mediana_cara:,.2f} por unidad).",
                    "Ese costo es la base para calcular si una venta deja ganancia; "
                    "un precio de venta bajo en categorías caras genera pérdida.",
                    "Cruzar estos costos con precios de venta en **Preguntas Estratégicas**.",
                ))

        banderas = []
        for col, label in [
            ("Stock_Imputado", "Stock imputado"),
            ("Costo_Unitario_Winsorizado", "Costo winsorizado"),
            ("Categoria_Imputada", "Categoría imputada"),
            ("Lead_Time_Imputado", "Lead time imputado"),
        ]:
            if col in df.columns:
                banderas.append({"Alerta": label, "SKUs": int(df[col].sum())})
        if banderas:
            st.caption("SKUs con transformaciones de limpieza")
            st.dataframe(pd.DataFrame(banderas), use_container_width=True, hide_index=True)

    conclusiones.append(_conclusion(
        "El inventario define cuánto dinero está 'quieto' en bodega",
        f"En total hay USD {valor_total:,.0f} en stock ({len(df):,} registros de SKU).",
        "Ese monto no genera ingreso hasta que se vende; además fija el costo mínimo "
        "que debe cubrir cada precio de venta.",
        "Usar esta base de costos al revisar márgenes por canal y categoría.",
    ))
    if pct_confiable < 1:
        conclusiones.append(_conclusion(
            "Parte de los datos de inventario fue estimada en la limpieza",
            f"{pct_confiable:.1%} de los SKUs quedó sin imputaciones artificiales.",
            "Costos o stocks rellenados pueden verse 'normales' aunque en realidad falte información.",
            "Dar preferencia a filas con `Registro_Confiable = True` en reportes ejecutivos.",
        ))
    _mostrar_conclusiones("📋 Qué nos dice el inventario", conclusiones)


def render_analisis_feedback(df, n_transacciones=None):
    """Visualizaciones de negocio para el dataset de feedback."""
    _banner_fuentes_cruzadas()
    conclusiones = []

    nps_prom = df["Satisfaccion_NPS"].mean() if "Satisfaccion_NPS" in df.columns else 0
    pct_detractor = (
        (df["Segmento_NPS"] == "Detractor").mean()
        if "Segmento_NPS" in df.columns else 0
    )
    pct_ticket = df["Ticket_Soporte_Abierto"].mean() if "Ticket_Soporte_Abierto" in df.columns else 0
    pct_sin_resp = (
        (df["Recomienda_Marca"] == "Sin_Respuesta").mean()
        if "Recomienda_Marca" in df.columns else 0
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("NPS promedio", f"{nps_prom:.1f}",
              help="Net Promoter Score: de -100 a 100; arriba de 50 suele ser excelente.")
    k2.metric("Detractores", f"{pct_detractor:.1%}",
              help="Clientes con NPS bajo que probablemente no recomienden la marca.")
    k3.metric("Tickets de soporte", f"{pct_ticket:.1%}",
              help="Proporción de opiniones con un reclamo o ticket abierto.")
    if n_transacciones:
        k4.metric("Cobertura sobre ventas", f"{len(df)/n_transacciones:.1%}", f"{len(df):,} opiniones",
                  help="Qué tan representativa es la encuesta frente al total de ventas.")
    else:
        k4.metric("Sin recomendar marca", f"{pct_sin_resp:.1%}",
                  help="Clientes que no respondieron si recomendarían la empresa.")

    tab_nps, tab_fidel, tab_soporte = st.tabs(
        ["📊 NPS y logística", "💬 Fidelidad y recomendación", "🎫 Soporte"]
    )

    with tab_nps:
        seg_sel = _multiselect_todos(
            "Segmentos NPS a incluir",
            df["Segmento_NPS"].unique() if "Segmento_NPS" in df.columns else [],
            key="fb_filtro_segmento",
        )
        df_fb = df[df["Segmento_NPS"].isin(seg_sel)] if seg_sel and "Segmento_NPS" in df.columns else df

        if "Segmento_NPS" in df_fb.columns and len(df_fb):
            segmento = df_fb["Segmento_NPS"].value_counts().reset_index()
            segmento.columns = ["Segmento", "Opiniones"]
            orden = ["Promotor", "Pasivo", "Detractor"]
            segmento["Segmento"] = pd.Categorical(segmento["Segmento"], categories=orden, ordered=True)
            segmento = segmento.sort_values("Segmento")
            fig = px.bar(
                segmento, x="Segmento", y="Opiniones",
                title="¿Cuántos clientes recomiendan, son neutros o critican?",
                color="Segmento",
                color_discrete_map={"Promotor": "#22c55e", "Pasivo": "#eab308", "Detractor": "#ef4444"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if "Rating_Logistica" in df_fb.columns and "Satisfaccion_NPS" in df_fb.columns and len(df_fb):
            umbral_rating = st.slider(
                "Rating mínimo de logística en la nube de puntos",
                min_value=1, max_value=5, value=1,
                key="fb_min_rating_log",
            )
            df_scatter = df_fb[df_fb["Rating_Logistica"] >= umbral_rating]
            fig2 = px.scatter(
                df_scatter, x="Rating_Logistica", y="Satisfaccion_NPS",
                color="Segmento_NPS" if "Segmento_NPS" in df_scatter.columns else None,
                title="Relación entre nota de entrega y NPS (cada punto es un cliente)",
                labels={"Rating_Logistica": "Nota entrega (1=mala, 5=excelente)", "Satisfaccion_NPS": "NPS"},
                trendline="ols",
                opacity=0.6,
            )
            st.plotly_chart(fig2, use_container_width=True)
            corr = df_scatter["Rating_Logistica"].corr(df_scatter["Satisfaccion_NPS"])
            if not pd.isna(corr):
                fuerza = "fuerte" if abs(corr) >= 0.5 else "moderada" if abs(corr) >= 0.3 else "débil"
                conclusiones.append(_conclusion(
                    "La experiencia de entrega arrastra la satisfacción general",
                    f"Entre nota de logística y NPS hay correlación **{corr:.2f}** ({fuerza}): "
                    f"cuando baja la nota de entrega, baja el NPS.",
                    "El cliente no separa 'producto' de 'llegó a tiempo y en buen estado': "
                    "una mala entrega contamina toda la percepción de la marca.",
                    "Atacar tiempos y estado del paquete antes de campañas de fidelización.",
                ))

    with tab_fidel:
        seg_fidel = _multiselect_todos(
            "Segmentos para fidelidad",
            df["Segmento_NPS"].unique() if "Segmento_NPS" in df.columns else [],
            key="fb_filtro_segmento_fidel",
        )
        df_fidel = df[df["Segmento_NPS"].isin(seg_fidel)] if seg_fidel and "Segmento_NPS" in df.columns else df

        if "Rating_Producto" in df_fidel.columns and "Rating_Logistica" in df_fidel.columns and len(df_fidel):
            ratings = pd.DataFrame({
                "Dimensión": ["Producto", "Logística"],
                "Rating promedio": [
                    df_fidel["Rating_Producto"].mean(),
                    df_fidel["Rating_Logistica"].mean(),
                ],
            })
            fig = px.bar(
                ratings, x="Dimensión", y="Rating promedio",
                title="Nota promedio: ¿qué evalúan mejor, producto o entrega?",
                color="Dimensión",
                range_y=[0, 5],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if "Recomienda_Marca" in df_fidel.columns and len(df_fidel):
            recom = df_fidel["Recomienda_Marca"].value_counts().reset_index()
            recom.columns = ["Respuesta", "Opiniones"]
            fig2 = px.bar(
                recom, x="Respuesta", y="Opiniones",
                title="¿El cliente dice que recomendaría la marca?",
                color="Respuesta",
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            pct_sin_fidel = (df_fidel["Recomienda_Marca"] == "Sin_Respuesta").mean()
            if pct_sin_fidel > 0:
                n_sin = int(pct_sin_fidel * len(df_fidel))
                conclusiones.append(_conclusion(
                    "Muchos clientes no responden si recomendarían la marca",
                    f"{pct_sin_fidel:.1%} de las opiniones filtradas ({n_sin:,}) dejó la pregunta "
                    "de recomendación en blanco.",
                    "Quien no responde suele estar desenganchado: no le importa lo suficiente "
                    "como para defender ni criticar la marca.",
                    "Contactar muestras de 'Sin respuesta' para entender si hubo mala experiencia silenciosa.",
                ))
            if "Rating_Producto" in df_fidel.columns and "Rating_Logistica" in df_fidel.columns:
                r_prod = df_fidel["Rating_Producto"].mean()
                r_log = df_fidel["Rating_Logistica"].mean()
                diff = abs(r_prod - r_log)
                peor = "la entrega" if r_log < r_prod else "el producto"
                conclusiones.append(_conclusion(
                    "Producto y entrega no puntúan igual",
                    f"Nota promedio producto: **{r_prod:.1f}/5**. "
                    f"Nota promedio entrega: **{r_log:.1f}/5** "
                    f"(diferencia de {diff:.1f} puntos).",
                    f"El cliente percibe más problemas en **{peor}**. Mejorar solo catálogo "
                    "no alcanza si la queja principal es operativa.",
                    f"Enfocar mejoras donde la nota es más baja ({peor}).",
                ))

    with tab_soporte:
        seg_sop = _multiselect_todos(
            "Segmentos para análisis de soporte",
            df["Segmento_NPS"].unique() if "Segmento_NPS" in df.columns else [],
            key="fb_filtro_segmento_soporte",
        )
        df_sop = df[df["Segmento_NPS"].isin(seg_sop)] if seg_sop and "Segmento_NPS" in df.columns else df

        if "Segmento_NPS" in df_sop.columns and "Ticket_Soporte_Abierto" in df_sop.columns and len(df_sop):
            tickets = (
                df_sop.groupby("Segmento_NPS", as_index=False)
                .agg(Tasa_Ticket=("Ticket_Soporte_Abierto", "mean"), N=("Feedback_ID", "count"))
            )
            orden = ["Promotor", "Pasivo", "Detractor"]
            tickets["Segmento_NPS"] = pd.Categorical(tickets["Segmento_NPS"], categories=orden, ordered=True)
            tickets = tickets.sort_values("Segmento_NPS")
            fig = px.bar(
                tickets, x="Segmento_NPS", y="Tasa_Ticket",
                title="¿Qué tan seguido abren ticket de soporte según su NPS?",
                labels={"Tasa_Ticket": "Proporción con ticket abierto", "Segmento_NPS": "Segmento"},
                color="Segmento_NPS",
                color_discrete_map={"Promotor": "#22c55e", "Pasivo": "#eab308", "Detractor": "#ef4444"},
                hover_data={"N": True},
            )
            fig.update_layout(showlegend=False)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            if len(tickets) >= 2:
                tasa_det = tickets.loc[tickets["Segmento_NPS"] == "Detractor", "Tasa_Ticket"]
                tasa_pro = tickets.loc[tickets["Segmento_NPS"] == "Promotor", "Tasa_Ticket"]
                if not tasa_det.empty and not tasa_pro.empty and tasa_pro.iloc[0] > 0:
                    ratio = tasa_det.iloc[0] / tasa_pro.iloc[0]
                    conclusiones.append(_conclusion(
                        "Los clientes insatisfechos recurren mucho más al soporte",
                        f"Los detractores abren ticket **{ratio:.1f} veces** más que los promotores "
                        f"({tasa_det.iloc[0]:.1%} vs. {tasa_pro.iloc[0]:.1%}).",
                        "La insatisfacción no se queda en una mala nota: termina consumiendo "
                        "tiempo del equipo de soporte y puede escalar en redes sociales.",
                        "Priorizar causas raíz de detractores (entrega, stock, promesas incumplidas).",
                    ))

        pct_confiable = df["Registro_Confiable"].mean() if "Registro_Confiable" in df.columns else 1
        st.metric("Opiniones medidas sin imputación de datos", f"{pct_confiable:.1%}")

    if pct_detractor > 0:
        n_det = int(pct_detractor * len(df))
        conclusiones.append(_conclusion(
            "Hay una base relevante de clientes muy insatisfechos",
            f"{pct_detractor:.1%} del feedback ({n_det:,} opiniones) es de **detractores**. "
            f"El NPS promedio del archivo es {nps_prom:.1f}.",
            "Los detractores no solo dejan de comprar: pueden desincentivar a otros clientes "
            "si no se atiende la causa del malestar.",
            "Cruzar detractores con tiempos de entrega y categorías en **Preguntas Estratégicas**.",
        ))
    if n_transacciones:
        cobertura = len(df) / n_transacciones
        conclusiones.append(_conclusion(
            "Las opiniones no cubren todas las ventas",
            f"Solo {cobertura:.1%} de las transacciones tiene feedback "
            f"({len(df):,} opiniones sobre {n_transacciones:,} ventas).",
            "El NPS que vemos puede no representar al cliente silencioso que nunca respondió la encuesta.",
            "Complementar con reclamos, devoluciones y tiempos de entrega del total de ventas.",
        ))
    _mostrar_conclusiones("📋 Qué nos dice el feedback de clientes", conclusiones)


def render_preguntas_estrategicas(df_maestro):
    """Las 5 preguntas de gerencia con gráficos, filtros y conclusiones ejecutivas."""
    st.title("📌 Preguntas Estratégicas de Alta Gerencia")
    st.markdown(
        "Respuestas basadas en la **Fuente Única de Verdad** (transacciones + inventario + feedback). "
        "Usa los filtros del sidebar para acotar categoría, bodega, ciudad o canal."
    )
    st.caption(f"Analizando **{len(df_maestro):,}** transacciones con los filtros actuales.")

    if df_maestro.empty:
        st.warning("No hay registros con los filtros actuales. Amplía la selección en el sidebar.")
        return

    conclusiones_globales = []

    tabs = st.tabs([
        "💸 Fuga de capital",
        "🚚 Crisis logística",
        "👻 Venta invisible",
        "💬 Fidelidad del cliente",
        "⚠️ Riesgo operativo",
    ])

    # --- 1. Fuga de capital -------------------------------------------------
    with tabs[0]:
        st.subheader("¿Qué productos se venden perdiendo dinero?")
        canales_p1 = _multiselect_todos(
            "Canales a analizar",
            df_maestro["Canal_Venta"].unique(),
            key="pe_canales_margen",
        )
        df_p1 = df_maestro[df_maestro["Canal_Venta"].isin(canales_p1)] if canales_p1 else df_maestro.iloc[0:0]
        df_neg = df_p1[df_p1["Margen_Utilidad"] < 0]

        if df_neg.empty:
            st.success("Con los filtros actuales no hay ventas con margen negativo.")
            df_margen = (
                df_p1[~df_p1["SKU_Fantasma"]]
                .groupby("Categoria", as_index=False)["Margen_Utilidad_Pct"]
                .mean()
                .sort_values("Margen_Utilidad_Pct")
                .head(12)
            )
            if not df_margen.empty:
                fig_bajo = px.bar(
                    df_margen, x="Margen_Utilidad_Pct", y="Categoria", orientation="h",
                    title="Categorías con margen más ajustado (aunque aún positivo)",
                    labels={"Margen_Utilidad_Pct": "Margen promedio (%)", "Categoria": "Categoría"},
                    color="Margen_Utilidad_Pct",
                    color_continuous_scale="Oranges",
                )
                st.plotly_chart(fig_bajo, use_container_width=True)
                _nota_grafico(
                    "Aunque no hay pérdidas netas, estas categorías están cerca del umbral "
                    "y conviene vigilarlas antes de que pasen a margen negativo."
                )
            conclusiones_globales.append(_conclusion(
                "¿Hay fuga de capital por precios?",
                "No se detectaron transacciones con margen negativo en el subconjunto filtrado.",
                "Eso no garantiza rentabilidad saludable: el margen puede ser positivo pero bajo.",
                "Revisar categorías con margen cercano a cero y presión de flete en canal Online.",
            ))
        else:
            perdida_total = df_neg["Margen_Utilidad"].sum()
            pct_canal_online = (df_neg["Canal_Venta"] == "Online").mean()
            pct_online_base = (df_p1["Canal_Venta"] == "Online").mean()

            c1, c2, c3 = st.columns(3)
            c1.metric("SKUs en pérdida", f"{df_neg['SKU_ID'].nunique():,}")
            c2.metric("Pérdida acumulada", f"USD {abs(perdida_total):,.0f}")
            c3.metric("Ventas en pérdida vía Online", f"{pct_canal_online:.1%}")

            st.plotly_chart(
                _grafico_comparacion_canales(df_p1, df_neg),
                use_container_width=True,
            )
            _nota_grafico(
                "Si la barra roja de un canal supera a la gris, ese canal aporta más pérdidas "
                "de las que vende en total — señal de precios mal calibrados en ese canal."
            )

            col1, col2 = st.columns(2)
            with col1:
                por_canal = (
                    df_neg.groupby("Canal_Venta")["Margen_Utilidad"]
                    .sum().sort_values().reset_index()
                )
                st.plotly_chart(px.bar(
                    por_canal, x="Margen_Utilidad", y="Canal_Venta", orientation="h",
                    title="Cuánto dinero se pierde por canal",
                    labels={"Margen_Utilidad": "Pérdida acumulada (USD)"},
                    color="Margen_Utilidad", color_continuous_scale="Reds_r",
                ), use_container_width=True)
                _nota_grafico("Muestra en USD dónde se concentra la fuga de capital.")
            with col2:
                top_skus = (
                    df_neg.groupby("SKU_ID")["Margen_Utilidad"].sum()
                    .sort_values().head(15).reset_index()
                )
                st.plotly_chart(px.bar(
                    top_skus, x="Margen_Utilidad", y="SKU_ID", orientation="h",
                    title="Productos que más dinero pierden",
                    labels={"Margen_Utilidad": "Pérdida acumulada (USD)"},
                ), use_container_width=True)
                _nota_grafico("Identifica SKUs concretos para revisión de precio o costo con proveedor.")

            por_cat_neg = (
                df_neg.groupby("Categoria")["Margen_Utilidad"].sum()
                .sort_values().head(10).reset_index()
            )
            if not por_cat_neg.empty:
                st.plotly_chart(px.bar(
                    por_cat_neg, x="Margen_Utilidad", y="Categoria", orientation="h",
                    title="Categorías que más contribuyen a la pérdida",
                    labels={"Margen_Utilidad": "Pérdida acumulada (USD)"},
                    color="Margen_Utilidad", color_continuous_scale="Reds",
                ), use_container_width=True)
                _nota_grafico(
                    "Complementa la respuesta: indica si el problema es de un canal, "
                    "de pocos SKUs o de líneas completas de producto."
                )

            if pct_canal_online > pct_online_base * 1.3:
                conclusiones_globales.append(_conclusion(
                    "El canal Online concentra las ventas que pierden dinero",
                    f"**{pct_canal_online:.1%}** de las ventas con margen negativo ocurre en Online, "
                    f"frente a **{pct_online_base:.1%}** de participación general del canal. "
                    f"Pérdida acumulada: USD {abs(perdida_total):,.0f}.",
                    "Cuando las pérdidas se concentran en un canal, suele haber precios mal calibrados "
                    "o promociones que no cubren costo + envío.",
                    "Auditar precios y costos en Online; pausar SKUs con peor margen hasta recostear.",
                ))
            else:
                conclusiones_globales.append(_conclusion(
                    "Las pérdidas están repartidas entre canales",
                    f"Online representa {pct_canal_online:.1%} de las ventas en pérdida "
                    f"(similar a su peso general {pct_online_base:.1%}). "
                    f"Total perdido: USD {abs(perdida_total):,.0f} en {df_neg['SKU_ID'].nunique()} SKUs.",
                    "El problema parece puntual por producto, no una falla masiva de un solo canal.",
                    "Revisar lista de SKUs perdedores y negociar costo con proveedor o ajustar precio.",
                ))

    # --- 2. Crisis logística ------------------------------------------------
    with tabs[1]:
        st.subheader("¿Dónde la mala entrega destruye la satisfacción del cliente?")
        df_fb = df_maestro[df_maestro["Tiene_Feedback"]].copy()
        min_muestra = st.slider(
            "Mínimo de opiniones por ciudad·bodega",
            5, 30, 8, key="pe_min_corr",
            help="Evita correlaciones basadas en muy pocos clientes.",
        )
        ciudades_p2 = _multiselect_todos(
            "Ciudades destino",
            df_fb["Ciudad_Destino"].unique() if len(df_fb) else [],
            key="pe_ciudades_log",
        )
        if ciudades_p2:
            df_fb = df_fb[df_fb["Ciudad_Destino"].isin(ciudades_p2)]

        df_fb["Zona"] = df_fb["Ciudad_Destino"].fillna("(Sin dato)") + " · " + df_fb["Bodega_Origen"].fillna("(Sin dato)")
        filas = []
        for zona, g in df_fb.groupby("Zona"):
            if len(g) >= min_muestra and g["Tiempo_Entrega_Real"].nunique() > 1:
                corr = g["Tiempo_Entrega_Real"].corr(g["Satisfaccion_NPS"])
                if pd.notna(corr):
                    filas.append({"Zona": zona, "Correlacion": corr, "n": len(g)})

        if not filas:
            st.info("No hay suficientes datos con feedback para este análisis. Baja el mínimo de opiniones.")
            conclusiones_globales.append(_conclusion(
                "¿La logística afecta el NPS?",
                "Con los filtros actuales no hay zonas con muestra suficiente para medir la relación entrega–NPS.",
                "Sin cruce venta–opinión no se puede priorizar dónde cambiar operador logístico.",
                "Ampliar encuestas post-entrega o bajar el umbral de muestra mínima.",
            ))
        else:
            df_corr = pd.DataFrame(filas).sort_values("Correlacion")
            peor = df_corr.iloc[0]
            peor_zona = peor["Zona"]

            fig_corr = px.bar(
                df_corr, x="Correlacion", y="Zona", orientation="h",
                title="Zonas donde más días de entrega = peor NPS (correlación negativa)",
                color="Correlacion", color_continuous_scale="RdYlGn",
                hover_data=["n"],
            )
            fig_corr.add_vline(x=peor["Correlacion"], line_dash="dot", line_color="#ef4444")
            st.plotly_chart(fig_corr, use_container_width=True)
            _nota_grafico(
                f"La zona **{peor_zona}** (primera en la lista) es la prioridad: "
                f"correlación {peor['Correlacion']:.2f} con {int(peor['n'])} opiniones."
            )

            df_peor = df_fb[df_fb["Zona"] == peor_zona]
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.scatter(
                    df_peor, x="Tiempo_Entrega_Real", y="Satisfaccion_NPS",
                    title=f"Detalle zona crítica: {peor_zona}",
                    trendline="ols",
                    labels={"Tiempo_Entrega_Real": "Días de entrega", "Satisfaccion_NPS": "NPS"},
                    color_discrete_sequence=["#ef4444"],
                ), use_container_width=True)
                _nota_grafico(
                    "Cada punto es un cliente de la zona citada en la respuesta: "
                    "confirma visualmente que a más días, cae el NPS."
                )
            with col2:
                por_bodega = (
                    df_fb.groupby("Bodega_Origen")["Entrega_Tardia"].mean()
                    .sort_values(ascending=False).reset_index()
                )
                fig3 = px.bar(
                    por_bodega, x="Entrega_Tardia", y="Bodega_Origen", orientation="h",
                    title=f"Entregas tardías por bodega (SLA {SLA_ENTREGA_DIAS} días)",
                    labels={"Entrega_Tardia": "Proporción tardías"},
                    color="Entrega_Tardia", color_continuous_scale="Reds",
                )
                fig3.update_xaxes(tickformat=".0%")
                st.plotly_chart(fig3, use_container_width=True)
                _nota_grafico(
                    "Contexto operativo: bodegas con más incumplimientos de plazo "
                    "alimentan la crisis logística detectada arriba."
                )

            resumen_zona = (
                df_fb.groupby("Zona", as_index=False)
                .agg(
                    Dias_Entrega=("Tiempo_Entrega_Real", "median"),
                    NPS_Prom=("Satisfaccion_NPS", "mean"),
                    Pct_Tardias=("Entrega_Tardia", "mean"),
                    Opiniones=("Transaccion_ID", "count"),
                )
                .query("Opiniones >= @min_muestra")
                .sort_values("NPS_Prom")
                .head(8)
            )
            if not resumen_zona.empty:
                fig4 = px.bar(
                    resumen_zona, x="NPS_Prom", y="Zona", orientation="h",
                    title="Zonas con peor NPS promedio (mínimo de opiniones aplicado)",
                    labels={"NPS_Prom": "NPS promedio", "Zona": "Ciudad · Bodega"},
                    color="Pct_Tardias", color_continuous_scale="Reds",
                    hover_data={"Dias_Entrega": ":.0f", "Opiniones": True},
                )
                st.plotly_chart(fig4, use_container_width=True)
                _nota_grafico(
                    "Cruza satisfacción con % de tardías por zona para priorizar "
                    "intervención logística donde más duele al cliente."
                )

            conclusiones_globales.append(_conclusion(
                "Hay zonas donde retrasar la entrega empeora fuerte el NPS",
                f"La combinación **{peor['Zona']}** tiene correlación **{peor['Correlacion']:.2f}** "
                f"(basada en {int(peor['n'])} opiniones): a más días, peor nota.",
                "En esas rutas la logística no es un detalle operativo: es un freno directo "
                "a la recomendación de la marca.",
                "Evaluar cambio de operador o replanteo de ruta en esa ciudad·bodega de inmediato.",
            ))

    # --- 3. Venta invisible -------------------------------------------------
    with tabs[2]:
        st.subheader("¿Cuánto dinero entra por productos que no existen en inventario?")
        canales_p3 = _multiselect_todos(
            "Canales para ingreso en riesgo",
            df_maestro["Canal_Venta"].unique(),
            key="pe_canales_fantasma",
        )
        df_p3 = df_maestro[df_maestro["Canal_Venta"].isin(canales_p3)] if canales_p3 else df_maestro.iloc[0:0]

        ingreso_total = df_p3["Ingreso_Bruto"].sum()
        ingreso_riesgo = df_p3["Ingreso_En_Riesgo"].sum()
        ingreso_seguro = ingreso_total - ingreso_riesgo
        pct_riesgo = (ingreso_riesgo / ingreso_total) if ingreso_total else 0
        n_fantasma = int(df_p3["SKU_Fantasma"].sum()) if "SKU_Fantasma" in df_p3.columns else 0
        pct_tx_fantasma = (df_p3["SKU_Fantasma"].mean()) if "SKU_Fantasma" in df_p3.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ingreso analizado", f"USD {ingreso_total:,.0f}")
        c2.metric("Ingreso en riesgo", f"USD {ingreso_riesgo:,.0f}")
        c3.metric("% ingreso en riesgo", f"{pct_riesgo:.1%}")
        c4.metric("Transacciones fantasma", f"{pct_tx_fantasma:.1%}", f"{n_fantasma:,} ventas")

        canal_top_riesgo = "—"
        col_pie, col_stack = st.columns(2)
        with col_pie:
            st.plotly_chart(
                _grafico_ingreso_riesgo_total(ingreso_seguro, ingreso_riesgo),
                use_container_width=True,
            )
            _nota_grafico(
                f"Visualiza directamente el **{pct_riesgo:.1%}** del ingreso sin respaldo "
                "en inventario citado en la respuesta."
            )
        with col_stack:
            por_canal_full = (
                df_p3.groupby("Canal_Venta", as_index=False)
                .agg(Ingreso_Total=("Ingreso_Bruto", "sum"), Ingreso_Riesgo=("Ingreso_En_Riesgo", "sum"))
            )
            por_canal_full["Ingreso_Seguro"] = (
                por_canal_full["Ingreso_Total"] - por_canal_full["Ingreso_Riesgo"]
            )
            por_canal_long = por_canal_full.melt(
                id_vars="Canal_Venta",
                value_vars=["Ingreso_Seguro", "Ingreso_Riesgo"],
                var_name="Tipo", value_name="USD",
            )
            por_canal_long["Tipo"] = por_canal_long["Tipo"].map({
                "Ingreso_Seguro": "Con respaldo",
                "Ingreso_Riesgo": "SKU fantasma",
            })
            fig_stack = px.bar(
                por_canal_long, x="Canal_Venta", y="USD", color="Tipo",
                title="Composición del ingreso por canal: seguro vs. en riesgo",
                labels={"USD": "USD", "Canal_Venta": "Canal"},
                color_discrete_map={"Con respaldo": "#22c55e", "SKU fantasma": "#ef4444"},
            )
            fig_stack.update_layout(barmode="stack")
            st.plotly_chart(fig_stack, use_container_width=True)
            canal_top_riesgo = (
                por_canal_full.sort_values("Ingreso_Riesgo", ascending=False).iloc[0]["Canal_Venta"]
                if not por_canal_full.empty else "—"
            )
            _nota_grafico(
                f"El canal **{canal_top_riesgo}** concentra la porción roja más grande: "
                "es el foco para auditar catálogo y bloqueos de venta."
            )

        col1, col2 = st.columns(2)
        with col1:
            por_mes = df_p3.groupby("Mes_Venta", as_index=False).agg(
                Ingreso_Riesgo=("Ingreso_En_Riesgo", "sum"),
                Ingreso_Total=("Ingreso_Bruto", "sum"),
            )
            por_mes["Pct_Riesgo"] = np.where(
                por_mes["Ingreso_Total"] > 0,
                por_mes["Ingreso_Riesgo"] / por_mes["Ingreso_Total"],
                0,
            )
            fig_mes = px.line(
                por_mes, x="Mes_Venta", y="Pct_Riesgo", markers=True,
                title="Evolución del % de ingreso en riesgo mes a mes",
                labels={"Pct_Riesgo": "Proporción en riesgo", "Mes_Venta": "Mes"},
            )
            fig_mes.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_mes, use_container_width=True)
            _nota_grafico("Si la línea sube, el problema de venta invisible está empeorando en el tiempo.")
        with col2:
            pct_fantasma_canal = (
                df_p3.groupby("Canal_Venta")["SKU_Fantasma"].mean()
                .sort_values(ascending=False).reset_index()
            )
            fig_pct = px.bar(
                pct_fantasma_canal, x="SKU_Fantasma", y="Canal_Venta", orientation="h",
                title="Proporción de ventas SKU fantasma por canal",
                labels={"SKU_Fantasma": "Proporción de ventas fantasma"},
                color="SKU_Fantasma", color_continuous_scale="Reds",
            )
            fig_pct.update_xaxes(tickformat=".1%")
            st.plotly_chart(fig_pct, use_container_width=True)
            _nota_grafico(
                "Complementa el monto en USD: muestra qué canal vende más seguido "
                "productos fuera de catálogo."
            )

        conclusiones_globales.append(_conclusion(
            "Parte de las ventas no tiene respaldo en el catálogo de inventario",
            f"**{pct_riesgo:.1%}** del ingreso filtrado (USD {ingreso_riesgo:,.0f}) corresponde a "
            f"**SKU fantasma** — productos vendidos sin registro confiable en bodega. "
            f"Canal con más riesgo: **{canal_top_riesgo}**.",
            "Es dinero que entra sin costo conocido ni trazabilidad: no se puede calcular margen "
            "real ni detectar fraude o errores de catalogación.",
            "Crear regla de bloqueo de venta si el SKU no existe en inventario; auditar altas en catálogo.",
        ))

    # --- 4. Fidelidad -------------------------------------------------------
    with tabs[3]:
        st.subheader("¿Hay categorías con mucho stock pero clientes insatisfechos?")
        cats_p4 = _multiselect_todos(
            "Categorías a evaluar",
            df_maestro.loc[~df_maestro["SKU_Fantasma"], "Categoria"].unique(),
            key="pe_categorias_fidel",
        )
        df_p4 = df_maestro[~df_maestro["SKU_Fantasma"]]
        if cats_p4:
            df_p4 = df_p4[df_p4["Categoria"].isin(cats_p4)]

        resumen_cat = (
            df_p4.groupby("Categoria")
            .agg(
                Ratio_Reorden_Prom=("Ratio_Stock_Reorden", "mean"),
                NPS_Promedio=("Satisfaccion_NPS", "mean"),
                Margen_Pct_Promedio=("Margen_Utilidad_Pct", "mean"),
                N=("Transaccion_ID", "count"),
            )
            .reset_index()
        )

        if resumen_cat.empty or resumen_cat["NPS_Promedio"].isna().all():
            st.info("No hay feedback suficiente por categoría con los filtros actuales.")
            conclusiones_globales.append(_conclusion(
                "¿Hay desajuste stock vs. satisfacción?",
                "No hay datos de NPS por categoría en el subconjunto filtrado.",
                "Sin opinión del cliente no se detecta la paradoja de mucho inventario + baja lealtad.",
                "Ampliar encuestas o cruzar con tickets de soporte por línea de producto.",
            ))
        else:
            mediana_stock = resumen_cat["Ratio_Reorden_Prom"].median()
            mediana_nps = resumen_cat["NPS_Promedio"].median()
            paradoja = resumen_cat[
                (resumen_cat["Ratio_Reorden_Prom"] >= mediana_stock)
                & (resumen_cat["NPS_Promedio"] < mediana_nps)
            ]
            resumen_cat["Grupo"] = np.where(
                resumen_cat["Categoria"].isin(paradoja["Categoria"]),
                "Paradoja: mucho stock + bajo NPS",
                "Resto de categorías",
            )

            fig = px.scatter(
                resumen_cat, x="Ratio_Reorden_Prom", y="NPS_Promedio", size="N",
                color="Grupo", text="Categoria",
                title="Cuadrante inferior-derecho = mucho stock y clientes insatisfechos",
                labels={
                    "Ratio_Reorden_Prom": "Stock vs. punto de reorden (promedio)",
                    "NPS_Promedio": "NPS promedio",
                },
                color_discrete_map={
                    "Paradoja: mucho stock + bajo NPS": "#ef4444",
                    "Resto de categorías": "#94a3b8",
                },
            )
            fig.update_traces(textposition="top center")
            fig.add_vline(x=mediana_stock, line_dash="dash", line_color="gray",
                          annotation_text="Mediana stock")
            fig.add_hline(y=mediana_nps, line_dash="dash", line_color="gray",
                          annotation_text="Mediana NPS")
            st.plotly_chart(fig, use_container_width=True)
            _nota_grafico(
                "Los puntos rojos son las categorías citadas en la respuesta: "
                "inventario alto pero satisfacción por debajo de la mediana."
            )

            if not paradoja.empty:
                nombres = ", ".join(paradoja["Categoria"].tolist())
                margen_paradoja = paradoja["Margen_Pct_Promedio"].mean()
                margen_resto = resumen_cat.loc[
                    ~resumen_cat["Categoria"].isin(paradoja["Categoria"]), "Margen_Pct_Promedio"
                ].mean()

                col_a, col_b = st.columns(2)
                with col_a:
                    comp_nps = paradoja.melt(
                        id_vars="Categoria",
                        value_vars=["Ratio_Reorden_Prom", "NPS_Promedio"],
                        var_name="Métrica", value_name="Valor",
                    )
                    comp_nps["Métrica"] = comp_nps["Métrica"].map({
                        "Ratio_Reorden_Prom": "Stock / reorden",
                        "NPS_Promedio": "NPS promedio",
                    })
                    fig_par = px.bar(
                        comp_nps, x="Categoria", y="Valor", color="Métrica", barmode="group",
                        title="Detalle de categorías en paradoja",
                        labels={"Valor": "Valor", "Categoria": "Categoría"},
                    )
                    fig_par.update_layout(xaxis_tickangle=-25)
                    st.plotly_chart(fig_par, use_container_width=True)
                    _nota_grafico(
                        f"Desglosa **{nombres}**: mucho stock (barra azul) pero NPS bajo (barra naranja)."
                    )
                with col_b:
                    df_margen_comp = pd.DataFrame({
                        "Grupo": ["Categorías en paradoja", "Resto"],
                        "Margen promedio (%)": [margen_paradoja, margen_resto],
                    })
                    fig_m = px.bar(
                        df_margen_comp, x="Grupo", y="Margen promedio (%)",
                        title="¿Es problema de precio o de producto?",
                        color="Grupo",
                        color_discrete_sequence=["#ef4444", "#94a3b8"],
                    )
                    fig_m.update_layout(showlegend=False)
                    st.plotly_chart(fig_m, use_container_width=True)
                    _nota_grafico(
                        "Si la paradoja tiene margen más alto, apunta a sobreprecio; "
                        "si es similar o menor, apunta a calidad del producto."
                    )

                if margen_paradoja > margen_resto:
                    causa = (
                        f"margen alto ({margen_paradoja:.1f}% vs. {margen_resto:.1f}% del resto), "
                        "lo que sugiere precio elevado frente al valor que percibe el cliente."
                    )
                    accion = "Revisar propuesta de valor, calidad percibida y política de precios en esas líneas."
                else:
                    causa = (
                        f"margen no superior al resto ({margen_paradoja:.1f}% vs. {margen_resto:.1f}%), "
                        "apuntando más a calidad de producto que a sobreprecio."
                    )
                    accion = "Investigar defectos, devoluciones y quejas abiertas en esas categorías."
                conclusiones_globales.append(_conclusion(
                    "Hay categorías con exceso de stock y clientes insatisfechos",
                    f"**{nombres}** combinan stock por encima de la mediana con NPS por debajo. {causa}",
                    "Tener mucho inventario de algo que no gusta immoviliza capital y empeora la experiencia.",
                    accion,
                ))
            else:
                st.plotly_chart(px.bar(
                    resumen_cat.sort_values("NPS_Promedio").head(8),
                    x="NPS_Promedio", y="Categoria", orientation="h",
                    title="Categorías con mejor NPS (sin paradoja detectada)",
                    labels={"NPS_Promedio": "NPS promedio"},
                    color="NPS_Promedio", color_continuous_scale="Greens",
                ), use_container_width=True)
                _nota_grafico(
                    "Ningún punto rojo en el scatter: no hay categoría con mucho stock y bajo NPS a la vez."
                )
                conclusiones_globales.append(_conclusion(
                    "No aparece la paradoja stock alto + NPS bajo",
                    "Ninguna categoría filtrada muestra simultáneamente exceso de stock y NPS bajo.",
                    "El problema de fidelidad, si existe, puede estar en logística o postventa más que en catálogo.",
                    "Cruzar NPS con tiempos de entrega en la pestaña Crisis logística.",
                ))

    # --- 5. Riesgo operativo ------------------------------------------------
    with tabs[4]:
        st.subheader("¿Qué bodegas operan sin control de stock y generan más reclamos?")
        bodegas_p5 = _multiselect_todos(
            "Bodegas a comparar",
            df_maestro["Bodega_Origen"].unique(),
            key="pe_bodegas_riesgo",
        )
        df_p5 = df_maestro[df_maestro["Bodega_Origen"].isin(bodegas_p5)] if bodegas_p5 else df_maestro.iloc[0:0]
        df_fb5 = df_p5[df_p5["Tiene_Feedback"]]

        antiguedad_bodega = df_p5.groupby("Bodega_Origen")["Antiguedad_Revision_Dias"].mean()
        soporte_bodega = df_fb5.groupby("Bodega_Origen")["Ticket_Soporte_Abierto"].mean()
        n_bodega = df_p5.groupby("Bodega_Origen").size()

        resumen_bodega = pd.DataFrame({
            "Antiguedad_Revision_Prom": antiguedad_bodega,
            "Tasa_Ticket_Soporte": soporte_bodega,
            "N": n_bodega,
        }).dropna().reset_index()

        if resumen_bodega.empty:
            st.info("No hay datos cruzados de inventario y soporte para las bodegas seleccionadas.")
            conclusiones_globales.append(_conclusion(
                "¿Qué bodegas operan a ciegas?",
                "No hay suficiente cruce entre antigüedad de revisión y tickets de soporte.",
                "Sin medir ambas cosas juntas no se prioriza dónde auditar stock físico.",
                "Seleccionar más bodegas o ampliar la muestra de feedback.",
            ))
        else:
            max_ant = resumen_bodega["Antiguedad_Revision_Prom"].max()
            if max_ant > 0:
                resumen_bodega["Riesgo_Operativo"] = (
                    resumen_bodega["Antiguedad_Revision_Prom"] / max_ant
                    + resumen_bodega["Tasa_Ticket_Soporte"]
                )
            else:
                resumen_bodega["Riesgo_Operativo"] = resumen_bodega["Tasa_Ticket_Soporte"]
            peor = resumen_bodega.sort_values("Riesgo_Operativo", ascending=False).iloc[0]
            peor_nombre = peor["Bodega_Origen"]
            resumen_bodega["Grupo"] = np.where(
                resumen_bodega["Bodega_Origen"] == peor_nombre,
                f"Prioridad: {peor_nombre}",
                "Otras bodegas",
            )

            fig = px.scatter(
                resumen_bodega, x="Antiguedad_Revision_Prom", y="Tasa_Ticket_Soporte",
                size="N", text="Bodega_Origen", color="Grupo",
                title="Bodegas con más días sin revisar y más tickets de soporte",
                labels={
                    "Antiguedad_Revision_Prom": "Días desde última revisión de stock",
                    "Tasa_Ticket_Soporte": "Proporción con ticket de soporte",
                },
                color_discrete_map={
                    f"Prioridad: {peor_nombre}": "#ef4444",
                    "Otras bodegas": "#94a3b8",
                },
            )
            fig.update_traces(textposition="top center")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            _nota_grafico(
                f"El punto rojo (**{peor_nombre}**) es la bodega citada en la respuesta: "
                "combina antigüedad de revisión y tasa de tickets."
            )

            top_bodegas = resumen_bodega.sort_values("Riesgo_Operativo", ascending=False).head(5)
            col1, col2 = st.columns(2)
            with col1:
                fig_ant = px.bar(
                    top_bodegas.sort_values("Antiguedad_Revision_Prom"),
                    x="Antiguedad_Revision_Prom", y="Bodega_Origen", orientation="h",
                    title="Días sin revisión física (top 5 riesgo)",
                    labels={"Antiguedad_Revision_Prom": "Días", "Bodega_Origen": "Bodega"},
                    color="Bodega_Origen",
                )
                fig_ant.update_layout(showlegend=False)
                st.plotly_chart(fig_ant, use_container_width=True)
                _nota_grafico("Cuanto más días sin contar stock, mayor riesgo de operar 'a ciegas'.")
            with col2:
                fig_tkt = px.bar(
                    top_bodegas.sort_values("Tasa_Ticket_Soporte"),
                    x="Tasa_Ticket_Soporte", y="Bodega_Origen", orientation="h",
                    title="Tasa de tickets de soporte (top 5 riesgo)",
                    labels={"Tasa_Ticket_Soporte": "Proporción con ticket", "Bodega_Origen": "Bodega"},
                    color="Bodega_Origen",
                )
                fig_tkt.update_xaxes(tickformat=".0%")
                fig_tkt.update_layout(showlegend=False)
                st.plotly_chart(fig_tkt, use_container_width=True)
                _nota_grafico(
                    "Valida que la bodega sin control también genera más reclamos de clientes."
                )

            conclusiones_globales.append(_conclusion(
                "Hay bodegas que no revisan stock y generan más incidencias",
                f"**{peor_nombre}** lleva ~{peor['Antiguedad_Revision_Prom']:.0f} días "
                f"sin revisión y {peor['Tasa_Ticket_Soporte']:.1%} de opiniones con ticket abierto "
                f"({int(peor['N']):,} transacciones en el filtro).",
                "Operar sin contar inventario produce promesas de stock falsas, retrasos "
                "y clientes que terminan en soporte.",
                "Conteo físico urgente + alinear sistema vs. realidad antes de nuevas campañas.",
            ))

    st.divider()
    _mostrar_conclusiones("📋 Respuestas y recomendaciones para la gerencia", conclusiones_globales)


# ============================================================================
# FILTROS GLOBALES SOBRE LA FUENTE ÚNICA DE VERDAD
# ============================================================================

SIN_DATO = "(Sin dato)"


def aplicar_filtros_maestro(df):
    """
    Renderiza los filtros globales en el sidebar y retorna el df filtrado.

    Los nulos de Categoria/Bodega_Origen (SKU fantasma) y de Ciudad_Destino
    (canal de venta contaminado el origen) se rellenan con la etiqueta
    '(Sin dato)' ANTES de armar las opciones del filtro. Así quedan como una
    categoría explícita, seleccionada por defecto: con la versión anterior,
    dejar los multiselect en "todos" igual excluía silenciosamente ~13 % de
    las filas (las de Ciudad_Destino nula), subestimando ingresos y pérdidas
    en el dashboard frente al total real del dataset.
    """
    st.sidebar.markdown("### 🔍 Filtros - Fuente Única de Verdad")

    df_filtrado = df.copy()
    columnas_filtro = ["Categoria", "Bodega_Origen", "Ciudad_Destino", "Canal_Venta"]
    for col in columnas_filtro:
        df_filtrado[col] = df_filtrado[col].fillna(SIN_DATO)

    categorias = sorted(df_filtrado["Categoria"].unique().tolist())
    bodegas = sorted(df_filtrado["Bodega_Origen"].unique().tolist())
    ciudades = sorted(df_filtrado["Ciudad_Destino"].unique().tolist())
    canales = sorted(df_filtrado["Canal_Venta"].unique().tolist())

    f_categorias = st.sidebar.multiselect("Categoría:", categorias, default=categorias)
    f_bodegas = st.sidebar.multiselect("Bodega de origen:", bodegas, default=bodegas)
    f_ciudades = st.sidebar.multiselect("Ciudad destino:", ciudades, default=ciudades)
    f_canales = st.sidebar.multiselect("Canal de venta:", canales, default=canales)
    incluir_sku_fantasma = st.sidebar.checkbox(
        "Incluir ventas de SKU fantasma (sin inventario)", value=True
    )

    df_filtrado = df_filtrado[
        df_filtrado["Categoria"].isin(f_categorias)
        & df_filtrado["Bodega_Origen"].isin(f_bodegas)
        & df_filtrado["Ciudad_Destino"].isin(f_ciudades)
        & df_filtrado["Canal_Venta"].isin(f_canales)
    ]
    if not incluir_sku_fantasma:
        df_filtrado = df_filtrado[~df_filtrado["SKU_Fantasma"]]

    st.sidebar.caption(f"Filas tras filtrar: {len(df_filtrado):,} / {len(df):,}")
    return df_filtrado


# ============================================================================
# MÓDULO DE IA (GROQ / LLAMA 3)
# ============================================================================

LLAMA_MODEL = "llama-3.3-70b-versatile"
LLAMA_MODEL_FALLBACK = "llama-3.1-70b-versatile"


def _resolver_groq_api_key(sidebar_key):
    """API Key desde sidebar, secrets.toml o variable de entorno."""
    if sidebar_key and str(sidebar_key).strip():
        return str(sidebar_key).strip()
    try:
        clave = st.secrets.get("GROQ_API_KEY", "")
        if clave:
            return clave
    except (AttributeError, FileNotFoundError, KeyError):
        pass
    return os.environ.get("GROQ_API_KEY", "")


def _huella_filtros(df):
    """Hash del subconjunto filtrado para detectar cambios entre generaciones."""
    if df.empty:
        return "vacio"
    cols = ["Transaccion_ID", "Ingreso_Bruto", "Margen_Utilidad", "SKU_Fantasma"]
    presentes = [c for c in cols if c in df.columns]
    payload = f"{len(df)}|" + df[presentes].head(500).to_csv(index=False)
    return hashlib.md5(payload.encode()).hexdigest()


def generar_resumen_estadistico(df, n_total=None):
    """
    Resumen estadístico estructurado del subconjunto filtrado.
    Alimenta el prompt de Llama-3 con métricas alineadas a las 5 preguntas del reto.
    """
    n = len(df)
    pct_muestra = (n / n_total * 100) if n_total else 100.0

    ingreso_total = float(df["Ingreso_Bruto"].sum())
    ingreso_riesgo = float(df["Ingreso_En_Riesgo"].sum())
    pct_riesgo = (ingreso_riesgo / ingreso_total * 100) if ingreso_total else 0.0

    df_margen = df[~df["SKU_Fantasma"]] if "SKU_Fantasma" in df.columns else df
    margen_pct_prom = float(df_margen["Margen_Utilidad_Pct"].mean()) if len(df_margen) else 0.0
    df_neg = df[df["Margen_Utilidad"] < 0]
    n_margen_negativo = len(df_neg)
    perdida_total = float(df_neg["Margen_Utilidad"].sum())
    pct_neg_online = (df_neg["Canal_Venta"] == "Online").mean() * 100 if n_margen_negativo else 0.0
    pct_online_base = (df["Canal_Venta"] == "Online").mean() * 100

    entrega_prom = float(df["Tiempo_Entrega_Real"].mean())
    pct_tardias = float(df["Entrega_Tardia"].mean() * 100)
    pct_sku_fantasma = float(df["SKU_Fantasma"].mean() * 100)

    df_fb = df[df["Tiene_Feedback"]] if "Tiene_Feedback" in df.columns else df.iloc[0:0]
    nps_prom = float(df_fb["Satisfaccion_NPS"].mean()) if len(df_fb) else float("nan")
    pct_feedback = len(df_fb) / n * 100 if n else 0.0
    pct_detractor = (
        (df_fb["Segmento_NPS"] == "Detractor").mean() * 100
        if len(df_fb) and "Segmento_NPS" in df_fb.columns else 0.0
    )
    pct_ticket = (
        df_fb["Ticket_Soporte_Abierto"].mean() * 100
        if len(df_fb) and "Ticket_Soporte_Abierto" in df_fb.columns else 0.0
    )

    top_categorias_margen = (
        df_margen.groupby("Categoria")["Margen_Utilidad_Pct"]
        .mean().sort_values().head(3)
    )
    top_ciudades_entrega = (
        df.groupby("Ciudad_Destino")["Tiempo_Entrega_Real"]
        .mean().sort_values(ascending=False).head(3)
    )
    top_ciudades_tardias = (
        df.groupby("Ciudad_Destino")["Entrega_Tardia"]
        .mean().sort_values(ascending=False).head(3) * 100
    )

    por_canal_ingreso = df.groupby("Canal_Venta")["Ingreso_Bruto"].sum().sort_values(ascending=False)
    por_canal_riesgo = df.groupby("Canal_Venta")["Ingreso_En_Riesgo"].sum().sort_values(ascending=False)

    resumen_cat = (
        df_margen.groupby("Categoria")
        .agg(
            Ratio_Stock=("Ratio_Stock_Reorden", "mean"),
            NPS=("Satisfaccion_NPS", "mean"),
        )
        .dropna(subset=["NPS"])
    )
    lineas_paradoja = ""
    if not resumen_cat.empty:
        med_s, med_n = resumen_cat["Ratio_Stock"].median(), resumen_cat["NPS"].median()
        paradoja = resumen_cat[
            (resumen_cat["Ratio_Stock"] >= med_s) & (resumen_cat["NPS"] < med_n)
        ]
        if not paradoja.empty:
            lineas_paradoja = paradoja.index.tolist()

    antig_bod = df.groupby("Bodega_Origen")["Antiguedad_Revision_Dias"].mean().sort_values(ascending=False)
    peor_bodega = antig_bod.index[0] if len(antig_bod) else "N/D"
    peor_bodega_dias = float(antig_bod.iloc[0]) if len(antig_bod) else 0.0

    filtros = {
        "categorias": int(df["Categoria"].nunique()) if "Categoria" in df.columns else 0,
        "bodegas": int(df["Bodega_Origen"].nunique()) if "Bodega_Origen" in df.columns else 0,
        "ciudades": int(df["Ciudad_Destino"].nunique()) if "Ciudad_Destino" in df.columns else 0,
        "canales": ", ".join(sorted(df["Canal_Venta"].dropna().astype(str).unique()[:8])),
    }

    resumen = f"""
RESUMEN ESTADÍSTICO — TECHLOGISTICS S.A.S.
(Subconjunto filtrado por el usuario en el dashboard)

=== ALCANCE DEL FILTRO ===
Transacciones analizadas: {n:,} ({pct_muestra:.1f}% del universo de {n_total:,} filas).
Categorías distintas: {filtros['categorias']} | Bodegas: {filtros['bodegas']} | Ciudades: {filtros['ciudades']}
Canales de venta presentes: {filtros['canales'] or 'N/D'}

=== P1 · FUGA DE CAPITAL Y RENTABILIDAD ===
Ingreso bruto total: USD {ingreso_total:,.2f}
Margen promedio (% sobre precio, excl. SKU fantasma del cálculo de costo): {margen_pct_prom:.1f}%
Transacciones con margen negativo: {n_margen_negativo:,} (pérdida acumulada USD {abs(perdida_total):,.2f})
De esas ventas en pérdida, {pct_neg_online:.1f}% ocurre en canal Online (vs. {pct_online_base:.1f}% de participación general de Online).
Categorías con menor margen promedio:
{top_categorias_margen.to_string() if not top_categorias_margen.empty else '  (sin datos)'}

=== P2 · CRISIS LOGÍSTICA ===
Tiempo de entrega promedio: {entrega_prom:.1f} días (SLA objetivo: {SLA_ENTREGA_DIAS} días).
Entregas tardías (incumplen SLA): {pct_tardias:.1f}%
Ciudades con mayor tiempo de entrega promedio:
{top_ciudades_entrega.to_string() if not top_ciudades_entrega.empty else '  (sin datos)'}
Ciudades con mayor % de entregas tardías:
{top_ciudades_tardias.to_string() if not top_ciudades_tardias.empty else '  (sin datos)'}

=== P3 · VENTA INVISIBLE (SKU FANTASMA) ===
Ventas sin SKU en inventario: {pct_sku_fantasma:.1f}% de las transacciones filtradas.
Ingreso en riesgo (sin respaldo de inventario): USD {ingreso_riesgo:,.2f} ({pct_riesgo:.1f}% del ingreso bruto).
Ingreso por canal (top):
{por_canal_ingreso.head(3).to_string() if not por_canal_ingreso.empty else '  (sin datos)'}
Ingreso en riesgo por canal (top):
{por_canal_riesgo.head(3).to_string() if not por_canal_riesgo.empty else '  (sin datos)'}

=== P4 · FIDELIDAD Y FEEDBACK ===
Transacciones con opinión de cliente: {pct_feedback:.1f}% ({len(df_fb):,} registros).
NPS promedio (solo con feedback): {nps_prom:.1f} (escala -100 a 100).
Detractores: {pct_detractor:.1f}% del feedback | Tickets de soporte abiertos: {pct_ticket:.1f}%
Categorías con alto stock y NPS bajo (paradoja): {', '.join(lineas_paradoja) if lineas_paradoja else 'ninguna detectada'}

=== P5 · RIESGO OPERATIVO EN BODEGAS ===
Bodega con revisión de stock más antigua: {peor_bodega} (~{peor_bodega_dias:.0f} días sin revisión).
Antigüedad promedio de revisión por bodega (top 3):
{antig_bod.head(3).to_string() if len(antig_bod) else '  (sin datos)'}
"""
    return resumen.strip()


def generar_recomendaciones_ia(resumen_estadistico, api_key, temperature=0.4, model=LLAMA_MODEL):
    """Streaming con Llama-3 (Groq): exactamente 3 párrafos estratégicos."""
    from groq import Groq

    client = Groq(api_key=api_key)
    prompt = f"""Eres consultor senior de datos ante la junta directiva de TechLogistics S.A.S.
Analiza ÚNICAMENTE el resumen estadístico siguiente (proviene del subconjunto que el usuario filtró en el dashboard).

{resumen_estadistico}

INSTRUCCIONES DE FORMATO (obligatorio):
- Responde en español, tono ejecutivo y directo.
- Escribe EXACTAMENTE 3 secciones con estos encabezados markdown (sin numeración extra ni bullet lists):
## Diagnóstico
(un párrafo de 4-6 oraciones citando cifras concretas del resumen)

## Riesgo prioritario
(un párrafo de 4-6 oraciones: el problema más urgente y por qué, con datos)

## Acción recomendada
(un párrafo de 4-6 oraciones: qué debe hacer la junta en los próximos 90 días)

REGLAS:
- No inventes cifras que no aparezcan en el resumen.
- No agregues secciones extra, tablas ni listas.
- Cada sección debe ser un solo párrafo continuo."""

    return client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un consultor de ciencia de datos especializado en retail y logística. "
                    "Respondes solo con las 3 secciones solicitadas, basándote en evidencia numérica."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=1400,
        stream=True,
    )


def _iter_tokens_stream(stream):
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _parsear_tres_parrafos(texto):
    """Extrae Diagnóstico, Riesgo y Acción del markdown devuelto por Llama."""
    secciones = {
        "Diagnóstico": "",
        "Riesgo prioritario": "",
        "Acción recomendada": "",
    }
    clave_actual = None
    for linea in texto.splitlines():
        linea_l = linea.strip().lower()
        if linea_l.startswith("## diagn"):
            clave_actual = "Diagnóstico"
            continue
        if linea_l.startswith("## riesgo"):
            clave_actual = "Riesgo prioritario"
            continue
        if linea_l.startswith("## acci"):
            clave_actual = "Acción recomendada"
            continue
        if clave_actual and linea.strip():
            secciones[clave_actual] += (linea.strip() + " ")
    for k in secciones:
        secciones[k] = secciones[k].strip()
    if not any(secciones.values()):
        partes = [p.strip() for p in re.split(r"\n\s*\n", texto.strip()) if p.strip()]
        claves = list(secciones.keys())
        for i, parte in enumerate(partes[:3]):
            secciones[claves[i]] = re.sub(r"^#+\s*", "", parte)
    return secciones


def _mostrar_recomendacion_estructurada(texto):
    """Presenta los 3 párrafos en tarjetas legibles."""
    partes = _parsear_tres_parrafos(texto)
    iconos = {"Diagnóstico": "🔍", "Riesgo prioritario": "⚠️", "Acción recomendada": "➡️"}
    for titulo, cuerpo in partes.items():
        with st.container(border=True):
            st.markdown(f"### {iconos.get(titulo, '📌')} {titulo}")
            if cuerpo:
                st.markdown(cuerpo)
            else:
                st.caption("_Sección no detectada en la respuesta del modelo._")


def _stream_llama_con_fallback(resumen, api_key, temperature):
    """Generador de tokens con fallback de modelo Llama-3."""
    ultimo_error = None
    for model in (LLAMA_MODEL, LLAMA_MODEL_FALLBACK):
        try:
            stream = generar_recomendaciones_ia(resumen, api_key, temperature, model=model)
            for token in _iter_tokens_stream(stream):
                yield token
            return
        except Exception as exc:
            ultimo_error = exc
    if ultimo_error:
        raise ultimo_error


def _mostrar_stream_en_vivo(resumen, api_key, temperature):
    """Streaming en tiempo real compatible con todas las versiones de Streamlit."""
    gen = _stream_llama_con_fallback(resumen, api_key, temperature)
    placeholder = st.empty()
    texto = ""
    for token in gen:
        texto += token
        placeholder.markdown(texto + "▌")
    placeholder.markdown(texto)
    return texto


def render_recomendaciones_ia(df_maestro, df_completo, api_key_sidebar):
    """Página del módulo de IA: resumen filtrado → Llama-3 → 3 párrafos en tiempo real."""
    st.title("🤖 Recomendaciones Estratégicas con IA")
    st.markdown(
        f"El modelo **{LLAMA_MODEL}** (familia **Llama-3**, vía Groq) analiza el "
        "**resumen estadístico** del subconjunto que filtraste en el sidebar y genera "
        "**tres párrafos** de recomendación para la junta directiva, en **tiempo real**."
    )

    api_key = _resolver_groq_api_key(api_key_sidebar)
    if not api_key:
        st.warning("Ingresa tu **Groq API Key** en el sidebar para activar el módulo.")
        st.info(
            "Obtén una clave gratuita en [console.groq.com](https://console.groq.com/). "
            "También puedes definir `GROQ_API_KEY` en `.streamlit/secrets.toml`."
        )
        return

    if df_maestro.empty:
        st.warning("No hay transacciones con los filtros actuales. Amplía la selección en el sidebar.")
        return

    resumen = generar_resumen_estadistico(df_maestro, n_total=len(df_completo))
    huella = _huella_filtros(df_maestro)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    ingreso = df_maestro["Ingreso_Bruto"].sum()
    riesgo = df_maestro["Ingreso_En_Riesgo"].sum()
    c1.metric("Transacciones", f"{len(df_maestro):,}")
    c2.metric("Ingreso bruto", f"USD {ingreso:,.0f}")
    c3.metric("Ingreso en riesgo", f"USD {riesgo:,.0f}")
    c4.metric("Margen negativo", f"{(df_maestro['Margen_Utilidad'] < 0).sum():,}")
    c5.metric("Entregas tardías", f"{df_maestro['Entrega_Tardia'].mean():.1%}")
    df_fb = df_maestro[df_maestro["Tiene_Feedback"]]
    nps = df_fb["Satisfaccion_NPS"].mean() if len(df_fb) else float("nan")
    c6.metric("NPS promedio", f"{nps:.1f}" if pd.notna(nps) else "N/D")

    with st.expander("📄 Resumen estadístico enviado a Llama-3 (datos filtrados)", expanded=False):
        st.code(resumen, language="text")
        st.caption(
            "Este texto es la única entrada del modelo. Si cambias filtros en el sidebar, "
            "vuelve a generar para actualizar las recomendaciones."
        )

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        temperature = st.select_slider(
            "Creatividad del modelo",
            options=[0.2, 0.3, 0.4, 0.5, 0.7],
            value=0.4,
            help="Valores bajos = más fiel a las cifras. Valores altos = redacción más variada.",
        )
    with col_cfg2:
        st.markdown(
            f"**Modelo:** `{LLAMA_MODEL}`  \n"
            f"**Proveedor:** Groq (inferencia en streaming)  \n"
            f"**Cobertura:** {len(df_maestro):,} / {len(df_completo):,} transacciones"
        )

    if (
        st.session_state.get("ia_huella_filtros")
        and st.session_state["ia_huella_filtros"] != huella
        and st.session_state.get("ultima_recomendacion_ia")
    ):
        st.warning("Los filtros cambiaron desde la última generación. Pulsa el botón para actualizar el análisis.")

    generar = st.button("✨ Generar 3 párrafos estratégicos", type="primary", use_container_width=True)

    if generar:
        try:
            with st.status("Conectando con Llama-3 vía Groq…", expanded=True) as status:
                st.write("Enviando resumen estadístico del subconjunto filtrado…")
                texto_completo = _mostrar_stream_en_vivo(resumen, api_key, temperature)
                status.update(label="Recomendaciones generadas", state="complete")

            st.session_state["ultima_recomendacion_ia"] = texto_completo
            st.session_state["ia_huella_filtros"] = huella
            st.session_state["ia_resumen_enviado"] = resumen

        except Exception as e:
            st.error(f"Error al conectar con Groq / Llama-3: {e}")
            if "invalid" in str(e).lower() or "auth" in str(e).lower():
                st.warning("Verifica que la API Key sea válida y tenga cuota disponible.")

    if st.session_state.get("ultima_recomendacion_ia"):
        st.divider()
        st.subheader("Recomendación para la junta directiva")
        _mostrar_recomendacion_estructurada(st.session_state["ultima_recomendacion_ia"])
        st.download_button(
            "💾 Descargar recomendación (.txt)",
            data=st.session_state["ultima_recomendacion_ia"],
            file_name="recomendacion_estrategica_techlogistics.txt",
            mime="text/plain",
            use_container_width=True,
        )


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

pagina = render_sidebar_navigation()

st.sidebar.divider()
render_sidebar_stats(calcular_metricas(datasets))

st.sidebar.divider()
st.sidebar.markdown("### 🔑 IA (Groq · Llama-3)")
groq_api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    help="https://console.groq.com/ — o usa GROQ_API_KEY en secrets.toml",
    placeholder="gsk_...",
)
if _resolver_groq_api_key("") and not groq_api_key:
    st.sidebar.caption("✓ API Key cargada desde secrets / entorno")

st.sidebar.divider()
render_sidebar_descargas()

# Filtros globales: solo se muestran (y aplican) en las páginas que
# consumen la Fuente Única de Verdad.
paginas_con_filtro = {"📌 Preguntas Estratégicas", "🤖 Recomendaciones IA"}
if pagina in paginas_con_filtro:
    st.sidebar.divider()
    df_maestro = aplicar_filtros_maestro(df_maestro_completo)
else:
    df_maestro = df_maestro_completo

metricas = calcular_metricas(datasets)

# ============================================================================
# PÁGINA 1: DASHBOARD PRINCIPAL
# ============================================================================

if pagina == "📈 Dashboard Principal":
    st.title("📊 TechLogistics - Data Hub & Sistema de Soporte a la Decisión")
    st.markdown(
        "Use el **sidebar** para navegar, filtrar (en Preguntas Estratégicas / IA) "
        "y **descargar reportes de limpieza**. Cada sección principal usa **pestañas** "
        "para organizar el análisis."
    )

    tab_resumen, tab_calidad, tab_datos = st.tabs(
        ["🏠 Resumen ejecutivo", "🩺 Calidad y trazabilidad", "📋 Vista previa de datos"]
    )

    with tab_resumen:
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

        st.markdown("#### Navegación rápida del DSS")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("**📌 Preguntas Estratégicas** — 5 hallazgos con gráficos y conclusiones.")
        with c2:
            st.info("**🤖 Recomendaciones IA** — Llama-3 sobre datos filtrados.")
        with c3:
            if REPORT_PDF.exists():
                st.success(
                    f"**PDF de consultoría** disponible en la raíz: `{REPORT_PDF.name}` "
                    "(también en sidebar → Descargas)."
                )
            else:
                st.warning("Genere el PDF con `make report` o `python scripts/generate_report.py`.")

    with tab_calidad:
        render_trazabilidad_limpieza()
        st.divider()
        st.subheader("🩺 Detalle por dataset — Health Score, nulidad y logs")
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

                    resultado_nulidad = resumen_nulidad_original(nombre)
                    if resultado_nulidad is not None:
                        resumen_nulos, duplicados, n_original = resultado_nulidad
                        with st.expander(
                            f"🔎 Nulidad y duplicados (crudo, {n_original:,} filas)"
                        ):
                            st.write(f"**Duplicados exactos detectados:** {duplicados:,}")
                            if not resumen_nulos.empty:
                                st.dataframe(resumen_nulos, use_container_width=True, hide_index=True)
                            else:
                                st.caption("Sin columnas con valores nulos en el dataset crudo.")
                else:
                    st.info("Ejecuta `python scripts/run_pipeline.py` para generar el Health Score.")

        st.subheader("📥 Reportes de limpieza")
        col1, col2, col3 = st.columns(3)
        with col1:
            df_log_consolidado = construir_reporte_limpieza_consolidado()
            if df_log_consolidado is not None:
                st.download_button(
                    "💾 Log de limpieza (CSV)",
                    data=df_log_consolidado.to_csv(index=False).encode("utf-8"),
                    file_name="log_limpieza_consolidado.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dash_log_csv",
                )
        with col2:
            df_health_consolidado = construir_health_score_consolidado()
            if df_health_consolidado is not None:
                st.download_button(
                    "💾 Health Score (CSV)",
                    data=df_health_consolidado.to_csv(index=False).encode("utf-8"),
                    file_name="health_score_consolidado.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dash_health_csv",
                )
        with col3:
            st.download_button(
                "💾 Trazabilidad ética (TXT)",
                data=construir_informe_trazabilidad_txt().encode("utf-8"),
                file_name="informe_trazabilidad_limpieza.txt",
                mime="text/plain",
                use_container_width=True,
                key="dash_traz_txt",
            )

    with tab_datos:
        st.subheader("👁️ Vista previa de datasets limpios")
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Transacciones", "Inventario", "Feedback", "Fuente Única de Verdad"]
        )
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
    render_preguntas_estrategicas(df_maestro)


# ============================================================================
# PÁGINA 3: RECOMENDACIONES IA (Groq / Llama 3.3)
# ============================================================================

elif pagina == "🤖 Recomendaciones IA":
    render_recomendaciones_ia(df_maestro, df_maestro_completo, groq_api_key)


# ============================================================================
# PÁGINA 4: TRANSACCIONES
# ============================================================================

elif pagina == "📦 Transacciones":
    st.title("📦 Análisis de Transacciones Logísticas")
    df = datasets["transacciones"]
    render_analisis_transacciones(df)

    with st.expander("🔬 Exploración técnica (describe y datos completos)"):
        st.dataframe(df.describe(), use_container_width=True)
        st.dataframe(df, use_container_width=True)


# ============================================================================
# PÁGINA 5: INVENTARIO
# ============================================================================

elif pagina == "🏭 Inventario":
    st.title("🏭 Análisis de Inventario Central")
    df = datasets["inventario"]
    render_analisis_inventario(df)

    with st.expander("🔬 Exploración técnica (describe y datos completos)"):
        st.dataframe(df.describe(), use_container_width=True)
        st.dataframe(df, use_container_width=True)


# ============================================================================
# PÁGINA 6: FEEDBACK
# ============================================================================

elif pagina == "💬 Feedback":
    st.title("💬 Análisis de Feedback de Clientes")
    df = datasets["feedback"]
    render_analisis_feedback(df, n_transacciones=len(datasets["transacciones"]))

    with st.expander("🔬 Exploración técnica (describe y datos completos)"):
        st.dataframe(df.describe(), use_container_width=True)
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
