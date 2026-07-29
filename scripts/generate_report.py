"""
generate_report.py
------------------
Genera el PDF de hallazgos para la junta directiva.

Uso:
    python scripts/generate_report.py

Salida:
    Informe_Consultoria_TechLogistics_Junta_Directiva_Hallazgos_Estrategicos.pdf
    (raíz del proyecto; copia en reports/deliverables/)
"""

import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from techlogistics.config import (
    HEALTH_SCORE_POR_DATASET,
    REPORT_PDF,
    REPORT_PDF_COPY,
    REPORTS_DELIVERABLES,
    SLA_ENTREGA_DIAS,
    ensure_dirs,
)
from techlogistics.pipelines.integration import construir_fuente_unica

plt.rcParams.update({"figure.autolayout": True, "font.size": 9})


def _figura_a_imagen(fig, ancho_cm=16):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    ancho = ancho_cm * cm
    alto = ancho * 0.55
    return Image(buf, width=ancho, height=alto)


def grafico_q1(df):
    df_neg = df[df["Margen_Utilidad"] < 0]
    top_skus = df_neg.groupby("SKU_ID")["Margen_Utilidad"].sum().sort_values().head(10)

    fig, ax = plt.subplots(figsize=(8, 4))
    top_skus.plot(kind="barh", ax=ax, color="firebrick")
    ax.set_xlabel("Margen acumulado (USD)")
    ax.set_title("Top 10 SKUs con mayor pérdida acumulada")
    ax.invert_yaxis()
    return _figura_a_imagen(fig)


def grafico_q2(df):
    df_fb = df[df["Tiene_Feedback"]].copy()
    por_bodega = df_fb.groupby("Bodega_Origen")["Entrega_Tardia"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    (por_bodega * 100).plot(kind="barh", ax=ax, color="darkorange")
    ax.set_xlabel(f"% de entregas tardías (> SLA {SLA_ENTREGA_DIAS}d)")
    ax.set_title("Cuellos de botella logísticos por bodega")
    ax.invert_yaxis()
    return _figura_a_imagen(fig)


def grafico_q3(df):
    por_mes = df.groupby("Mes_Venta")["Ingreso_En_Riesgo"].sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    por_mes.plot(kind="line", marker="o", ax=ax, color="purple")
    ax.set_ylabel("USD en riesgo")
    ax.set_xlabel("Mes de venta")
    ax.set_title("Ingreso en riesgo por SKU fantasma, por mes")
    ax.tick_params(axis="x", rotation=60)
    return _figura_a_imagen(fig)


def grafico_q4(df):
    resumen_cat = (
        df.dropna(subset=["Categoria"])
        .groupby("Categoria")
        .agg(Ratio_Reorden=("Ratio_Stock_Reorden", "mean"), NPS=("Satisfaccion_NPS", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(resumen_cat["Ratio_Reorden"], resumen_cat["NPS"], s=120, color="teal")
    for _, fila in resumen_cat.iterrows():
        ax.annotate(
            fila["Categoria"],
            (fila["Ratio_Reorden"], fila["NPS"]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.axvline(resumen_cat["Ratio_Reorden"].median(), linestyle="--", color="gray")
    ax.axhline(resumen_cat["NPS"].median(), linestyle="--", color="gray")
    ax.set_xlabel("Stock / Punto de reorden (promedio)")
    ax.set_ylabel("NPS promedio")
    ax.set_title("Disponibilidad de stock vs. satisfacción por categoría")
    return _figura_a_imagen(fig)


def grafico_q5(df):
    df_fb = df[df["Tiene_Feedback"]]
    resumen = pd.DataFrame(
        {
            "Antiguedad": df.groupby("Bodega_Origen")["Antiguedad_Revision_Dias"].mean(),
            "Tasa_Soporte": df_fb.groupby("Bodega_Origen")["Ticket_Soporte_Abierto"].mean(),
        }
    ).dropna().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(resumen["Antiguedad"], resumen["Tasa_Soporte"] * 100, s=120, color="crimson")
    for _, fila in resumen.iterrows():
        ax.annotate(
            fila["Bodega_Origen"],
            (fila["Antiguedad"], fila["Tasa_Soporte"] * 100),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Antigüedad promedio de revisión (días)")
    ax.set_ylabel("Tasa de tickets de soporte (%)")
    ax.set_title("Riesgo operativo: revisión de stock vs. tickets de soporte")
    return _figura_a_imagen(fig)


def grafico_health_score_resumen():
    """Barras comparativas de Health Score antes/después (los 3 datasets)."""
    filas = []
    for nombre, etiqueta in [
        ("transacciones", "Transacciones"),
        ("inventario", "Inventario"),
        ("feedback", "Feedback"),
    ]:
        ruta = HEALTH_SCORE_POR_DATASET.get(nombre)
        if ruta is not None and ruta.exists():
            df_h = pd.read_csv(ruta)
            t = df_h[df_h["metrica"] == "health_score_total"].iloc[0]
            filas.append({"Dataset": etiqueta, "Antes": t["antes"], "Después": t["despues"]})
    if not filas:
        return None
    df_plot = pd.DataFrame(filas)
    x = range(len(df_plot))
    ancho = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - ancho / 2 for i in x], df_plot["Antes"], width=ancho, label="Antes", color="#94a3b8")
    ax.bar([i + ancho / 2 for i in x], df_plot["Después"], width=ancho, label="Después", color="#2563eb")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df_plot["Dataset"])
    ax.set_ylabel("Health Score (0-100)")
    ax.set_title("Calidad de datos: Health Score antes y después de la limpieza")
    ax.legend()
    ax.set_ylim(0, 105)
    return _figura_a_imagen(fig)


def calcular_hallazgos(df):
    hallazgos = {}

    df_neg = df[df["Margen_Utilidad"] < 0]
    pct_online_neg = (df_neg["Canal_Venta"] == "Online").mean() * 100 if not df_neg.empty else 0
    pct_online_base = (df["Canal_Venta"] == "Online").mean() * 100
    hallazgos["q1"] = dict(
        n_sku=df_neg["SKU_ID"].nunique(),
        perdida=df_neg["Margen_Utilidad"].sum(),
        pct_online_neg=pct_online_neg,
        pct_online_base=pct_online_base,
    )

    df_fb = df[df["Tiene_Feedback"]].copy()
    df_fb["Zona"] = df_fb["Ciudad_Destino"] + " · " + df_fb["Bodega_Origen"]
    filas = []
    for zona, g in df_fb.groupby("Zona"):
        if len(g) >= 8 and g["Tiempo_Entrega_Real"].nunique() > 1:
            corr = g["Tiempo_Entrega_Real"].corr(g["Satisfaccion_NPS"])
            if pd.notna(corr):
                filas.append((zona, corr, len(g)))
    filas.sort(key=lambda x: x[1])
    hallazgos["q2"] = dict(peor_zona=filas[0] if filas else None)

    ingreso_total = df["Ingreso_Bruto"].sum()
    ingreso_riesgo = df["Ingreso_En_Riesgo"].sum()
    hallazgos["q3"] = dict(
        ingreso_total=ingreso_total,
        ingreso_riesgo=ingreso_riesgo,
        pct_riesgo=(ingreso_riesgo / ingreso_total * 100) if ingreso_total else 0,
    )

    resumen_cat = (
        df.dropna(subset=["Categoria"])
        .groupby("Categoria")
        .agg(
            Ratio_Reorden=("Ratio_Stock_Reorden", "mean"),
            NPS=("Satisfaccion_NPS", "mean"),
            Margen=("Margen_Utilidad_Pct", "mean"),
        )
        .reset_index()
    )
    mediana_stock = resumen_cat["Ratio_Reorden"].median()
    mediana_nps = resumen_cat["NPS"].median()
    paradoja = resumen_cat[
        (resumen_cat["Ratio_Reorden"] >= mediana_stock) & (resumen_cat["NPS"] < mediana_nps)
    ]
    hallazgos["q4"] = dict(categorias=paradoja["Categoria"].tolist() if not paradoja.empty else [])

    antiguedad_bodega = df.groupby("Bodega_Origen")["Antiguedad_Revision_Dias"].mean()
    soporte_bodega = df_fb.groupby("Bodega_Origen")["Ticket_Soporte_Abierto"].mean()
    resumen_bodega = pd.DataFrame({"Antiguedad": antiguedad_bodega, "Soporte": soporte_bodega}).dropna()
    if not resumen_bodega.empty:
        peor_bodega = resumen_bodega.sort_values(["Antiguedad", "Soporte"], ascending=False).iloc[0]
        hallazgos["q5"] = dict(
            bodega=peor_bodega.name,
            antiguedad=peor_bodega["Antiguedad"],
            soporte=peor_bodega["Soporte"],
        )
    else:
        hallazgos["q5"] = dict(bodega=None)

    return hallazgos


def _tabla_health_score(nombre_dataset, styles):
    ruta = HEALTH_SCORE_POR_DATASET.get(nombre_dataset)
    if ruta is None or not ruta.exists():
        return Paragraph(f"(No se encontró health_score_{nombre_dataset}.csv)", styles["Normal"])

    df_health = pd.read_csv(ruta)
    fila = df_health[df_health["metrica"] == "health_score_total"].iloc[0]
    datos = [
        ["Dataset", "Antes", "Después", "Mejora"],
        [
            nombre_dataset.capitalize(),
            f"{fila['antes']:.1f}",
            f"{fila['despues']:.1f}",
            f"{fila['delta']:+.1f}",
        ],
    ]
    tabla = Table(datos, colWidths=[4 * cm, 3 * cm, 3 * cm, 3 * cm])
    tabla.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return tabla


def construir_informe():
    """Genera el PDF de consultoría en la raíz del proyecto."""
    ensure_dirs()
    REPORTS_DELIVERABLES.mkdir(parents=True, exist_ok=True)

    print("Construyendo la Fuente Única de Verdad para el informe...")
    df = construir_fuente_unica()
    if df.empty:
        raise ValueError("La Fuente Única de Verdad quedó vacía; ejecute primero run_pipeline.py.")

    hallazgos = calcular_hallazgos(df)
    ingreso_total = df["Ingreso_Bruto"].sum()
    pct_fantasma = df["SKU_Fantasma"].mean() * 100
    pct_tardias = df["Entrega_Tardia"].mean() * 100
    nps = df.loc[df["Tiene_Feedback"], "Satisfaccion_NPS"].mean()

    styles = getSampleStyleSheet()
    estilo_titulo = ParagraphStyle("TituloInforme", parent=styles["Title"], fontSize=20)
    estilo_h2 = ParagraphStyle("H2Informe", parent=styles["Heading2"], spaceBefore=14)
    estilo_cuerpo = ParagraphStyle("CuerpoInforme", parent=styles["BodyText"], spaceAfter=10, leading=15)

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=letter,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )
    story = []

    story.append(Paragraph("TechLogistics S.A.S.", estilo_titulo))
    story.append(Paragraph("Informe de Consultoría de Datos — Junta Directiva", styles["Heading2"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(
        Paragraph(
            "Diagnóstico de <b>invisibilidad operativa</b> entre Inventario, Logística y Feedback, "
            "con cinco hallazgos estratégicos y hoja de ruta para recuperar margen y lealtad. "
            "Challenge 02 — Fundamentos en Ciencia de Datos (Maestría), Universidad EAFIT, 2026-1.",
            estilo_cuerpo,
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Resumen ejecutivo", estilo_h2))
    story.append(
        Paragraph(
            f"Se integraron <b>{len(df):,}</b> transacciones en una Sola Fuente de Verdad. "
            f"El ingreso bruto analizado asciende a <b>USD {ingreso_total:,.0f}</b>. "
            f"<b>{pct_fantasma:.1f}%</b> de las ventas carece de respaldo en inventario (SKU fantasma); "
            f"<b>{pct_tardias:.1f}%</b> de entregas incumple el SLA de {SLA_ENTREGA_DIAS} días; "
            f"el NPS promedio donde hay feedback es <b>{nps:.1f}</b>. "
            "Las gráficas siguientes reproducen los mismos indicadores visualizados en el dashboard Streamlit.",
            estilo_cuerpo,
        )
    )
    story.append(Spacer(1, 0.8 * cm))

    story.append(Paragraph("Auditoría de calidad — Health Score", estilo_h2))
    fig_hs = grafico_health_score_resumen()
    if fig_hs is not None:
        story.append(fig_hs)
    for nombre in ["transacciones", "inventario", "feedback"]:
        story.append(_tabla_health_score(nombre, styles))
        story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "Metodología: Completitud 40 % + Unicidad 25 % + Consistencia 20 % + Validez 15 %. "
            "Toda transformación queda registrada en reports/quality/log_limpieza_*.csv.",
            estilo_cuerpo,
        )
    )
    story.append(PageBreak())

    h1 = hallazgos["q1"]
    story.append(Paragraph("1. Fuga de Capital y Rentabilidad", estilo_h2))
    story.append(grafico_q1(df))
    if h1["n_sku"] > 0:
        veredicto = (
            "una falla crítica de precios en el canal Online"
            if h1["pct_online_neg"] > h1["pct_online_base"] * 1.3
            else "una pérdida aceptable por volumen, concentrada en SKUs puntuales"
        )
        story.append(
            Paragraph(
                f"Se identificaron <b>{h1['n_sku']:,} SKUs</b> con margen negativo, por una pérdida "
                f"acumulada de <b>USD {h1['perdida']:,.2f}</b>. El {h1['pct_online_neg']:.1f}% de esas "
                f"ventas ocurre en el canal Online (frente a un {h1['pct_online_base']:.1f}% de "
                f"participación general de ese canal), lo que apunta a {veredicto}.",
                estilo_cuerpo,
            )
        )
    else:
        story.append(Paragraph("No se identificaron transacciones con margen negativo.", estilo_cuerpo))

    h2 = hallazgos["q2"]
    story.append(Paragraph("2. Crisis Logística y Cuellos de Botella", estilo_h2))
    story.append(grafico_q2(df))
    if h2["peor_zona"]:
        zona, corr, n = h2["peor_zona"]
        story.append(
            Paragraph(
                f"La zona <b>{zona}</b> muestra la correlación más negativa entre tiempo de entrega y "
                f"NPS ({corr:.2f}, n={n}): a mayor demora, más cae la satisfacción. Es la zona "
                "prioritaria para un cambio inmediato de operador logístico.",
                estilo_cuerpo,
            )
        )
    else:
        story.append(Paragraph("No hay suficientes datos con feedback para aislar una zona crítica.", estilo_cuerpo))

    h3 = hallazgos["q3"]
    story.append(Paragraph("3. Análisis de la Venta Invisible", estilo_h2))
    story.append(grafico_q3(df))
    story.append(
        Paragraph(
            f"El <b>{h3['pct_riesgo']:.2f}%</b> del ingreso bruto (USD {h3['ingreso_riesgo']:,.2f} de un "
            f"total de USD {h3['ingreso_total']:,.2f}) corresponde a ventas de SKUs sin respaldo en el "
            "maestro de inventario. Es dinero que hoy opera sin control de costo ni trazabilidad.",
            estilo_cuerpo,
        )
    )
    story.append(PageBreak())

    h4 = hallazgos["q4"]
    story.append(Paragraph("4. Diagnóstico de Fidelidad", estilo_h2))
    story.append(grafico_q4(df))
    if h4["categorias"]:
        story.append(
            Paragraph(
                f"Las categorías <b>{', '.join(h4['categorias'])}</b> combinan alta disponibilidad de "
                "stock con satisfacción por debajo de la mediana: la paradoja de tener producto "
                "disponible pero no vendido con buena experiencia de cliente.",
                estilo_cuerpo,
            )
        )
    else:
        story.append(Paragraph("No se detecta la paradoja de alto stock + bajo NPS en los datos actuales.", estilo_cuerpo))

    h5 = hallazgos["q5"]
    story.append(Paragraph("5. Storytelling de Riesgo Operativo", estilo_h2))
    story.append(grafico_q5(df))
    if h5["bodega"]:
        story.append(
            Paragraph(
                f"La bodega <b>{h5['bodega']}</b> combina la mayor antigüedad de revisión de stock "
                f"({h5['antiguedad']:.0f} días) con una alta tasa de tickets de soporte "
                f"({h5['soporte']:.1%}): es la bodega que más 'opera a ciegas' sobre su propio "
                "inventario, con impacto directo en la satisfacción final del cliente.",
                estilo_cuerpo,
            )
        )
    else:
        story.append(Paragraph("No hay suficientes datos cruzados de inventario y soporte por bodega.", estilo_cuerpo))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Recomendaciones generales para la junta", estilo_h2))
    story.append(
        Paragraph(
            "<b>1.</b> Bloquear ventas de SKU no catalogados y auditar altas en inventario. "
            "<b>2.</b> Renegociar o sustituir operadores logísticos en la zona con peor "
            "correlación entrega–NPS. <b>3.</b> Establecer conteo cíclico obligatorio en "
            "bodegas con mayor antigüedad de revisión. <b>4.</b> Recalibrar precios en canal "
            "Online si concentra márgenes negativos. <b>5.</b> Usar el dashboard DSS para "
            "monitoreo continuo con trazabilidad de imputaciones.",
            estilo_cuerpo,
        )
    )

    doc.build(story)

    import shutil
    shutil.copy2(REPORT_PDF, REPORT_PDF_COPY)
    print(f"\nInforme generado (raíz): {os.path.normpath(REPORT_PDF)}")
    print(f"Copia en:              {os.path.normpath(REPORT_PDF_COPY)}")
    return REPORT_PDF


if __name__ == "__main__":
    try:
        construir_informe()
    except FileNotFoundError as exc:
        print(f"\nERROR — archivo no encontrado: {exc}")
        print("Ejecute primero: python scripts/run_pipeline.py")
        sys.exit(1)
    except (ValueError, PermissionError, OSError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR inesperado al generar el PDF: {exc}")
        sys.exit(1)
