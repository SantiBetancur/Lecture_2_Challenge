"""
feedback.py
-----------
Pipeline de auditoría, limpieza y feature engineering del dataset
`feedback_clientes_v2.csv`.

Challenge 02 - Fundamentos en Ciencia de Datos (Maestría) - EAFIT 2026-1

Salidas generadas en ../data y ../reports:
    - feedback_limpio.csv           Dataset curado
    - log_limpieza_feedback.csv     Trazabilidad de cada transformación
    - health_score_feedback.csv     Health Score antes vs. después
"""

import os
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import (
        LogLimpieza,
        cargar_datos,
        check_data_quality,
        comparar_health_scores,
        imprimir_health_score,
    )
else:
    from .common import (
        LogLimpieza,
        cargar_datos,
        check_data_quality,
        comparar_health_scores,
        imprimir_health_score,
    )

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")

DATA_PATH = os.path.join(DATA_DIR, "feedback_clientes_v2.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "feedback_limpio.csv")
LOG_PATH = os.path.join(REPORTS_DIR, "log_limpieza_feedback.csv")
HEALTH_PATH = os.path.join(REPORTS_DIR, "health_score_feedback.csv")

RATING_MIN, RATING_MAX = 1, 5
EDAD_MAXIMA_PLAUSIBLE = 100

# Marcadores de "sin respuesta" que contaminan Comentario_Texto como texto
# libre en vez de quedar como nulo real.
COMENTARIO_PLACEHOLDERS = {"n/a", "na", "---", "-", ""}

MAPEO_RECOMIENDA = {
    "si": "Si",
    "sí": "Si",
    "no": "No",
    "maybe": "Talvez",
    "tal vez": "Talvez",
}

# 'Sí'/'1' abren ticket; 'No'/'0' no. Se resuelve en minúscula y sin tilde
# para blindar el mapeo ante fallos de encoding del archivo origen.
MAPEO_TICKET = {
    "si": True,
    "s": True,
    "1": True,
    "no": False,
    "0": False,
}


# ---------------------------------------------------------------------------
# FASE 1: HEALTH SCORE - reglas de negocio específicas de feedback
# ---------------------------------------------------------------------------

def _reglas_negocio_feedback(df):
    """Valida las reglas de negocio propias del feedback de clientes."""
    n = len(df)
    val = {}
    if "Rating_Producto" in df.columns:
        fuera_rango = int(
            (~df["Rating_Producto"].between(RATING_MIN, RATING_MAX)).sum()
        )
        val["rating_producto_fuera_de_rango"] = {
            "cantidad": fuera_rango,
            "porcentaje": round(fuera_rango / n * 100, 2),
        }
    if "Rating_Logistica" in df.columns:
        val["rating_logistica_fuera_de_rango"] = {
            "cantidad": int(
                (~df["Rating_Logistica"].between(RATING_MIN, RATING_MAX)).sum()
            )
        }
    if "Edad_Cliente" in df.columns:
        val["edades_imposibles"] = {
            "cantidad": int((df["Edad_Cliente"] > EDAD_MAXIMA_PLAUSIBLE).sum())
        }

    mask_invalidos = pd.Series(False, index=df.index)
    if "Rating_Producto" in df.columns:
        mask_invalidos |= ~df["Rating_Producto"].between(RATING_MIN, RATING_MAX)
    if "Rating_Logistica" in df.columns:
        mask_invalidos |= ~df["Rating_Logistica"].between(RATING_MIN, RATING_MAX)
    if "Edad_Cliente" in df.columns:
        mask_invalidos |= df["Edad_Cliente"] > EDAD_MAXIMA_PLAUSIBLE
    return val, mask_invalidos


def _auditar_feedback(df, etiqueta):
    """Envoltorio de common.check_data_quality con la configuración de este dataset."""
    return check_data_quality(
        df,
        etiqueta=etiqueta,
        id_col="Feedback_ID",
        excluir_outliers={"Satisfaccion_NPS"},
        reglas_negocio=_reglas_negocio_feedback,
    )


# ---------------------------------------------------------------------------
# FASE 2: LIMPIEZA
# ---------------------------------------------------------------------------

def eliminar_duplicados(df, log):
    """
    Elimina duplicados exactos y por Feedback_ID.

    Hallazgo: 500 Feedback_ID se repiten (el "duplicado intencional" del
    reto). Un mismo formulario contado dos veces infla artificialmente el
    volumen de opiniones y sesga el NPS agregado.
    """
    n0 = len(df)
    df = df.drop_duplicates()
    exactos = n0 - len(df)

    n1 = len(df)
    df = df.drop_duplicates(subset=["Feedback_ID"], keep="first")
    por_id = n1 - len(df)

    log.registrar(
        "Limpieza", "Feedback_ID", "drop_duplicates (exactos + por ID)",
        "El Feedback_ID es la unidad atómica de una opinión; un ID repetido "
        "duplica la voz de un mismo cliente en el NPS agregado.",
        exactos + por_id,
    )
    return df


def tratar_rating_producto(df, log):
    """
    Neutraliza Rating_Producto fuera de la escala 1-5 (hasta 99 observado).

    Se imputa con la mediana global: es una variable ordinal discreta y la
    mediana no se ve afectada por los valores de captura extrema (45, 99).
    """
    fuera_rango = int((~df["Rating_Producto"].between(RATING_MIN, RATING_MAX)).sum())
    df["Rating_Producto_Imputado"] = ~df["Rating_Producto"].between(
        RATING_MIN, RATING_MAX
    )
    df.loc[df["Rating_Producto_Imputado"], "Rating_Producto"] = np.nan

    mediana = df["Rating_Producto"].median()
    df["Rating_Producto"] = df["Rating_Producto"].fillna(mediana)

    log.registrar(
        "Limpieza", "Rating_Producto", f"Fuera de [1,5] -> NaN -> mediana ({mediana:.0f})",
        f"Se detectaron {fuera_rango} valores como 45 o 99 en una escala "
        "1-5; son errores de captura, no calificaciones reales. La mediana "
        "es robusta ante ese tipo de outlier extremo.",
        fuera_rango,
    )
    return df


def tratar_edad_cliente(df, log):
    """Neutraliza edades imposibles (>100 años, hasta 195 observado)."""
    imposibles = int((df["Edad_Cliente"] > EDAD_MAXIMA_PLAUSIBLE).sum())
    df["Edad_Cliente_Imputada"] = df["Edad_Cliente"] > EDAD_MAXIMA_PLAUSIBLE
    df.loc[df["Edad_Cliente_Imputada"], "Edad_Cliente"] = np.nan

    mediana = df["Edad_Cliente"].median()
    df["Edad_Cliente"] = df["Edad_Cliente"].fillna(mediana)

    log.registrar(
        "Limpieza", "Edad_Cliente", f"> {EDAD_MAXIMA_PLAUSIBLE} -> NaN -> mediana ({mediana:.0f})",
        "195 años no es una edad humana plausible; se trata como dato "
        "faltante y se imputa con la mediana por ser robusta ante los "
        "propios valores extremos que la contaminan.",
        imposibles,
    )
    return df


def normalizar_recomienda_marca(df, log):
    """
    Unifica SI/Sí/No/Maybe y marca los nulos como categoría explícita.

    Con 25 % de nulos, imputar con la moda fabricaría la opinión de un
    cuarto de los clientes; la ausencia de respuesta es en sí un dato.
    """
    nulos = int(df["Recomienda_Marca"].isna().sum())
    clave = df["Recomienda_Marca"].astype(str).str.strip().str.lower()
    df["Recomienda_Marca"] = clave.map(MAPEO_RECOMIENDA).fillna("Sin_Respuesta")

    log.registrar(
        "Limpieza", "Recomienda_Marca", "map() a Si/No/Talvez; NaN -> 'Sin_Respuesta'",
        "Con 25 % de nulos, imputar con la moda inventaría la opinión de "
        "una cuarta parte de los clientes; se preserva como hallazgo.",
        nulos,
    )
    return df


def normalizar_ticket_soporte(df, log):
    """
    Normaliza Ticket_Soporte_Abierto (mezcla 'Sí'/'No'/'1'/'0') a booleano.

    Se resuelve sin depender de tildes: el archivo origen tiene fallos de
    codificación en este campo, así que el mapeo usa el primer carácter en
    minúscula ('s' de Sí/Si, '1') para no perder registros por un acento
    mal codificado.
    """
    crudo = df["Ticket_Soporte_Abierto"].astype(str).str.strip().str.lower()
    primera_letra = crudo.str[0]

    df["Ticket_Soporte_Abierto"] = primera_letra.map(MAPEO_TICKET)
    sin_mapear = int(df["Ticket_Soporte_Abierto"].isna().sum())
    df["Ticket_Soporte_Abierto"] = df["Ticket_Soporte_Abierto"].fillna(False)

    log.registrar(
        "Limpieza", "Ticket_Soporte_Abierto", "Normalización de codificación mixta -> bool",
        "El campo mezcla 'Sí'/'No' con '1'/'0' y sufre fallos de tilde en "
        "origen; se mapea por la primera letra en minúscula para no perder "
        "registros por un acento corrupto.",
        sin_mapear,
    )
    return df


def normalizar_comentario(df, log):
    """Convierte placeholders ('N/A', '---') y nulos reales en una sola categoría."""
    crudo = df["Comentario_Texto"].astype(str).str.strip()
    es_placeholder = df["Comentario_Texto"].isna() | crudo.str.lower().isin(
        COMENTARIO_PLACEHOLDERS
    )
    n_placeholder = int(es_placeholder.sum())

    df["Comentario_Texto"] = crudo
    df.loc[es_placeholder, "Comentario_Texto"] = "Sin_Comentario"

    log.registrar(
        "Limpieza", "Comentario_Texto", "Placeholders y NaN -> 'Sin_Comentario'",
        "Es texto libre de baja relevancia cuantitativa; unificar 'N/A', "
        "'---' y los nulos reales evita fragmentar la categoría de 'sin "
        "opinión escrita'.",
        n_placeholder,
    )
    return df


# ---------------------------------------------------------------------------
# FASE 3: FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def crear_variables_derivadas(df, log):
    """Construye los KPIs de experiencia de cliente que alimentan el dashboard."""
    print("\nFASE 3 - Feature Engineering (Feedback)")

    # 1. Segmentación del NPS: el reto exige "normalizar" la escala para
    # interpretarla. El rango -100/100 observado ES el estándar de NPS
    # (no se recorta), así que la normalización correcta es categorizarlo.
    condiciones = [
        df["Satisfaccion_NPS"] >= 50,
        df["Satisfaccion_NPS"] >= 0,
    ]
    etiquetas = ["Promotor", "Pasivo"]
    df["Segmento_NPS"] = np.select(condiciones, etiquetas, default="Detractor")
    log.registrar(
        "Feature Eng.", "Segmento_NPS", "Promotor >=50 / Pasivo [0,50) / Detractor <0",
        "La escala -100/100 es el estándar de NPS; se conserva el valor y "
        "se deriva un segmento legible para la junta directiva.",
        len(df),
    )

    # 2. Rating combinado producto + logística.
    df["Rating_Promedio"] = df[["Rating_Producto", "Rating_Logistica"]].mean(axis=1)
    log.registrar(
        "Feature Eng.", "Rating_Promedio", "mean(Rating_Producto, Rating_Logistica)",
        "Resume en un solo KPI la experiencia de producto y de entrega.",
        len(df),
    )

    # 3. Bandera consolidada de confiabilidad del registro.
    banderas = [
        c
        for c in ["Rating_Producto_Imputado", "Edad_Cliente_Imputada"]
        if c in df.columns
    ]
    df["Registro_Confiable"] = ~df[banderas].any(axis=1)
    log.registrar(
        "Feature Eng.", "Registro_Confiable", "Consolidación de banderas de imputación",
        "Permite filtrar entre opiniones observadas e imputadas: sin esto "
        "el análisis de feedback no es auditable.",
        int((~df["Registro_Confiable"]).sum()),
    )
    return df


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def ejecutar_pipeline():
    """Orquesta la auditoría, limpieza, ingeniería y exportación del feedback."""
    print("=" * 78)
    print("PIPELINE DE CURACIÓN - FEEDBACK DE CLIENTES v2")
    print("Challenge 02 | Fundamentos en Ciencia de Datos | EAFIT 2026-1")
    print("=" * 78)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    log = LogLimpieza()

    # Fase 0 - Carga y auditoría inicial
    df_original = cargar_datos(DATA_PATH)
    salud_antes = _auditar_feedback(df_original, "Antes de la curación")
    imprimir_health_score(salud_antes)

    # Fase 2 - Limpieza
    print(f"\n{'=' * 78}")
    print("FASE 2 - LIMPIEZA Y ESTANDARIZACIÓN")
    print("=" * 78)
    df = df_original.copy()
    df = eliminar_duplicados(df, log)
    df = tratar_rating_producto(df, log)
    df = tratar_edad_cliente(df, log)
    df = normalizar_recomienda_marca(df, log)
    df = normalizar_ticket_soporte(df, log)
    df = normalizar_comentario(df, log)
    df = crear_variables_derivadas(df, log)

    # Fase 4 - Auditoría final
    salud_despues = _auditar_feedback(df, "Después de la curación")
    imprimir_health_score(salud_despues)
    comparativo = comparar_health_scores(salud_antes, salud_despues)

    # Fase 5 - Exportación
    print(f"\n{'=' * 78}")
    print("EXPORTACIÓN DE ARTEFACTOS")
    print("=" * 78)
    try:
        df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
        print(f"  Dataset limpio : {os.path.normpath(OUTPUT_PATH)}")

        df_log = log.to_frame()
        df_log.to_csv(LOG_PATH, index=False, encoding="utf-8")
        print(f"  Log de limpieza: {os.path.normpath(LOG_PATH)} "
              f"({len(df_log)} transformaciones)")

        comparativo.to_csv(HEALTH_PATH, index=False, encoding="utf-8")
        print(f"  Health Score   : {os.path.normpath(HEALTH_PATH)}")
    except PermissionError:
        raise PermissionError(
            "No se pudo escribir la salida. Cierre los CSV si están abiertos en Excel."
        )

    return df, df_log, comparativo


# ---------------------------------------------------------------------------
# INTERFAZ ESTÁNDAR PARA app.py
# ---------------------------------------------------------------------------

def procesar_feedback():
    """Interfaz estándar para app.py: retorna únicamente el dataset limpio."""
    df_limpio, _df_log, _df_health = ejecutar_pipeline()
    return df_limpio


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        df_limpio, df_log, df_health = ejecutar_pipeline()
    except (FileNotFoundError, ValueError, PermissionError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    print(f"\n{'=' * 78}")
    print("PROCESO COMPLETADO")
    print("=" * 78)
    print(f"  Registros finales     : {len(df_limpio):,}")
    print(f"  Columnas finales      : {len(df_limpio.columns)}")
    print(f"  Transformaciones      : {len(df_log)}")
    print(f"  Registros confiables  : {int(df_limpio['Registro_Confiable'].sum()):,} "
          f"({df_limpio['Registro_Confiable'].mean() * 100:.1f} %)")
    print(f"  NPS promedio          : {df_limpio['Satisfaccion_NPS'].mean():.1f}")
