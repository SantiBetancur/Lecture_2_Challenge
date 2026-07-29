"""
inventory.py
------------
Pipeline de auditoría, limpieza y feature engineering del dataset
`inventario_central_v2.csv`.

Challenge 02 - Fundamentos en Ciencia de Datos (Maestría) - EAFIT 2026-1

Salidas generadas en data/interim y reports/quality:
    - inventario_limpio.csv           Dataset curado
    - log_limpieza_inventario.csv     Trazabilidad de cada transformación
    - health_score_inventario.csv     Health Score antes vs. después
"""

import os
import re
import sys

import numpy as np
import pandas as pd

from techlogistics.config import (
    HEALTH_SCORE_INVENTORY,
    INTERIM_INVENTORY,
    LOG_LIMPIEZA_INVENTORY,
    RAW_INVENTORY,
    ensure_dirs,
)
from techlogistics.io import cargar_datos
from techlogistics.quality import (
    LogLimpieza,
    check_data_quality,
    comparar_health_scores,
    imprimir_health_score,
)

# Umbral de costo unitario a partir del cual se considera atípico incluso
# antes de calcular IQR (evidencia del reto: "desde $0.01 hasta $850k").
COSTO_MAXIMO_PLAUSIBLE = 850_000

# Diccionario de normalización de categorías. La clave es la forma
# minúscula y sin espacios; el valor es el nombre canónico.
MAPEO_CATEGORIAS = {
    "smart-phone": "Smartphones",
    "smartphones": "Smartphones",
    "smartphone": "Smartphones",
    "laptop": "Laptops",
    "laptops": "Laptops",
    "accesorios": "Accesorios",
    "monitores": "Monitores",
    "tablets": "Tablets",
}

# Marcador de categoría corrupta/no informada durante la extracción.
CATEGORIA_INVALIDA = {"???"}

MAPEO_BODEGAS = {
    "norte": "Norte",
    "sur": "Sur",
    "occidente": "Occidente",
}


# ---------------------------------------------------------------------------
# FASE 1: HEALTH SCORE - reglas de negocio específicas de inventario
# ---------------------------------------------------------------------------

def _reglas_negocio_inventario(df):
    """Valida las reglas de negocio propias del maestro de inventario."""
    n = len(df)
    val = {}
    if "Stock_Actual" in df.columns:
        neg = int((df["Stock_Actual"] < 0).sum())
        val["stock_negativo"] = {
            "cantidad": neg,
            "porcentaje": round(neg / n * 100, 2),
        }
    if "Costo_Unitario_USD" in df.columns:
        val["costos_no_positivos"] = {
            "cantidad": int((df["Costo_Unitario_USD"] <= 0).sum())
        }
        val["costos_extremos"] = {
            "cantidad": int((df["Costo_Unitario_USD"] >= COSTO_MAXIMO_PLAUSIBLE).sum())
        }
    if "Categoria" in df.columns:
        clave = df["Categoria"].astype(str).str.strip()
        val["categorias_invalidas"] = {
            "cantidad": int(clave.isin(CATEGORIA_INVALIDA).sum())
        }

    mask_invalidos = pd.Series(False, index=df.index)
    if "Stock_Actual" in df.columns:
        mask_invalidos |= df["Stock_Actual"] < 0
    if "Costo_Unitario_USD" in df.columns:
        mask_invalidos |= df["Costo_Unitario_USD"] <= 0
        mask_invalidos |= df["Costo_Unitario_USD"] >= COSTO_MAXIMO_PLAUSIBLE
    if "Categoria" in df.columns:
        mask_invalidos |= df["Categoria"].astype(str).str.strip().isin(CATEGORIA_INVALIDA)
    return val, mask_invalidos


def _auditar_inventario(df, etiqueta):
    """Envoltorio de check_data_quality con la configuración de este dataset."""
    return check_data_quality(
        df,
        etiqueta=etiqueta,
        id_col="SKU_ID",
        excluir_outliers=set(),
        mapeos_categoricos={"Categoria": MAPEO_CATEGORIAS, "Bodega_Origen": MAPEO_BODEGAS},
        reglas_negocio=_reglas_negocio_inventario,
    )


# ---------------------------------------------------------------------------
# FASE 2: LIMPIEZA
# ---------------------------------------------------------------------------

def eliminar_duplicados(df, log):
    """Elimina duplicados exactos y por SKU_ID (clave de negocio del maestro)."""
    n0 = len(df)
    df = df.drop_duplicates()
    exactos = n0 - len(df)

    n1 = len(df)
    df = df.drop_duplicates(subset=["SKU_ID"], keep="first")
    por_id = n1 - len(df)

    log.registrar(
        "Limpieza", "SKU_ID", "drop_duplicates (exactos + por ID)",
        "El SKU es la clave primaria del maestro; un duplicado infla el stock "
        "y distorsiona el valor de inventario.",
        exactos + por_id,
    )
    return df


def normalizar_sku(df, log):
    """Estandariza SKU_ID: debe ser idéntico al de transacciones para el JOIN."""
    antes = df["SKU_ID"].nunique()
    df["SKU_ID"] = df["SKU_ID"].astype(str).str.strip().str.upper()
    despues = df["SKU_ID"].nunique()

    log.registrar(
        "Limpieza", "SKU_ID", "strip + upper",
        "Debe coincidir exactamente con el SKU_ID normalizado de transacciones; "
        "de lo contrario el merge genera falsos SKU fantasma.",
        antes - despues,
    )
    return df


def normalizar_categoria(df, log):
    """Unifica variantes de Categoria y marca '???' como nulo explícito."""
    antes = df["Categoria"].nunique()
    clave = df["Categoria"].astype(str).str.strip().str.lower()

    mask_invalida = clave.isin(CATEGORIA_INVALIDA)
    n_invalidas = int(mask_invalida.sum())

    df["Categoria"] = clave.map(MAPEO_CATEGORIAS).fillna(clave.str.title())
    df.loc[mask_invalida, "Categoria"] = np.nan

    moda = df["Categoria"].mode(dropna=True)[0]
    df["Categoria_Imputada"] = df["Categoria"].isna()
    df["Categoria"] = df["Categoria"].fillna(moda)

    despues = df["Categoria"].nunique()

    log.registrar(
        "Limpieza", "Categoria", "map() de variantes a nombre canónico",
        "'smart-phone' y 'Smartphones' son la misma categoría con distinta "
        "captura; sin unificar, el análisis por categoría (Pregunta 4) se "
        "fragmenta artificialmente.",
        antes - despues,
    )
    log.registrar(
        "Limpieza", "Categoria", f"'???' -> NaN -> moda ('{moda}')",
        "La moda es el criterio recomendado para variables categóricas: "
        "preserva la distribución de clases sin inventar una etiqueta "
        "arbitraria para un valor corrupto.",
        n_invalidas,
    )
    return df


def normalizar_bodega(df, log):
    """
    Unifica variantes de capitalización de Bodega_Origen (Norte/norte).

    'ZONA_FRANCA' y 'BOD-EXT-99' se conservan tal cual: no son errores
    tipográficos de Norte/Sur/Occidente sino códigos de bodegas externas
    reales, y colapsarlas ocultaría operación tercerizada relevante para
    la Pregunta 5 (bodegas que "operan a ciegas").
    """
    antes = df["Bodega_Origen"].nunique()
    clave = df["Bodega_Origen"].astype(str).str.strip()
    clave_lower = clave.str.lower()

    df["Bodega_Origen"] = clave_lower.map(MAPEO_BODEGAS).fillna(clave)
    despues = df["Bodega_Origen"].nunique()

    log.registrar(
        "Limpieza", "Bodega_Origen", "map() de capitalización a nombre canónico",
        "'norte' y 'Norte' son la misma bodega; los códigos externos "
        "(ZONA_FRANCA, BOD-EXT-99) se preservan como categorías propias.",
        antes - despues,
    )
    return df


def tratar_stock(df, log):
    """
    Neutraliza Stock_Actual negativo (existencia imposible) y nulo.

    Se imputa con la mediana por Categoría: el stock varía de forma muy
    distinta entre smartphones y monitores, así que una mediana global
    distorsionaría categorías de bajo volumen.
    """
    negativos = int((df["Stock_Actual"] < 0).sum())
    nulos = int(df["Stock_Actual"].isna().sum())

    df["Stock_Imputado"] = (df["Stock_Actual"] < 0) | df["Stock_Actual"].isna()
    df.loc[df["Stock_Actual"] < 0, "Stock_Actual"] = np.nan

    df["Stock_Actual"] = df.groupby("Categoria")["Stock_Actual"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Stock_Actual"] = df["Stock_Actual"].fillna(df["Stock_Actual"].median())

    log.registrar(
        "Limpieza", "Stock_Actual", "Negativos y nulos -> mediana por Categoría",
        "Una existencia negativa desafía la lógica contable; se trata como "
        "dato faltante y se imputa con la mediana de su categoría por ser "
        "robusta ante los propios outliers de stock.",
        negativos + nulos,
    )
    return df


def tratar_costo_unitario(df, log):
    """
    Winsoriza Costo_Unitario_USD por Categoría (clip, no eliminación).

    El producto sigue siendo una venta válida; lo que falló es la captura
    del precio. Eliminar la fila perdería la transacción completa cuando
    basta con acotar el valor extremo.
    """
    tocados = pd.Series(False, index=df.index)

    for categoria, grupo in df.groupby("Categoria"):
        valores = grupo["Costo_Unitario_USD"]
        q1, q3 = valores.quantile(0.25), valores.quantile(0.75)
        iqr = q3 - q1
        low = max(0.01, q1 - 1.5 * iqr)
        high = q3 + 1.5 * iqr

        mask = (valores < low) | (valores > high)
        tocados.loc[grupo.index] = mask
        df.loc[grupo.index, "Costo_Unitario_USD"] = valores.clip(lower=low, upper=high)

    df["Costo_Unitario_Winsorizado"] = tocados

    log.registrar(
        "Limpieza", "Costo_Unitario_USD", "clip() por IQR dentro de cada Categoría",
        "El reto documenta costos desde $0.01 hasta $850k. La winsorización "
        "(Guía Senior Toolkit) acota el valor extremo sin sacrificar la "
        "muestra completa, y se hace por categoría porque un monitor y un "
        "accesorio tienen escalas de precio muy distintas.",
        int(tocados.sum()),
    )
    return df


def tratar_lead_time(df, log):
    """
    Convierte Lead_Time_Dias (texto mixto) a un valor numérico interpretable.

    'Inmediato' -> 0 días. Rangos ('25-30 días') -> promedio del rango.
    Nulos -> mediana por Categoría, ya que el lead time depende del tipo
    de producto (un smartphone importado tarda distinto que un accesorio).
    """
    original = df["Lead_Time_Dias"].copy()

    def _parsear(valor):
        if pd.isna(valor):
            return np.nan
        texto = str(valor).strip().lower()
        if "inmediato" in texto:
            return 0.0
        rango = re.findall(r"\d+", texto)
        if len(rango) >= 2:
            return (int(rango[0]) + int(rango[1])) / 2
        if len(rango) == 1:
            return float(rango[0])
        return np.nan

    df["Lead_Time_Dias_Original"] = original
    df["Lead_Time_Dias"] = original.apply(_parsear)

    irrecuperables = int(df["Lead_Time_Dias"].isna().sum())
    df["Lead_Time_Imputado"] = df["Lead_Time_Dias"].isna()
    df["Lead_Time_Dias"] = df.groupby("Categoria")["Lead_Time_Dias"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Lead_Time_Dias"] = df["Lead_Time_Dias"].fillna(df["Lead_Time_Dias"].median())

    log.registrar(
        "Limpieza", "Lead_Time_Dias", "Parseo de texto mixto -> numérico",
        "'Inmediato' se traduce a 0 días y los rangos ('25-30 días') al "
        "promedio del rango; se conserva el texto original para auditoría.",
        len(df) - irrecuperables,
    )
    log.registrar(
        "Limpieza", "Lead_Time_Dias", "Nulos -> mediana por Categoría",
        "El 16 % de nulos no puede eliminarse sin perder una quinta parte "
        "del maestro; la mediana por categoría respeta que el lead time es "
        "propio del tipo de producto.",
        irrecuperables,
    )
    return df


def normalizar_fecha_revision(df, log):
    """Convierte Ultima_Revision a datetime y deriva la antigüedad en días."""
    if "Ultima_Revision" not in df.columns:
        return df

    validas_antes = df["Ultima_Revision"].notna().sum()
    df["Ultima_Revision"] = pd.to_datetime(df["Ultima_Revision"], errors="coerce")
    irrecuperables = int(validas_antes - df["Ultima_Revision"].notna().sum())

    corte = df["Ultima_Revision"].max()
    df["Antiguedad_Revision_Dias"] = (corte - df["Ultima_Revision"]).dt.days

    log.registrar(
        "Feature Eng.", "Ultima_Revision -> Antiguedad_Revision_Dias",
        f"Días desde la revisión hasta {corte:%d/%m/%Y}",
        "Insumo directo de la Pregunta 5: bodegas con revisiones antiguas "
        "'operan a ciegas' sobre su propio stock.",
        irrecuperables,
    )
    return df


# ---------------------------------------------------------------------------
# FASE 3: FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def crear_variables_derivadas(df, log):
    """Construye los KPIs de inventario que alimentan el dashboard."""
    print("\nFASE 3 - Feature Engineering (Inventario)")

    # 1. Valor total inmovilizado en inventario por SKU.
    df["Valor_Inventario"] = df["Stock_Actual"] * df["Costo_Unitario_USD"]
    log.registrar(
        "Feature Eng.", "Valor_Inventario", "Stock_Actual x Costo_Unitario_USD",
        "Cuantifica el capital atado a cada SKU; base para priorizar "
        "revisiones y detectar sobre-stock costoso.",
        len(df),
    )

    # 2. Disponibilidad frente al punto de reorden (proxy de "stock alto").
    df["Ratio_Stock_Reorden"] = np.where(
        df["Punto_Reorden"] > 0, df["Stock_Actual"] / df["Punto_Reorden"], np.nan
    )
    df["Alta_Disponibilidad"] = df["Ratio_Stock_Reorden"] >= 2
    log.registrar(
        "Feature Eng.", "Ratio_Stock_Reorden / Alta_Disponibilidad",
        "Stock_Actual / Punto_Reorden (umbral 2x)",
        "Insumo directo de la Pregunta 4: categorías con alta disponibilidad "
        "pero feedback negativo señalan sobre-stock, no escasez.",
        int(df["Alta_Disponibilidad"].sum()),
    )

    # 3. Bandera consolidada de confiabilidad del registro.
    banderas = [
        c
        for c in ["Stock_Imputado", "Costo_Unitario_Winsorizado", "Lead_Time_Imputado", "Categoria_Imputada"]
        if c in df.columns
    ]
    df["Registro_Confiable"] = ~df[banderas].any(axis=1)
    log.registrar(
        "Feature Eng.", "Registro_Confiable", "Consolidación de banderas de imputación",
        "Permite filtrar entre datos observados e imputados: sin esto el "
        "análisis de inventario no es auditable.",
        int((~df["Registro_Confiable"]).sum()),
    )
    return df


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def ejecutar_pipeline():
    """Orquesta la auditoría, limpieza, ingeniería y exportación del inventario."""
    print("=" * 78)
    print("PIPELINE DE CURACIÓN - INVENTARIO CENTRAL v2")
    print("Challenge 02 | Fundamentos en Ciencia de Datos | EAFIT 2026-1")
    print("=" * 78)

    ensure_dirs()
    log = LogLimpieza()

    # Fase 0 - Carga y auditoría inicial
    df_original = cargar_datos(RAW_INVENTORY)
    salud_antes = _auditar_inventario(df_original, "Antes de la curación")
    imprimir_health_score(salud_antes)

    # Fase 2 - Limpieza
    print(f"\n{'=' * 78}")
    print("FASE 2 - LIMPIEZA Y ESTANDARIZACIÓN")
    print("=" * 78)
    df = df_original.copy()
    df = eliminar_duplicados(df, log)
    df = normalizar_sku(df, log)
    df = normalizar_categoria(df, log)
    df = normalizar_bodega(df, log)
    df = tratar_stock(df, log)
    df = tratar_costo_unitario(df, log)
    df = tratar_lead_time(df, log)
    df = normalizar_fecha_revision(df, log)
    df = crear_variables_derivadas(df, log)

    # Fase 4 - Auditoría final
    salud_despues = _auditar_inventario(df, "Después de la curación")
    imprimir_health_score(salud_despues)
    comparativo = comparar_health_scores(salud_antes, salud_despues)

    # Fase 5 - Exportación
    print(f"\n{'=' * 78}")
    print("EXPORTACIÓN DE ARTEFACTOS")
    print("=" * 78)
    try:
        df.to_csv(INTERIM_INVENTORY, index=False, encoding="utf-8")
        print(f"  Dataset limpio : {os.path.normpath(INTERIM_INVENTORY)}")

        df_log = log.to_frame()
        df_log.to_csv(LOG_LIMPIEZA_INVENTORY, index=False, encoding="utf-8")
        print(f"  Log de limpieza: {os.path.normpath(LOG_LIMPIEZA_INVENTORY)} "
              f"({len(df_log)} transformaciones)")

        comparativo.to_csv(HEALTH_SCORE_INVENTORY, index=False, encoding="utf-8")
        print(f"  Health Score   : {os.path.normpath(HEALTH_SCORE_INVENTORY)}")
    except PermissionError:
        raise PermissionError(
            "No se pudo escribir la salida. Cierre los CSV si están abiertos en Excel."
        )

    return df, df_log, comparativo


# ---------------------------------------------------------------------------
# INTERFAZ ESTÁNDAR PARA app.py
# ---------------------------------------------------------------------------

def procesar_inventario():
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
    print(f"  Valor total inventario: USD {df_limpio['Valor_Inventario'].sum():,.2f}")
