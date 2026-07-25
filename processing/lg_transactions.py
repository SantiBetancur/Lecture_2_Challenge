"""
lg_transactions.py
------------------
Pipeline de auditoría, limpieza y feature engineering del dataset
`transacciones_logistica_v2.csv`.

Challenge 02 - Fundamentos en Ciencia de Datos (Maestría) - EAFIT 2026-1
Autor: Santiago Betancur

Salidas generadas en ../data y ../reports:
    - transacciones_logistica_limpio.csv   Dataset curado
    - log_limpieza_transacciones.csv       Trazabilidad de cada transformación
    - health_score_transacciones.csv       Health Score antes vs. después
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")

DATA_PATH = os.path.join(DATA_DIR, "transacciones_logistica_v2.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "transacciones_logistica_limpio.csv")
LOG_PATH = os.path.join(REPORTS_DIR, "log_limpieza_transacciones.csv")
HEALTH_PATH = os.path.join(REPORTS_DIR, "health_score_transacciones.csv")

FORMATO_FECHA = "%d/%m/%Y"

# Valor centinela usado por el sistema origen para "cantidad no registrada".
CENTINELA_CANTIDAD = -5

# Valor centinela para "tiempo de entrega desconocido".
CENTINELA_TIEMPO = 999

# Diccionario de normalización de ciudades. La clave es la forma
# minúscula y sin espacios; el valor es el nombre canónico.
MAPEO_CIUDADES = {
    "bog": "Bogota",
    "bogota": "Bogota",
    "bogotá": "Bogota",
    "med": "Medellin",
    "medellin": "Medellin",
    "medellín": "Medellin",
    "cali": "Cali",
    "barranquilla": "Barranquilla",
    "bucaramanga": "Bucaramanga",
}

# Valores que NO son ciudades: son canales de venta que contaminaron
# la columna Ciudad_Destino durante la extracción.
CIUDADES_INVALIDAS = {"ventas_web", "ventas web", "web", "online", "app"}

# Etiqueta explícita para nulos de Estado_Envio (no se inventa un estado).
ESTADO_DESCONOCIDO = "Sin_Informacion"


# ---------------------------------------------------------------------------
# SISTEMA DE TRAZABILIDAD
# ---------------------------------------------------------------------------

class LogLimpieza:
    """Acumula cada transformación aplicada para exportarla como evidencia."""

    def __init__(self):
        self.entradas = []
        self._paso = 0

    def registrar(self, etapa, columna, accion, justificacion, afectados):
        self._paso += 1
        self.entradas.append(
            {
                "paso": self._paso,
                "etapa": etapa,
                "columna": columna,
                "accion": accion,
                "justificacion": justificacion,
                "registros_afectados": int(afectados),
            }
        )
        print(f"  [{self._paso:02d}] {columna:<22} {accion:<38} -> {afectados:>6} reg.")

    def to_frame(self):
        return pd.DataFrame(self.entradas)


# ---------------------------------------------------------------------------
# FASE 0: CARGA
# ---------------------------------------------------------------------------

def cargar_datos(ruta):
    """Carga el dataset crudo con manejo explícito de fallos de E/S."""
    print(f"\nCargando datos desde: {os.path.normpath(ruta)}")
    try:
        df = pd.read_csv(ruta, encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No se encontró el dataset en '{ruta}'. "
            "Verifique que el CSV esté en la carpeta ../data/."
        )
    except UnicodeDecodeError:
        print("  Advertencia: fallo UTF-8, reintentando con latin-1.")
        df = pd.read_csv(ruta, encoding="latin-1")
    except pd.errors.EmptyDataError:
        raise ValueError("El archivo existe pero está vacío o corrupto.")
    except pd.errors.ParserError as exc:
        raise ValueError(f"El CSV está malformado y no pudo parsearse: {exc}")

    print(f"  Cargados {len(df):,} registros x {len(df.columns)} columnas")
    return df


# ---------------------------------------------------------------------------
# FASE 1: HEALTH SCORE
# ---------------------------------------------------------------------------

def check_data_quality(df, etiqueta="dataset"):
    """
    Calcula el Health Score compuesto del dataset.

    Ponderación:
        Completitud    40 %   - ausencia de nulos
        Unicidad       25 %   - ausencia de duplicados
        Consistencia   20 %   - ausencia de variantes textuales del mismo valor
        Validez        15 %   - ausencia de outliers (IQR) y valores imposibles
    """
    calidad = {
        "etiqueta": etiqueta,
        "resumen_general": {},
        "duplicados": {},
        "nulos": {},
        "inconsistencias": {},
        "outliers": {},
        "validaciones": {},
    }

    n = len(df)
    if n == 0:
        raise ValueError("No se puede evaluar la calidad de un DataFrame vacío.")

    # --- Unicidad -----------------------------------------------------------
    dup_totales = int(df.duplicated().sum())
    if "Transaccion_ID" in df.columns:
        dup_id = int(df.duplicated(subset=["Transaccion_ID"]).sum())
    else:
        dup_id = dup_totales
    pct_duplicados = dup_totales / n * 100

    calidad["duplicados"] = {
        "duplicados_exactos": dup_totales,
        "duplicados_por_id": dup_id,
        "porcentaje": round(pct_duplicados, 2),
    }

    # --- Completitud --------------------------------------------------------
    nulos_pct = (df.isna().sum() / n * 100).round(2)
    calidad["nulos"] = nulos_pct.to_dict()
    completitud = 100 - nulos_pct.mean()

    # --- Consistencia textual ----------------------------------------------
    inconsistencias = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        serie = df[col].dropna().astype(str)
        if serie.empty:
            continue
        unicos = serie.nunique()
        clave_norm = serie.str.strip().str.lower()
        unicos_norm = clave_norm.nunique()

        # Además de las variantes tipográficas, se detectan las variantes
        # SEMÁNTICAS: 'BOG' y 'Bogotá' son strings distintos pero la misma
        # entidad. Sin esto el score de consistencia da un falso 100 %.
        if col == "Ciudad_Destino":
            canonicos = clave_norm.map(MAPEO_CIUDADES).fillna(clave_norm)
            unicos_norm = canonicos.nunique()

        reduccion = unicos - unicos_norm
        inconsistencias[col] = {
            "valores_unicos": int(unicos),
            "inconsistencias_detectadas": int(reduccion),
            "porcentaje_inconsistencia": round(
                reduccion / unicos * 100 if unicos else 0, 2
            ),
        }
    calidad["inconsistencias"] = inconsistencias

    # --- Validez: outliers por IQR -----------------------------------------
    # Se excluyen las derivadas de calendario: un año no tiene "outliers"
    # en sentido estadístico, solo refleja el rango temporal del dataset.
    excluir_outliers = {"Anio_Venta"}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in excluir_outliers:
            continue
        valores = df[col].dropna()
        if valores.empty:
            continue
        q1, q3 = valores.quantile(0.25), valores.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = valores[(valores < low) | (valores > high)]
        calidad["outliers"][col] = {
            "cantidad_outliers": int(len(outliers)),
            "porcentaje": round(len(outliers) / len(valores) * 100, 2),
            "rango_valido": f"[{round(low, 2)}, {round(high, 2)}]",
        }

    # --- Validaciones de reglas de negocio ---------------------------------
    val = {}
    if "Cantidad_Vendida" in df.columns:
        neg = int((df["Cantidad_Vendida"] < 0).sum())
        val["cantidades_negativas"] = {
            "cantidad": neg,
            "porcentaje": round(neg / n * 100, 2),
        }
    if "Precio_Venta_Final" in df.columns:
        val["precios_no_positivos"] = {
            "cantidad": int((df["Precio_Venta_Final"] <= 0).sum())
        }
    if "Costo_Envio" in df.columns:
        val["costos_negativos"] = {"cantidad": int((df["Costo_Envio"] < 0).sum())}
    if "Tiempo_Entrega_Real" in df.columns:
        val["tiempos_negativos"] = {
            "cantidad": int((df["Tiempo_Entrega_Real"] < 0).sum())
        }
        val["tiempos_centinela_999"] = {
            "cantidad": int((df["Tiempo_Entrega_Real"] >= CENTINELA_TIEMPO).sum())
        }
    if "Ciudad_Destino" in df.columns:
        clave = df["Ciudad_Destino"].astype(str).str.strip().str.lower()
        val["ciudades_invalidas"] = {
            "cantidad": int(clave.isin(CIUDADES_INVALIDAS).sum()),
            "valores_unicos": int(df["Ciudad_Destino"].nunique()),
        }
    # Registros que violan al menos una regla de negocio dura. Se cuentan
    # como conjunto (no como suma) para no penalizar dos veces la misma fila.
    mask_invalidos = pd.Series(False, index=df.index)
    if "Cantidad_Vendida" in df.columns:
        mask_invalidos |= df["Cantidad_Vendida"] < 0
    if "Precio_Venta_Final" in df.columns:
        mask_invalidos |= df["Precio_Venta_Final"] <= 0
    if "Costo_Envio" in df.columns:
        mask_invalidos |= df["Costo_Envio"] < 0
    if "Tiempo_Entrega_Real" in df.columns:
        mask_invalidos |= df["Tiempo_Entrega_Real"] < 0
        mask_invalidos |= df["Tiempo_Entrega_Real"] >= CENTINELA_TIEMPO
    if "Ciudad_Destino" in df.columns:
        mask_invalidos |= (
            df["Ciudad_Destino"].astype(str).str.strip().str.lower()
            .isin(CIUDADES_INVALIDAS)
        )

    pct_reglas_violadas = mask_invalidos.sum() / n * 100
    val["registros_con_regla_violada"] = {
        "cantidad": int(mask_invalidos.sum()),
        "porcentaje": round(pct_reglas_violadas, 2),
    }
    calidad["validaciones"] = val

    # --- Score compuesto ----------------------------------------------------
    score_completitud = completitud
    score_unicidad = max(0, 100 - pct_duplicados * 2)
    score_consistencia = 100 - (
        sum(v["porcentaje_inconsistencia"] for v in inconsistencias.values())
        / max(len(inconsistencias), 1)
    )

    # La validez pesa dos señales: dispersión estadística (outliers IQR) y
    # violación de reglas de negocio. Un centinela de 999 días es válido para
    # el IQR pero imposible para el negocio, así que sin el segundo término
    # el score no reflejaría el problema real del dataset.
    pct_outliers = (
        sum(v["porcentaje"] for v in calidad["outliers"].values())
        / max(len(calidad["outliers"]), 1)
        if calidad["outliers"]
        else 0
    )
    score_validez = max(
        0, 100 - (pct_outliers * 0.4) - (pct_reglas_violadas * 0.6 * 3)
    )

    health = (
        score_completitud * 0.40
        + score_unicidad * 0.25
        + score_consistencia * 0.20
        + score_validez * 0.15
    )

    calidad["resumen_general"] = {
        "etiqueta": etiqueta,
        "health_score_total": round(health, 2),
        "score_completitud": round(score_completitud, 2),
        "score_unicidad": round(score_unicidad, 2),
        "score_consistencia": round(score_consistencia, 2),
        "score_validez": round(score_validez, 2),
        "total_registros": n,
        "total_columnas": len(df.columns),
        "estado": _clasificar_health_score(health),
    }
    return calidad


def _clasificar_health_score(score):
    """Traduce el score numérico a una categoría legible para la junta."""
    if score >= 90:
        return "Excelente"
    if score >= 80:
        return "Muy Bueno"
    if score >= 70:
        return "Bueno"
    if score >= 60:
        return "Aceptable"
    return "Requiere Mejora"


# ---------------------------------------------------------------------------
# FASE 2: LIMPIEZA
# ---------------------------------------------------------------------------

def eliminar_duplicados(df, log):
    """Elimina duplicados exactos y por Transaccion_ID (clave de negocio)."""
    n0 = len(df)
    df = df.drop_duplicates()
    exactos = n0 - len(df)

    n1 = len(df)
    df = df.drop_duplicates(subset=["Transaccion_ID"], keep="first")
    por_id = n1 - len(df)

    log.registrar(
        "Limpieza", "Transaccion_ID", "drop_duplicates (exactos + por ID)",
        "La transacción es la unidad atómica; un ID repetido inflaría el ingreso.",
        exactos + por_id,
    )
    return df


def normalizar_fechas(df, log):
    """Convierte Fecha_Venta a datetime y deriva variables de calendario."""
    if "Fecha_Venta" not in df.columns:
        return df

    validas_antes = df["Fecha_Venta"].notna().sum()
    convertida = pd.to_datetime(
        df["Fecha_Venta"], format=FORMATO_FECHA, errors="coerce"
    )

    # Segundo intento flexible para registros que no siguen el formato base.
    fallidas = convertida.isna() & df["Fecha_Venta"].notna()
    if fallidas.any():
        convertida.loc[fallidas] = pd.to_datetime(
            df.loc[fallidas, "Fecha_Venta"], errors="coerce", dayfirst=True
        )

    df["Fecha_Venta"] = convertida
    irrecuperables = int(validas_antes - df["Fecha_Venta"].notna().sum())

    log.registrar(
        "Limpieza", "Fecha_Venta", f"to_datetime(format='{FORMATO_FECHA}', coerce)",
        "Sin tipo fecha no hay análisis temporal ni cálculo de antigüedad.",
        irrecuperables,
    )

    df["Anio_Venta"] = df["Fecha_Venta"].dt.year
    df["Mes_Venta"] = df["Fecha_Venta"].dt.to_period("M").astype(str)
    df["Trimestre_Venta"] = df["Fecha_Venta"].dt.to_period("Q").astype(str)

    log.registrar(
        "Feature Eng.", "Anio/Mes/Trimestre_Venta", "Derivación de calendario",
        "Habilita el análisis de estacionalidad y los filtros del dashboard.",
        len(df),
    )
    return df


def normalizar_sku(df, log):
    """Estandariza SKU_ID: sin este paso el merge con inventario falla."""
    if "SKU_ID" not in df.columns:
        return df

    antes = df["SKU_ID"].nunique()
    df["SKU_ID"] = df["SKU_ID"].astype(str).str.strip().str.upper()
    despues = df["SKU_ID"].nunique()

    log.registrar(
        "Limpieza", "SKU_ID", "strip + upper",
        "La llave del JOIN debe ser idéntica en ambos lados; "
        "un espacio genera un falso SKU fantasma.",
        antes - despues,
    )
    return df


def normalizar_ciudades(df, log):
    """
    Unifica variantes de ciudad (BOG/Bogotá, MED/Medellín) y marca como nulos
    los valores que no son ciudades sino canales de venta filtrados.
    """
    if "Ciudad_Destino" not in df.columns:
        return df

    antes = df["Ciudad_Destino"].nunique()
    clave = df["Ciudad_Destino"].astype(str).str.strip().str.lower()

    # Aislar contaminación por canal antes de mapear.
    mask_invalida = clave.isin(CIUDADES_INVALIDAS)
    n_invalidas = int(mask_invalida.sum())

    df["Ciudad_Destino"] = clave.map(MAPEO_CIUDADES).fillna(clave.str.title())
    df.loc[mask_invalida, "Ciudad_Destino"] = np.nan

    # Bandera auditable: no se pierde la evidencia del error de origen.
    df["Ciudad_Invalida_Origen"] = mask_invalida

    despues = df["Ciudad_Destino"].nunique()

    log.registrar(
        "Limpieza", "Ciudad_Destino", "map() de variantes a nombre canónico",
        "Sin unificar, Bogotá se parte en dos y la correlación por "
        "ciudad de la Pregunta 2 queda subestimada.",
        antes - despues,
    )
    log.registrar(
        "Limpieza", "Ciudad_Destino", "Canal filtrado en columna -> NaN",
        "'Ventas_Web' es un canal, no un destino: imputarlo como ciudad "
        "fabricaría un mercado inexistente. Se marca y se excluye del geo-análisis.",
        n_invalidas,
    )
    return df


def tratar_cantidad_vendida(df, log):
    """
    Trata el centinela -5 en Cantidad_Vendida.

    Hallazgo: los 100 registros negativos valen exactamente -5 y se distribuyen
    de forma uniforme entre los seis estados de envío (solo 11 son 'Devuelto').
    Una devolución real seguiría la distribución de 'Devuelto', no la uniforme.
    Se concluye que es un código de error del ERP, no un evento de negocio.
    """
    if "Cantidad_Vendida" not in df.columns:
        return df

    mask = df["Cantidad_Vendida"] < 0
    n_neg = int(mask.sum())

    df["Cantidad_Sospechosa_Origen"] = mask
    df.loc[mask, "Cantidad_Vendida"] = np.nan

    mediana = df["Cantidad_Vendida"].median()
    df["Cantidad_Vendida"] = df["Cantidad_Vendida"].fillna(mediana)

    log.registrar(
        "Limpieza", "Cantidad_Vendida", f"Centinela {CENTINELA_CANTIDAD} -> NaN -> mediana",
        f"Los {n_neg} negativos son idénticos (-5) y no correlacionan con "
        "'Devuelto': es un centinela de error, no una devolución. Se imputa con "
        f"mediana ({mediana:.0f}) por robustez ante outliers y se conserva bandera.",
        n_neg,
    )
    return df


def imputar_costos_envio(df, log):
    """Imputa Costo_Envio con la mediana por canal de venta."""
    if "Costo_Envio" not in df.columns:
        return df

    nulos = int(df["Costo_Envio"].isna().sum())
    df["Costo_Envio_Imputado"] = df["Costo_Envio"].isna()

    df["Costo_Envio"] = df.groupby("Canal_Venta")["Costo_Envio"].transform(
        lambda x: x.fillna(x.median())
    )
    # Red de seguridad si un canal completo quedara sin datos.
    df["Costo_Envio"] = df["Costo_Envio"].fillna(df["Costo_Envio"].median())

    log.registrar(
        "Limpieza", "Costo_Envio", "groupby('Canal_Venta') -> mediana",
        "La mediana es robusta ante outliers de flete y la segmentación por "
        "canal preserva la estructura de costos de cada uno.",
        nulos,
    )
    return df


def imputar_estado_envio(df, log):
    """
    Marca los nulos de Estado_Envio como categoría propia.

    No se imputa con la moda: el 16.8 % de nulos es demasiado alto y asignar
    'Entregado' o 'Retrasado' inventaría un desempeño logístico que no se midió.
    La ausencia de estado es, en sí misma, un síntoma de invisibilidad operativa.
    """
    if "Estado_Envio" not in df.columns:
        return df

    nulos = int(df["Estado_Envio"].isna().sum())
    df["Estado_Envio"] = df["Estado_Envio"].fillna(ESTADO_DESCONOCIDO)
    df["Estado_Envio"] = df["Estado_Envio"].astype(str).str.strip().str.title()

    log.registrar(
        "Limpieza", "Estado_Envio", f"NaN -> '{ESTADO_DESCONOCIDO}'",
        "Con 16.8 % de nulos, imputar con la moda sesgaría el KPI logístico. "
        "La falta de registro es en sí un hallazgo para la junta.",
        nulos,
    )
    return df


def tratar_tiempo_entrega(df, log):
    """
    Neutraliza el centinela 999 y los valores físicamente imposibles.

    Se imputa con la mediana por ciudad: los tiempos de entrega dependen de la
    geografía, así que una mediana global distorsionaría las plazas lejanas.
    """
    if "Tiempo_Entrega_Real" not in df.columns:
        return df

    negativos = int((df["Tiempo_Entrega_Real"] < 0).sum())
    centinelas = int((df["Tiempo_Entrega_Real"] >= CENTINELA_TIEMPO).sum())

    df["Tiempo_Entrega_Imputado"] = (df["Tiempo_Entrega_Real"] < 0) | (
        df["Tiempo_Entrega_Real"] >= CENTINELA_TIEMPO
    )
    df.loc[df["Tiempo_Entrega_Imputado"], "Tiempo_Entrega_Real"] = np.nan

    df["Tiempo_Entrega_Real"] = df.groupby("Ciudad_Destino")[
        "Tiempo_Entrega_Real"
    ].transform(lambda x: x.fillna(x.median()))
    df["Tiempo_Entrega_Real"] = df["Tiempo_Entrega_Real"].fillna(
        df["Tiempo_Entrega_Real"].median()
    )

    log.registrar(
        "Limpieza", "Tiempo_Entrega_Real", f"Centinela {CENTINELA_TIEMPO} y negativos -> NaN",
        "999 días no es una entrega lenta sino un 'sin dato'. Dejarlo "
        "multiplicaría por 50 el promedio y arruinaría la Pregunta 2.",
        negativos + centinelas,
    )
    log.registrar(
        "Limpieza", "Tiempo_Entrega_Real", "Imputación mediana por ciudad",
        "El tiempo de entrega es geográfico; la mediana por ciudad respeta "
        "las diferencias reales entre plazas.",
        negativos + centinelas,
    )
    return df


# ---------------------------------------------------------------------------
# FASE 3: FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def crear_variables_derivadas(df, log):
    """Construye los KPIs de negocio que alimentan el dashboard."""
    print("\nFASE 3 - Feature Engineering")

    # 1. Ingreso bruto de la línea de venta.
    df["Ingreso_Bruto"] = df["Cantidad_Vendida"] * df["Precio_Venta_Final"]
    log.registrar(
        "Feature Eng.", "Ingreso_Bruto", "Cantidad x Precio",
        "Base monetaria para cuantificar la venta invisible (Pregunta 3).",
        len(df),
    )

    # 2. Ingreso neto de flete (proxy de margen sin datos de inventario).
    df["Ingreso_Neto_Logistico"] = df["Ingreso_Bruto"] - df["Costo_Envio"]
    log.registrar(
        "Feature Eng.", "Ingreso_Neto_Logistico", "Ingreso_Bruto - Costo_Envio",
        "Aísla el peso del flete antes de cruzar con el costo de inventario.",
        len(df),
    )

    # 3. Peso porcentual del envío sobre la venta.
    df["Ratio_Envio_Venta"] = np.where(
        df["Ingreso_Bruto"] > 0, df["Costo_Envio"] / df["Ingreso_Bruto"] * 100, np.nan
    )
    log.registrar(
        "Feature Eng.", "Ratio_Envio_Venta", "Costo_Envio / Ingreso_Bruto * 100",
        "Detecta ventas pequeñas donde el flete devora el margen.",
        len(df),
    )

    # 4. Semáforo de cumplimiento logístico (SLA de 15 días).
    sla = 15
    df["Entrega_Tardia"] = df["Tiempo_Entrega_Real"] > sla
    df["Brecha_SLA"] = df["Tiempo_Entrega_Real"] - sla
    log.registrar(
        "Feature Eng.", "Entrega_Tardia / Brecha_SLA", f"vs. SLA de {sla} días",
        "Convierte el tiempo en un KPI accionable para la Pregunta 2.",
        int(df["Entrega_Tardia"].sum()),
    )

    # 5. Antigüedad de la transacción respecto al corte del dataset.
    if pd.api.types.is_datetime64_any_dtype(df.get("Fecha_Venta")):
        corte = df["Fecha_Venta"].max()
        df["Antiguedad_Dias"] = (corte - df["Fecha_Venta"]).dt.days
        log.registrar(
            "Feature Eng.", "Antiguedad_Dias", f"Días desde la venta hasta {corte:%d/%m/%Y}",
            "Insumo para el análisis de riesgo operativo (Pregunta 5).",
            len(df),
        )

    # 6. Bandera consolidada de confiabilidad del registro.
    banderas = [
        c
        for c in [
            "Cantidad_Sospechosa_Origen",
            "Costo_Envio_Imputado",
            "Tiempo_Entrega_Imputado",
            "Ciudad_Invalida_Origen",
        ]
        if c in df.columns
    ]
    df["Registro_Confiable"] = ~df[banderas].any(axis=1)
    log.registrar(
        "Feature Eng.", "Registro_Confiable", "Consolidación de banderas de imputación",
        "Permite al dashboard filtrar entre datos observados e imputados: "
        "sin esto el análisis no es auditable.",
        int((~df["Registro_Confiable"]).sum()),
    )
    return df


# ---------------------------------------------------------------------------
# REPORTES
# ---------------------------------------------------------------------------

def imprimir_health_score(calidad):
    """Muestra en consola el detalle de un Health Score."""
    r = calidad["resumen_general"]
    print(f"\n{'=' * 78}")
    print(f"HEALTH SCORE - {r['etiqueta'].upper()}")
    print("=" * 78)
    print(f"  Score total    : {r['health_score_total']}/100  ({r['estado']})")
    print(f"  Completitud    : {r['score_completitud']}")
    print(f"  Unicidad       : {r['score_unicidad']}")
    print(f"  Consistencia   : {r['score_consistencia']}")
    print(f"  Validez        : {r['score_validez']}")
    print(f"  Dimensiones    : {r['total_registros']:,} x {r['total_columnas']}")

    nulos = {k: v for k, v in calidad["nulos"].items() if v > 0}
    if nulos:
        print("\n  Nulos por columna:")
        for col, pct in sorted(nulos.items(), key=lambda x: -x[1]):
            print(f"    - {col:<28} {pct:>6.2f} %")
    else:
        print("\n  Sin valores nulos.")

    incons = {
        k: v
        for k, v in calidad["inconsistencias"].items()
        if v["inconsistencias_detectadas"] > 0
    }
    if incons:
        print("\n  Inconsistencias textuales:")
        for col, d in incons.items():
            print(
                f"    - {col:<28} {d['inconsistencias_detectadas']} variantes "
                f"de {d['valores_unicos']} valores"
            )

    if calidad["outliers"]:
        print("\n  Outliers (IQR):")
        for col, d in calidad["outliers"].items():
            if d["cantidad_outliers"] > 0:
                print(
                    f"    - {col:<28} {d['cantidad_outliers']:>5} "
                    f"({d['porcentaje']:>5.2f} %)  válido: {d['rango_valido']}"
                )

    print("\n  Validaciones de negocio:")
    for nombre, detalle in calidad["validaciones"].items():
        print(f"    - {nombre}: {detalle}")


def comparar_health_scores(antes, despues):
    """Construye la tabla comparativa antes/después exigida por el challenge."""
    a, d = antes["resumen_general"], despues["resumen_general"]
    metricas = [
        "health_score_total",
        "score_completitud",
        "score_unicidad",
        "score_consistencia",
        "score_validez",
        "total_registros",
        "total_columnas",
    ]
    comp = pd.DataFrame(
        {
            "metrica": metricas,
            "antes": [a[m] for m in metricas],
            "despues": [d[m] for m in metricas],
        }
    )
    comp["delta"] = (comp["despues"] - comp["antes"]).round(2)

    print(f"\n{'=' * 78}")
    print("COMPARATIVO HEALTH SCORE: ANTES vs. DESPUÉS")
    print("=" * 78)
    print(comp.to_string(index=False))
    print(f"\n  Estado inicial : {a['estado']}")
    print(f"  Estado final   : {d['estado']}")
    return comp


# ---------------------------------------------------------------------------
# RESULTADO DEL PIPELINE
# ---------------------------------------------------------------------------

@dataclass
class ResultadoCuracion:
    """
    Contenedor de todos los artefactos del pipeline.

    Se usa cuando el consumidor necesita algo más que el DataFrame, por
    ejemplo el dashboard de Streamlit, que debe mostrar el Health Score
    comparativo y ofrecer el log de limpieza como descarga.
    """

    df: pd.DataFrame
    log: pd.DataFrame
    health_antes: dict
    health_despues: dict
    comparativo: pd.DataFrame

    def resumen(self):
        """Métricas de una línea para tarjetas del dashboard."""
        return {
            "registros": len(self.df),
            "columnas": len(self.df.columns),
            "transformaciones": len(self.log),
            "health_antes": self.health_antes["resumen_general"]["health_score_total"],
            "health_despues": self.health_despues["resumen_general"]["health_score_total"],
            "registros_confiables": int(self.df["Registro_Confiable"].sum()),
            "pct_confiables": round(self.df["Registro_Confiable"].mean() * 100, 2),
            "ingreso_bruto_total": round(self.df["Ingreso_Bruto"].sum(), 2),
        }


# ---------------------------------------------------------------------------
# PIPELINE (punto de entrada importable)
# ---------------------------------------------------------------------------

def procesar_transacciones(
    ruta_entrada=None,
    exportar=True,
    verbose=True,
    retornar_artefactos=False,
):
    """
    Ejecuta la curación completa del dataset de transacciones.

    Este es el punto de entrada del módulo. Está diseñado para ser importado
    desde el script de integración (merge) y desde la app de Streamlit sin
    provocar efectos secundarios al importar.

    Parámetros
    ----------
    ruta_entrada : str, opcional
        Ruta al CSV crudo. Si es None usa DATA_PATH (../data/).
    exportar : bool, por defecto True
        Escribe el CSV limpio, el log y el Health Score en disco.
        Ponerlo en False evita E/S innecesaria al llamarlo desde el merge.
    verbose : bool, por defecto True
        Imprime el detalle en consola. En Streamlit conviene False.
    retornar_artefactos : bool, por defecto False
        Si es False retorna solo el DataFrame limpio (uso típico del merge).
        Si es True retorna un objeto ResultadoCuracion con log y Health Score.

    Retorna
    -------
    pandas.DataFrame o ResultadoCuracion

    Ejemplos
    --------
    Uso desde el script de integración:

    >>> from lg_transactions import procesar_transacciones
    >>> df_ventas = procesar_transacciones(exportar=False, verbose=False)
    >>> df_unificado = df_ventas.merge(df_inventario, on="SKU_ID", how="left")

    Uso desde Streamlit:

    >>> res = procesar_transacciones(retornar_artefactos=True, verbose=False)
    >>> st.dataframe(res.comparativo)
    >>> st.download_button("Log de limpieza", res.log.to_csv(index=False))
    """
    ruta = ruta_entrada or DATA_PATH

    # Silencia la salida del pipeline sin tocar cada print individual.
    stdout_original = sys.stdout
    if not verbose:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    try:
        print("=" * 78)
        print("PIPELINE DE CURACIÓN - TRANSACCIONES LOGÍSTICAS v2")
        print("Challenge 02 | Fundamentos en Ciencia de Datos | EAFIT 2026-1")
        print("=" * 78)

        log = LogLimpieza()

        # Fase 0 - Carga y auditoría inicial
        df_original = cargar_datos(ruta)
        salud_antes = check_data_quality(df_original, "Antes de la curación")
        imprimir_health_score(salud_antes)

        # Fase 2 - Limpieza
        print(f"\n{'=' * 78}")
        print("FASE 2 - LIMPIEZA Y ESTANDARIZACIÓN")
        print("=" * 78)
        df = df_original.copy()
        df = eliminar_duplicados(df, log)
        df = normalizar_fechas(df, log)
        df = normalizar_sku(df, log)
        df = normalizar_ciudades(df, log)
        df = tratar_cantidad_vendida(df, log)
        df = imputar_costos_envio(df, log)
        df = imputar_estado_envio(df, log)
        df = tratar_tiempo_entrega(df, log)

        # Fase 3 - Feature engineering
        df = crear_variables_derivadas(df, log)

        # Fase 4 - Auditoría final
        salud_despues = check_data_quality(df, "Después de la curación")
        imprimir_health_score(salud_despues)
        comparativo = comparar_health_scores(salud_antes, salud_despues)

        df_log = log.to_frame()

        # Metadatos nativos de pandas. Advertencia: df.attrs NO sobrevive a un
        # merge, así que el script de integración debe guardar el log aparte.
        df.attrs["origen"] = "transacciones_logistica_v2"
        df.attrs["health_score_antes"] = salud_antes["resumen_general"][
            "health_score_total"
        ]
        df.attrs["health_score_despues"] = salud_despues["resumen_general"][
            "health_score_total"
        ]

        # Fase 5 - Exportación (opcional)
        if exportar:
            print(f"\n{'=' * 78}")
            print("EXPORTACIÓN DE ARTEFACTOS")
            print("=" * 78)
            os.makedirs(REPORTS_DIR, exist_ok=True)
            try:
                df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
                print(f"  Dataset limpio : {os.path.normpath(OUTPUT_PATH)}")

                df_log.to_csv(LOG_PATH, index=False, encoding="utf-8")
                print(f"  Log de limpieza: {os.path.normpath(LOG_PATH)} "
                      f"({len(df_log)} transformaciones)")

                comparativo.to_csv(HEALTH_PATH, index=False, encoding="utf-8")
                print(f"  Health Score   : {os.path.normpath(HEALTH_PATH)}")
            except PermissionError:
                raise PermissionError(
                    "No se pudo escribir la salida. Cierre los CSV si están "
                    "abiertos en Excel."
                )
    finally:
        # Restaura stdout incluso si el pipeline falla a mitad de camino.
        if not verbose:
            sys.stdout.close()
            sys.stdout = stdout_original

    if retornar_artefactos:
        return ResultadoCuracion(
            df=df,
            log=df_log,
            health_antes=salud_antes,
            health_despues=salud_despues,
            comparativo=comparativo,
        )
    return df


# ---------------------------------------------------------------------------
# MAIN (solo para pruebas en ejecución directa)
# ---------------------------------------------------------------------------

def main():
    """Ejecución standalone del módulo para verificación local."""
    try:
        resultado = procesar_transacciones(
            exportar=True, verbose=True, retornar_artefactos=True
        )
    except (FileNotFoundError, ValueError, PermissionError) as exc:
        print(f"\nERROR: {exc}")
        return 1

    r = resultado.resumen()
    print(f"\n{'=' * 78}")
    print("PROCESO COMPLETADO")
    print("=" * 78)
    print(f"  Registros finales     : {r['registros']:,}")
    print(f"  Columnas finales      : {r['columnas']}")
    print(f"  Transformaciones      : {r['transformaciones']}")
    print(f"  Health Score          : {r['health_antes']} -> {r['health_despues']}")
    print(f"  Registros confiables  : {r['registros_confiables']:,} "
          f"({r['pct_confiables']} %)")
    print(f"  Ingreso bruto total   : USD {r['ingreso_bruto_total']:,.2f}")
    print("\n  Listo para el merge con inventario y feedback vía 'SKU_ID'.")
    return 0

