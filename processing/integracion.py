"""
integracion.py
---------------
Fase 2.2 del Challenge 02: unión estratégica de los tres datasets curados
(transacciones, inventario, feedback) en una Sola Fuente de Verdad, y
construcción de las variables derivadas que alimentan las 5 preguntas de
alta gerencia.

Dilema del SKU Fantasma
------------------------
480 de 2,889 SKUs únicos en transacciones (16.6 %) no existen en el maestro
de inventario. Sin más contexto del ERP no puede distinguirse si son
productos nuevos aún no catalogados o errores de digitación, así que la
decisión es conservadora en dos frentes:

1. Se conservan TODAS las ventas (`how='left'` en el merge) para no ocultar
   el fenómeno: eliminarlas subestimaría el ingreso real y la Pregunta 3
   exige precisamente cuantificarlas.
2. Se EXCLUYEN del cálculo de margen (`Margen_Utilidad`): no hay costo de
   inventario conocido para ellas, así que inventar uno fabricaría
   rentabilidad donde no hay evidencia. Se incluyen, en cambio, en
   `Ingreso_En_Riesgo` para medir su impacto financiero sin fingir certeza
   sobre su costo.

Challenge 02 - Fundamentos en Ciencia de Datos (Maestría) - EAFIT 2026-1
"""

import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import LogLimpieza
    from lg_transactions import procesar_transacciones
    from inventory import procesar_inventario
    from feedback import procesar_feedback
else:
    from .common import LogLimpieza
    from .lg_transactions import procesar_transacciones
    from .inventory import procesar_inventario
    from .feedback import procesar_feedback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")

OUTPUT_PATH = os.path.join(DATA_DIR, "fuente_unica_verdad.csv")
LOG_PATH = os.path.join(REPORTS_DIR, "log_integracion.csv")

# SLA de entrega usado como "tiempo prometido" (ya calculado en transacciones
# como Brecha_SLA); se documenta aquí para que la Pregunta 2 sea legible.
SLA_ENTREGA_DIAS = 15


def _renombrar_columnas_compartidas(df, sufijo, columnas=("Registro_Confiable",)):
    """Evita colisiones de nombre al mergear los tres datasets."""
    renombres = {c: f"{c}_{sufijo}" for c in columnas if c in df.columns}
    return df.rename(columns=renombres)


def _agregar_feedback_por_transaccion(df_feedback):
    """
    Colapsa el feedback a una fila por Transaccion_ID.

    Un mismo pedido puede recibir más de una opinión (encuestas repetidas);
    para no multiplicar filas en la fuente única se agrega antes del merge:
    promedios para las métricas numéricas, 'any' para tickets de soporte
    (uno solo ya es una señal de riesgo) y la moda para las categóricas.
    """
    agregaciones = {
        "Rating_Producto": "mean",
        "Rating_Logistica": "mean",
        "Rating_Promedio": "mean",
        "Satisfaccion_NPS": "mean",
        "Ticket_Soporte_Abierto": "any",
        "Recomienda_Marca": lambda s: s.mode().iat[0] if not s.mode().empty else "Sin_Respuesta",
    }
    agregaciones = {k: v for k, v in agregaciones.items() if k in df_feedback.columns}

    df_agg = df_feedback.groupby("Transaccion_ID").agg(agregaciones).reset_index()

    condiciones = [df_agg["Satisfaccion_NPS"] >= 50, df_agg["Satisfaccion_NPS"] >= 0]
    df_agg["Segmento_NPS"] = np.select(
        condiciones, ["Promotor", "Pasivo"], default="Detractor"
    )
    return df_agg


def construir_fuente_unica():
    """Ejecuta los tres pipelines de curación y produce la fuente única de verdad."""
    print("=" * 78)
    print("INTEGRACIÓN - CONSTRUCCIÓN DE LA FUENTE ÚNICA DE VERDAD")
    print("Challenge 02 | Fundamentos en Ciencia de Datos | EAFIT 2026-1")
    print("=" * 78)

    log = LogLimpieza()

    df_tx = procesar_transacciones()
    df_inv = procesar_inventario()
    df_fb = procesar_feedback()

    df_tx = _renombrar_columnas_compartidas(df_tx, "Transaccion")
    df_inv = _renombrar_columnas_compartidas(df_inv, "Inventario")
    df_fb = _renombrar_columnas_compartidas(df_fb, "Feedback")

    # --- Merge 1: transacciones + inventario (SKU fantasma) -----------------
    n_tx = len(df_tx)
    df_maestro = df_tx.merge(df_inv, on="SKU_ID", how="left", suffixes=("", "_inv"))
    df_maestro["SKU_Fantasma"] = df_maestro["Categoria"].isna()
    n_fantasma = int(df_maestro["SKU_Fantasma"].sum())

    log.registrar(
        "Integración", "SKU_ID", "merge(transacciones, inventario, how='left')",
        "Se preservan todas las ventas aunque el SKU no exista en el "
        "maestro; eliminarlas subestimaría el ingreso real (Pregunta 3).",
        n_fantasma,
    )
    assert len(df_maestro) == n_tx, "El merge con inventario no debe duplicar filas de venta."

    # --- Merge 2: + feedback agregado por transacción ------------------------
    df_fb_agg = _agregar_feedback_por_transaccion(df_fb)
    df_maestro = df_maestro.merge(df_fb_agg, on="Transaccion_ID", how="left")
    df_maestro["Tiene_Feedback"] = df_maestro["Rating_Promedio"].notna()
    n_con_feedback = int(df_maestro["Tiene_Feedback"].sum())

    log.registrar(
        "Integración", "Transaccion_ID", "merge(+ feedback agregado, how='left')",
        "El feedback es una muestra voluntaria (4,000 opiniones para 10,000 "
        "ventas); no todas las transacciones tienen voz de cliente y eso "
        "debe quedar visible, no imputado.",
        n_con_feedback,
    )

    df_maestro = _crear_variables_derivadas(df_maestro, log)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    df_maestro.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    log.to_frame().to_csv(LOG_PATH, index=False, encoding="utf-8")

    print(f"\n  Fuente única de verdad : {os.path.normpath(OUTPUT_PATH)}")
    print(f"  Filas x columnas       : {df_maestro.shape[0]:,} x {df_maestro.shape[1]}")
    print(f"  SKU fantasma           : {n_fantasma:,} ({n_fantasma / n_tx * 100:.1f} %)")
    print(f"  Con feedback           : {n_con_feedback:,} ({n_con_feedback / n_tx * 100:.1f} %)")

    return df_maestro


def _crear_variables_derivadas(df, log):
    """Variables derivadas de Fase 2.2 (mínimo 3 exigidas por el reto)."""
    print("\nFASE 2.2 - Feature Engineering de integración")

    # 1. Margen de Utilidad (a nivel de LÍNEA, igual que Ingreso_Bruto).
    #
    # Decisión revisada: la primera versión calculaba el margen por unidad
    # (Precio - Costo) sin multiplicar por Cantidad_Vendida, lo que lo hacía
    # inconsistente con Ingreso_Bruto (que sí es Cantidad x Precio) y
    # subestimaba la pérdida real por un factor de ~7x al compararla con el
    # resto del equipo. Se corrige para que ambas métricas midan lo mismo:
    # dólares totales de la línea de venta.
    #
    # Además, en vez de dejar el margen en NaN para el SKU fantasma (17.5 %
    # de las transacciones quedaban fuera de la Pregunta 1 por completo), se
    # imputa un costo estimado usando la tasa de margen MEDIANA observada en
    # las transacciones con costo real, aplicada al precio de venta de cada
    # fila. Es una imputación flexible (varía por transacción, no un valor
    # fijo) y queda marcada con `Costo_Fantasma_Imputado` para que el
    # análisis siga siendo auditable.
    costo_conocido = df["Costo_Unitario_USD"]
    margen_pct_mediano = (
        (df["Precio_Venta_Final"] - costo_conocido) / df["Precio_Venta_Final"] * 100
    ).median()
    costo_estimado = df["Precio_Venta_Final"] * (1 - margen_pct_mediano / 100)

    df["Costo_Fantasma_Imputado"] = costo_conocido.isna()
    df["Costo_Unitario_USD"] = costo_conocido.fillna(costo_estimado)

    df["Margen_Utilidad_Unitario"] = df["Precio_Venta_Final"] - df["Costo_Unitario_USD"]
    df["Margen_Utilidad_Pct"] = np.where(
        df["Precio_Venta_Final"] > 0,
        df["Margen_Utilidad_Unitario"] / df["Precio_Venta_Final"] * 100,
        np.nan,
    )
    df["Margen_Utilidad"] = df["Margen_Utilidad_Unitario"] * df["Cantidad_Vendida"]

    log.registrar(
        "Feature Eng.", "Margen_Utilidad", "(Precio - Costo) x Cantidad_Vendida",
        "Se corrige a nivel de línea para ser consistente con Ingreso_Bruto "
        "(antes solo medía margen por unidad y subestimaba la pérdida total).",
        len(df),
    )
    log.registrar(
        "Feature Eng.", "Costo_Unitario_USD (SKU fantasma)",
        f"NaN -> Precio x (1 - {margen_pct_mediano:.1f}% margen mediano conocido)",
        "Imputación flexible en vez de excluir el 17.5% de SKU fantasma del "
        "análisis de rentabilidad: se estima su costo con la tasa de margen "
        "mediana de las transacciones sí controladas, en lugar de un valor "
        "fijo, y se deja la bandera 'Costo_Fantasma_Imputado' para auditar "
        "cuáles filas usan un costo real vs. estimado.",
        int(df["Costo_Fantasma_Imputado"].sum()),
    )

    # 2. Brecha de Entrega vs Prometido (alias legible del SLA ya calculado).
    if "Brecha_SLA" in df.columns:
        df["Brecha_Entrega_vs_Prometido"] = df["Brecha_SLA"]
    else:
        df["Brecha_Entrega_vs_Prometido"] = df["Tiempo_Entrega_Real"] - SLA_ENTREGA_DIAS
    log.registrar(
        "Feature Eng.", "Brecha_Entrega_vs_Prometido", f"Tiempo_Entrega_Real - SLA({SLA_ENTREGA_DIAS}d)",
        "Nombre de negocio para la brecha ya calculada en transacciones; "
        "insumo directo de la Pregunta 2.",
        len(df),
    )

    # 3. Ingreso en riesgo por venta invisible (Pregunta 3).
    df["Ingreso_En_Riesgo"] = np.where(df["SKU_Fantasma"], df["Ingreso_Bruto"], 0.0)
    log.registrar(
        "Feature Eng.", "Ingreso_En_Riesgo", "Ingreso_Bruto donde SKU_Fantasma",
        "Cuantifica en USD el ingreso que hoy no tiene respaldo de "
        "inventario, sin mezclarlo con las ventas sí controladas.",
        int(df["SKU_Fantasma"].sum()),
    )

    # 4. Ratio de Soporte por Categoría (ejemplo explícito del reto).
    if "Ticket_Soporte_Abierto" in df.columns and "Categoria" in df.columns:
        df["Ratio_Soporte_Categoria"] = df.groupby("Categoria")[
            "Ticket_Soporte_Abierto"
        ].transform(lambda s: s.fillna(False).mean())
        log.registrar(
            "Feature Eng.", "Ratio_Soporte_Categoria", "groupby('Categoria').transform('mean')",
            "Tasa de tickets de soporte por categoría; contrasta stock alto "
            "con fricción de servicio en la Pregunta 4.",
            len(df),
        )

    return df


if __name__ == "__main__":
    try:
        df_fuente = construir_fuente_unica()
    except (FileNotFoundError, ValueError, PermissionError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    print(f"\n{'=' * 78}")
    print("INTEGRACIÓN COMPLETADA")
    print("=" * 78)
    print(f"  Ingreso bruto total    : USD {df_fuente['Ingreso_Bruto'].sum():,.2f}")
    print(f"  Ingreso en riesgo      : USD {df_fuente['Ingreso_En_Riesgo'].sum():,.2f}")
    print(
        f"  % Ingreso en riesgo    : "
        f"{df_fuente['Ingreso_En_Riesgo'].sum() / df_fuente['Ingreso_Bruto'].sum() * 100:.2f} %"
    )
