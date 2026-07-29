"""
Rutas y constantes centralizadas del proyecto TechLogistics DSS.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- Datos ---
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"

RAW_INVENTORY = DATA_RAW / "inventario_central_v2.csv"
RAW_TRANSACTIONS = DATA_RAW / "transacciones_logistica_v2.csv"
RAW_FEEDBACK = DATA_RAW / "feedback_clientes_v2.csv"

INTERIM_INVENTORY = DATA_INTERIM / "inventario_limpio.csv"
INTERIM_TRANSACTIONS = DATA_INTERIM / "transacciones_logistica_limpio.csv"
INTERIM_FEEDBACK = DATA_INTERIM / "feedback_limpio.csv"

PROCESSED_MASTER = DATA_PROCESSED / "fuente_unica_verdad.csv"

# --- Reportes ---
REPORTS_QUALITY = ROOT / "reports" / "quality"
REPORTS_INTEGRATION = ROOT / "reports" / "integration"
REPORTS_DELIVERABLES = ROOT / "reports" / "deliverables"

LOG_LIMPIEZA_INVENTORY = REPORTS_QUALITY / "log_limpieza_inventario.csv"
LOG_LIMPIEZA_TRANSACTIONS = REPORTS_QUALITY / "log_limpieza_transacciones.csv"
LOG_LIMPIEZA_FEEDBACK = REPORTS_QUALITY / "log_limpieza_feedback.csv"
LOG_INTEGRATION = REPORTS_INTEGRATION / "log_integracion.csv"

HEALTH_SCORE_INVENTORY = REPORTS_QUALITY / "health_score_inventario.csv"
HEALTH_SCORE_TRANSACTIONS = REPORTS_QUALITY / "health_score_transacciones.csv"
HEALTH_SCORE_FEEDBACK = REPORTS_QUALITY / "health_score_feedback.csv"

REPORT_PDF_FILENAME = (
    "Informe_Consultoria_TechLogistics_Junta_Directiva_Hallazgos_Estrategicos.pdf"
)
REPORT_PDF = ROOT / REPORT_PDF_FILENAME
REPORT_PDF_COPY = REPORTS_DELIVERABLES / REPORT_PDF_FILENAME

# --- Negocio ---
SLA_ENTREGA_DIAS = 15

RAW_CSV_POR_DATASET = {
    "transacciones": RAW_TRANSACTIONS,
    "inventario": RAW_INVENTORY,
    "feedback": RAW_FEEDBACK,
}

LOG_LIMPIEZA_POR_DATASET = {
    "transacciones": LOG_LIMPIEZA_TRANSACTIONS,
    "inventario": LOG_LIMPIEZA_INVENTORY,
    "feedback": LOG_LIMPIEZA_FEEDBACK,
}

HEALTH_SCORE_POR_DATASET = {
    "transacciones": HEALTH_SCORE_TRANSACTIONS,
    "inventario": HEALTH_SCORE_INVENTORY,
    "feedback": HEALTH_SCORE_FEEDBACK,
}


def ensure_dirs():
    """Crea carpetas de salida si no existen."""
    for path in (
        DATA_INTERIM,
        DATA_PROCESSED,
        REPORTS_QUALITY,
        REPORTS_INTEGRATION,
        REPORTS_DELIVERABLES,
    ):
        path.mkdir(parents=True, exist_ok=True)
