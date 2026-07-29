"""Pipelines de curación e integración de datasets."""

from techlogistics.pipelines.feedback import procesar_feedback
from techlogistics.pipelines.integration import construir_fuente_unica
from techlogistics.pipelines.inventory import procesar_inventario
from techlogistics.pipelines.transactions import procesar_transacciones

__all__ = [
    "procesar_inventario",
    "procesar_transacciones",
    "procesar_feedback",
    "construir_fuente_unica",
]
