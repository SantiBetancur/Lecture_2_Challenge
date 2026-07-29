"""
run_pipeline.py
---------------
Ejecuta los pipelines de curación e integración del proyecto.

Uso:
    python scripts/run_pipeline.py              # todos los pipelines
    python scripts/run_pipeline.py --stage inventory
    python scripts/run_pipeline.py --stage transactions feedback integration
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from techlogistics.pipelines.feedback import ejecutar_pipeline as run_feedback
from techlogistics.pipelines.integration import construir_fuente_unica
from techlogistics.pipelines.inventory import ejecutar_pipeline as run_inventory
from techlogistics.pipelines.transactions import ejecutar_pipeline as run_transactions

STAGES = {
    "inventory": ("Inventario", run_inventory),
    "transactions": ("Transacciones", run_transactions),
    "feedback": ("Feedback", run_feedback),
    "integration": ("Integración", lambda: construir_fuente_unica()),
}


def main():
    parser = argparse.ArgumentParser(description="Ejecuta pipelines de TechLogistics DSS")
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=list(STAGES.keys()) + ["all"],
        default=["all"],
        help="Pipeline(s) a ejecutar (default: all)",
    )
    args = parser.parse_args()

    stages = list(STAGES.keys()) if "all" in args.stage else args.stage

    for stage in stages:
        nombre, fn = STAGES[stage]
        print(f"\n{'#' * 78}\n# {nombre.upper()}\n{'#' * 78}")
        fn()

    print(f"\n{'=' * 78}\nPIPELINE COMPLETADO\n{'=' * 78}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"\nERROR — archivo no encontrado: {exc}")
        print("Verifique que data/raw/ contenga los CSV del reto.")
        sys.exit(1)
    except (ValueError, PermissionError, OSError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPipeline cancelado por el usuario.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR inesperado: {exc}")
        sys.exit(1)
