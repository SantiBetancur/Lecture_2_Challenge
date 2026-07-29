"""Utilidades de entrada/salida de datos."""

import os

import pandas as pd


def cargar_datos(ruta):
    """Carga un dataset crudo con manejo explícito de fallos de E/S."""
    ruta = os.fspath(ruta)
    print(f"\nCargando datos desde: {os.path.normpath(ruta)}")
    try:
        df = pd.read_csv(ruta, encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No se encontró el dataset en '{ruta}'. "
            "Verifique que el CSV esté en data/raw/."
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
