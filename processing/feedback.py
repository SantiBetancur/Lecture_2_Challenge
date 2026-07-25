import pandas as pd
import numpy as np


def check_data_quality(df):
    """
    Revisa la calidad de los datos de un DataFrame.
    """

    calidad = {}

    # Duplicados
    calidad["duplicados"] = df.duplicated().sum()

    # Porcentaje de nulos
    calidad["nulos"] = (df.isna().sum() / len(df) * 100).round(2).to_dict()

    # Inconsistencias en columnas de texto
    inconsistencias = {}

    columnas_texto = df.select_dtypes(include=["object", "string"]).columns

    for col in columnas_texto:

        # Ignorar nulos
        originales = df[col].dropna()

        unicos_originales = originales.nunique()

        normalizados = originales.astype(str).str.strip().str.lower()

        unicos_normalizados = normalizados.nunique()

        reduccion = unicos_originales - unicos_normalizados

        porcentaje = reduccion / unicos_originales * 100 if unicos_originales > 0 else 0

        inconsistencias[col] = {
            "valores_unicos_originales": unicos_originales,
            "valores_unicos_normalizados": unicos_normalizados,
            "categorias_inconsistentes": reduccion,
            "porcentaje_inconsistencia": round(porcentaje, 2),
        }

    calidad["inconsistencias"] = inconsistencias

    return calidad


def check_for_outliers(df):
    # Outliers usando IQR
    outliers = {}

    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        inferior = Q1 - 1.5 * IQR
        superior = Q3 + 1.5 * IQR

        outliers[col] = ((df[col] < inferior) | (df[col] > superior)).sum()

    print("\nOutliers")
    print(outliers)


#### Limpieza


def clean_feedback_dataset(df):
    ##Comentario_texto
    ##Reemplaza los valores nulos por un dato "Sin Comentario"
    df["Comentario_Texto"] = df["Comentario_Texto"].fillna("Sin comentario")

    ##Recomienda_marca
    ##Reemplaza los valores nulos por la moda

    df["Recomienda_Marca"] = df["Recomienda_Marca"].fillna(
        df["Recomienda_Marca"].mode()[0]
    )

    # Eliminar errores de captura
    df = df[df["Edad_Cliente"] <= 100]
    df = df[df["Rating_Producto"] <= 5]
    df = df[df["Satisfaccion_NPS"] >= 0]
    # Eliminar registros con satisfacción negativa

    # -----------------------------
    # HEALTH SCORE DESPUÉS
    # -----------------------------
    nulos_final = df.isnull().sum().sum()
    duplicados_final = df.duplicated().sum()

    print("\nRegistros finales:", len(df))
    print("Nulos finales:", nulos_final)
    print("Duplicados finales:", duplicados_final)

    return df


if __name__ == "__main__":
    df = pd.read_csv("Data_Science_Challenge_02\\data\\feedback_clientes_v2.csv")

    print(df.head())
    print(df.info())
    print(df.describe())

    calidad = check_data_quality(df)

    for key in calidad:
        print(f"{key} : {calidad[key]}")
        print("\n")

    check_for_outliers(df)

    dataset_limpio = clean_feedback_dataset(df)
    print(dataset_limpio.describe())
    print(dataset_limpio.info())
