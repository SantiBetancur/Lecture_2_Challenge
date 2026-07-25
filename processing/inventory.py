import pandas as pd

file_path = "data/inventario_central_v2.csv"


df = pd.read_csv(file_path, sep=",")

def check_data_quality(df):
    """
    Revisa la calidad de los datos de un DataFrame.
    """

    calidad = {}

    # Duplicados
    calidad["duplicados"] = df.duplicated().sum()

    # Porcentaje de nulos
    calidad["nulos"] = (
        df.isna().sum() / len(df) * 100
    ).round(2).to_dict()

    # Inconsistencias en columnas de texto
    inconsistencias = {}

    columnas_texto = df.select_dtypes(include=["object", "string"]).columns

    for col in columnas_texto:

        # Ignorar nulos
        originales = df[col].dropna()

        unicos_originales = originales.nunique()

        normalizados = (
            originales
            .astype(str)
            .str.strip()
            .str.lower()
        )

        unicos_normalizados = normalizados.nunique()

        reduccion = unicos_originales - unicos_normalizados

        porcentaje = (
            reduccion / unicos_originales * 100
            if unicos_originales > 0
            else 0
        )

        inconsistencias[col] = {
            "valores_unicos_originales": unicos_originales,
            "valores_unicos_normalizados": unicos_normalizados,
            "categorias_inconsistentes": reduccion,
            "porcentaje_inconsistencia": round(porcentaje, 2)
        }

    calidad["inconsistencias"] = inconsistencias

    return calidad

print(check_data_quality(df))

print(df.info())

import pandas as pd

def detectar_outliers(df):
    """
    Detecta outliers y valores extremadamente alejados en columnas numéricas.

    Retorna:
    --------
    resumen : dict
        Estadísticas por columna.

    registros : dict
        DataFrames con los registros que contienen outliers y extremos.
    """

    resumen = {}
    registros = {}

    columnas_numericas = df.select_dtypes(include="number").columns

    for col in columnas_numericas:

        datos = df[col].dropna()

        if len(datos) < 4:
            continue

        Q1 = datos.quantile(0.25)
        Q3 = datos.quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            continue

        # Límites clásicos
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR

        # Límites extremos
        lim_inf_ext = Q1 - 3 * IQR
        lim_sup_ext = Q3 + 3 * IQR

        mascara_outliers = (
            (df[col] < lim_inf) |
            (df[col] > lim_sup)
        )

        mascara_extremos = (
            (df[col] < lim_inf_ext) |
            (df[col] > lim_sup_ext)
        )

        cantidad_outliers = mascara_outliers.sum()
        cantidad_extremos = mascara_extremos.sum()

        resumen[col] = {
            "cantidad_outliers": int(cantidad_outliers),
            "porcentaje_outliers": round(
                cantidad_outliers / len(df) * 100, 2
            ),
            "cantidad_extremos": int(cantidad_extremos),
            "porcentaje_extremos": round(
                cantidad_extremos / len(df) * 100, 2
            ),
            "limite_inferior": lim_inf,
            "limite_superior": lim_sup,
            "limite_inferior_extremo": lim_inf_ext,
            "limite_superior_extremo": lim_sup_ext
        }

        registros[col] = {
            "outliers": df.loc[mascara_outliers].copy(),
            "extremos": df.loc[mascara_extremos].copy()
        }

    return resumen, registros

def clean_data(df):
    """
    Limpia el DataFrame y genera un reporte de las transformaciones realizadas.
    """

    reporte = []

    filas_iniciales = len(df)

    # =========================
    # Duplicados
    # =========================
    duplicados = df.duplicated().sum()

    if duplicados > 0:
        df = df.drop_duplicates()

    reporte.append(f"REGISTROS INICIALES: {filas_iniciales}")
    reporte.append(f"Duplicados eliminados: {duplicados}")

    # =========================
    # Normalización de texto
    # =========================
    columnas_texto = ["Categoria", "Bodega_Origen"]

    for col in columnas_texto:
        df[col] = (
            df[col]
            .str.strip()
            .str.lower()
        )

    reporte.append(
        "Columnas normalizadas (minúsculas y eliminación de espacios): "
        + ", ".join(columnas_texto)
    )

    # =========================
    # Stock negativo y nulos
    # =========================
    negativos = (df["Stock_Actual"] < 0).sum()
    nulos_stock = df["Stock_Actual"].isna().sum()

    df.loc[
        (df["Stock_Actual"] < 0) |
        (df["Stock_Actual"].isna()),
        "Stock_Actual"
    ] = 0

    reporte.append(
        f"Stock_Actual: {negativos} valores negativos y "
        f"{nulos_stock} nulos reemplazados por 0."
    )

    # =========================
    # Lead Time
    # =========================
    nulos_lead = df["Lead_Time_Dias"].isna().sum()

    df["Lead_Time_Dias"] = (
        df["Lead_Time_Dias"]
        .fillna("desconocido")
    )

    reporte.append(
        f"Lead_Time_Dias: {nulos_lead} valores nulos reemplazados por 'desconocido'."
    )

    # =========================
    # Valores imposibles
    # =========================
    imposibles = (df["Costo_Unitario_USD"] >= 850000).sum()

    df = df[df["Costo_Unitario_USD"] < 850000]

    reporte.append(
        f"Registros eliminados por costo unitario imposible: {imposibles}"
    )

    # =========================
    # Estado final
    # =========================
    filas_finales = len(df)

    reporte.append(f"REGISTROS FINALES: {filas_finales}")
    reporte.append(f"Registros eliminados en total: {filas_iniciales - filas_finales}")

    reporte.append("\nPORCENTAJE DE NULOS DESPUÉS DE LA LIMPIEZA")

    for col in df.columns:

        porcentaje = round(df[col].isna().mean()*100,2)

        reporte.append(f"{col}: {porcentaje}%")

    return df, "\n".join(reporte)



def get_cleaned_data(df):
    cleaned_df, clean_data_report = clean_data(df)
    return cleaned_df, clean_data_report

