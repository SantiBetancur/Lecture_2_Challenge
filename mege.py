from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def load_datasets():
    feedback_df = pd.read_csv(DATA_DIR / "feedback_clientes_v2.csv")
    inventory_df = pd.read_csv(DATA_DIR / "inventario_central_v2.csv")
    transactions_df = pd.read_csv(DATA_DIR / "transacciones_logistica_v2.csv")
    return feedback_df, inventory_df, transactions_df


def add_business_columns(merged_df):
    merged_df = merged_df.copy()

    precio_unitario = pd.to_numeric(merged_df["Precio_Venta_Final"], errors="coerce").fillna(0)
    cantidad = pd.to_numeric(merged_df["Cantidad_Vendida"], errors="coerce").fillna(0)
    costo_unitario = pd.to_numeric(merged_df["Costo_Unitario_USD"], errors="coerce").fillna(0)
    costo_envio = pd.to_numeric(merged_df["Costo_Envio"], errors="coerce").fillna(0)

    merged_df["Ingreso_Total_USD"] = (precio_unitario * cantidad).round(2)
    merged_df["Costo_Unitario_Total_USD"] = (costo_unitario * cantidad).round(2)
    merged_df["Margen_Contribucion_USD"] = (
        merged_df["Ingreso_Total_USD"]
        - merged_df["Costo_Unitario_Total_USD"]
        - costo_envio
    ).round(2)

    recomendacion_map = {
        "si": 1.0,
        "sí": 1.0,
        "yes": 1.0,
        "y": 1.0,
        "true": 1.0,
        "no": 0.0,
        "n": 0.0,
        "false": 0.0,
        "maybe": 0.5,
        "quizas": 0.5,
        "tal vez": 0.5,
    }
    recomendacion_norm = (
        merged_df["Recomienda_Marca"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(recomendacion_map)
        .fillna(0.5)
    )
    nps_norm = ((pd.to_numeric(merged_df["Satisfaccion_NPS"], errors="coerce").fillna(0) + 100) / 200 * 5).clip(0, 5)
    rating_producto = pd.to_numeric(merged_df["Rating_Producto"], errors="coerce").fillna(0)
    rating_logistica = pd.to_numeric(merged_df["Rating_Logistica"], errors="coerce").fillna(0)

    merged_df["Indice_Lealtad_Cliente"] = (
        0.35 * rating_producto + 0.25 * rating_logistica + 0.20 * nps_norm + 0.20 * recomendacion_norm
    ).round(2)

    stock_bajo = pd.to_numeric(merged_df["Stock_Actual"], errors="coerce").fillna(0) <= pd.to_numeric(merged_df["Punto_Reorden"], errors="coerce").fillna(0)
    retraso_entrega = (
        pd.to_numeric(merged_df["Tiempo_Entrega_Real"], errors="coerce").fillna(999)
        > pd.to_numeric(merged_df["Lead_Time_Dias"], errors="coerce").fillna(999)
    )
    estado_problema = merged_df["Estado_Envio"].astype(str).str.strip().str.lower() != "entregado"
    merged_df["Riesgo_Operativo"] = ((stock_bajo | retraso_entrega | estado_problema).astype(int))

    return merged_df


def merge_datasets(feedback_df, inventory_df, transactions_df):
    merged_df = (
        feedback_df.merge(transactions_df, on="Transaccion_ID", how="left")
        .merge(inventory_df, on="SKU_ID", how="left")
    )
    return add_business_columns(merged_df)


def main():
    feedback_df, inventory_df, transactions_df = load_datasets()
    merged_df = merge_datasets(feedback_df, inventory_df, transactions_df)

    print("DataFrames cargados correctamente")
    print(f"Feedback: {feedback_df.shape}")
    print(f"Inventario: {inventory_df.shape}")
    print(f"Transacciones: {transactions_df.shape}")
    print(f"Merged: {merged_df.shape}")
    print("\nVista previa del merge:")
    print(merged_df.head())
    print("\nColumnas nuevas creadas:")
    print(merged_df[["Margen_Contribucion_USD", "Indice_Lealtad_Cliente", "Riesgo_Operativo"]].head())

    return merged_df


if __name__ == "__main__":
    main()


