"""Health Score compuesto y reportes de calidad de datos."""

import numpy as np
import pandas as pd


def check_data_quality(
    df,
    etiqueta="dataset",
    id_col=None,
    excluir_outliers=None,
    mapeos_categoricos=None,
    reglas_negocio=None,
):
    """
    Calcula el Health Score compuesto de un dataset.

    Ponderación:
        Completitud    40 %   - ausencia de nulos
        Unicidad       25 %   - ausencia de duplicados
        Consistencia   20 %   - ausencia de variantes textuales del mismo valor
        Validez        15 %   - ausencia de outliers (IQR) y violaciones de
                                 reglas de negocio
    """
    excluir_outliers = excluir_outliers or set()
    mapeos_categoricos = mapeos_categoricos or {}

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

    dup_totales = int(df.duplicated().sum())
    if id_col and id_col in df.columns:
        dup_id = int(df.duplicated(subset=[id_col]).sum())
    else:
        dup_id = dup_totales
    pct_duplicados = max(dup_totales, dup_id) / n * 100

    calidad["duplicados"] = {
        "duplicados_exactos": dup_totales,
        "duplicados_por_id": dup_id,
        "porcentaje": round(pct_duplicados, 2),
    }

    nulos_pct = (df.isna().sum() / n * 100).round(2)
    calidad["nulos"] = nulos_pct.to_dict()
    completitud = 100 - nulos_pct.mean()

    inconsistencias = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        serie = df[col].dropna().astype(str)
        if serie.empty:
            continue
        unicos = serie.nunique()
        clave_norm = serie.str.strip().str.lower()
        unicos_norm = clave_norm.nunique()

        if col in mapeos_categoricos:
            canonicos = clave_norm.map(mapeos_categoricos[col]).fillna(clave_norm)
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

    for col in df.select_dtypes(include=[np.number]).columns:
        if col in excluir_outliers:
            continue
        valores = df[col].dropna()
        if valores.empty:
            continue
        q1, q3 = valores.quantile(0.25), valores.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = valores[(valores < low) | (valores > high)]
        calidad["outliers"][col] = {
            "cantidad_outliers": int(len(outliers)),
            "porcentaje": round(len(outliers) / len(valores) * 100, 2),
            "rango_valido": f"[{round(low, 2)}, {round(high, 2)}]",
        }

    if reglas_negocio is not None:
        val, mask_invalidos = reglas_negocio(df)
    else:
        val, mask_invalidos = {}, pd.Series(False, index=df.index)

    pct_reglas_violadas = mask_invalidos.sum() / n * 100
    val["registros_con_regla_violada"] = {
        "cantidad": int(mask_invalidos.sum()),
        "porcentaje": round(pct_reglas_violadas, 2),
    }
    calidad["validaciones"] = val

    score_completitud = completitud
    score_unicidad = max(0, 100 - pct_duplicados * 2)
    score_consistencia = 100 - (
        sum(v["porcentaje_inconsistencia"] for v in inconsistencias.values())
        / max(len(inconsistencias), 1)
    )

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
    if score >= 90:
        return "Excelente"
    if score >= 80:
        return "Muy Bueno"
    if score >= 70:
        return "Bueno"
    if score >= 60:
        return "Aceptable"
    return "Requiere Mejora"


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
