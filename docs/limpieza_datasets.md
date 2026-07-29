# Resumen de limpieza de datasets

**Challenge 02 — Fundamentos en Ciencia de Datos (EAFIT 2026-1)**

Este documento resume las transformaciones aplicadas a los tres datasets del reto. Cada uno fue procesado con un pipeline propio (`src/techlogistics/pipelines/inventory.py`, `transactions.py`, `feedback.py`) que comparte la infraestructura de auditoría y trazabilidad de `src/techlogistics/quality/`.

---

## Metodología común

Todos los pipelines siguen la misma estructura:

1. **Auditoría inicial** — Health Score con cuatro dimensiones: Completitud, Unicidad, Consistencia y Validez (reglas de negocio específicas por dataset).
2. **Limpieza y estandarización** — Cada transformación queda registrada en un log con justificación y número de registros afectados.
3. **Feature engineering** — Variables derivadas para el dashboard y las preguntas del reto.
4. **Auditoría final** — Comparación antes/después del Health Score.
5. **Exportación** — Dataset limpio en `data/` y reportes en `reports/`.

---

## 1. Inventario Central

| | |
|---|---|
| **Archivo origen** | `data/raw/inventario_central_v2.csv` |
| **Archivo limpio** | `data/interim/inventario_limpio.csv` |
| **Registros** | 2.500 (sin eliminación de filas) |
| **Health Score** | 93.48 → **99.46** (+5.98) |

### Problemas detectados

- Categorías con variantes tipográficas (`smart-phone`, `Smartphones`, etc.) y valores corruptos (`???`).
- Bodegas con distinta capitalización (`norte` vs `Norte`).
- Stock negativo o nulo (existencia imposible).
- Costos unitarios extremos (desde $0.01 hasta $850.000).
- `Lead_Time_Dias` en texto mixto (`Inmediato`, `25-30 días`) con ~16 % de nulos.
- Fechas de última revisión sin tipo datetime.

### Transformaciones aplicadas

| Paso | Columna | Acción | Registros afectados |
|------|---------|--------|---------------------|
| 1 | `SKU_ID` | Eliminación de duplicados exactos y por ID | 0 |
| 2 | `SKU_ID` | Normalización `strip + upper` (clave del JOIN con transacciones) | 0 |
| 3 | `Categoria` | Mapeo a nombres canónicos (Smartphones, Laptops, etc.) | 3 |
| 4 | `Categoria` | `???` → NaN → imputación con moda (`Laptops`) | 305 |
| 5 | `Bodega_Origen` | Unificación de capitalización; se preservan códigos externos (`ZONA_FRANCA`, `BOD-EXT-99`) | 1 |
| 6 | `Stock_Actual` | Negativos y nulos → mediana por `Categoria` | 160 |
| 7 | `Costo_Unitario_USD` | Winsorización por IQR dentro de cada categoría (clip, no eliminación) | 1 |
| 8 | `Lead_Time_Dias` | Parseo de texto a numérico (`Inmediato` → 0, rangos → promedio) | 2.097 |
| 9 | `Lead_Time_Dias` | Nulos → mediana por `Categoria` | 403 |
| 10 | `Ultima_Revision` | Conversión a datetime y derivación de `Antiguedad_Revision_Dias` | 0 |

### Variables derivadas

- `Valor_Inventario` = Stock × Costo unitario
- `Ratio_Stock_Reorden` y bandera `Alta_Disponibilidad` (stock ≥ 2× punto de reorden)
- `Registro_Confiable` — consolida banderas de imputación (`Stock_Imputado`, `Costo_Unitario_Winsorizado`, `Lead_Time_Imputado`, `Categoria_Imputada`)

**Registros confiables finales:** ~69.7 % (757 registros con al menos una imputación).

---

## 2. Transacciones Logísticas

| | |
|---|---|
| **Archivo origen** | `data/raw/transacciones_logistica_v2.csv` |
| **Archivo limpio** | `data/interim/transacciones_logistica_limpio.csv` |
| **Registros** | 10.000 (sin eliminación de filas) |
| **Health Score** | 94.43 → **99.69** (+5.26) |

### Problemas detectados

- Fechas en formato texto (`dd/mm/yyyy`).
- Ciudades con variantes (`BOG`, `Bogotá`, `Medellín`) y canales de venta contaminando la columna (`Ventas_Web`, `online`).
- Centinela `-5` en `Cantidad_Vendida` (100 registros negativos idénticos, no devoluciones reales).
- ~8.3 % de nulos en `Costo_Envio`.
- ~16.8 % de nulos en `Estado_Envio`.
- Centinela `999` en `Tiempo_Entrega_Real` (“sin dato”, no entrega lenta).

### Transformaciones aplicadas

| Paso | Columna | Acción | Registros afectados |
|------|---------|--------|---------------------|
| 1 | `Transaccion_ID` | Eliminación de duplicados exactos y por ID | 0 |
| 2 | `Fecha_Venta` | Conversión a datetime (`%d/%m/%Y`) | 0 |
| 3 | — | Derivación de `Anio_Venta`, `Mes_Venta`, `Trimestre_Venta` | 10.000 |
| 4 | `SKU_ID` | Normalización `strip + upper` | 0 |
| 5 | `Ciudad_Destino` | Mapeo a nombres canónicos | 3 |
| 6 | `Ciudad_Destino` | Canales filtrados (`Ventas_Web`, etc.) → NaN + bandera `Ciudad_Invalida_Origen` | 1.290 |
| 7 | `Cantidad_Vendida` | Centinela `-5` → NaN → mediana global (7) + bandera `Cantidad_Sospechosa_Origen` | 100 |
| 8 | `Costo_Envio` | Imputación con mediana por `Canal_Venta` | 834 |
| 9 | `Estado_Envio` | NaN → categoría explícita `Sin_Informacion` (no se usa la moda) | 1.683 |
| 10–11 | `Tiempo_Entrega_Real` | Centinela 999 y negativos → NaN → mediana por `Ciudad_Destino` | 50 |

### Variables derivadas

- `Ingreso_Bruto`, `Ingreso_Neto_Logistico`, `Ratio_Envio_Venta`
- `Entrega_Tardia` y `Brecha_SLA` (SLA de 15 días)
- `Antiguedad_Dias` desde la venta hasta el corte del dataset
- `Registro_Confiable` — consolida banderas de imputación y ciudad inválida

**Registros confiables finales:** ~79.4 % (2.065 registros con al menos una bandera).

---

## 3. Feedback de Clientes

| | |
|---|---|
| **Archivo origen** | `data/raw/feedback_clientes_v2.csv` |
| **Archivo limpio** | `data/interim/feedback_limpio.csv` |
| **Registros** | 4.500 → **4.000** (−500 duplicados intencionales) |
| **Health Score** | 92.40 → **100.00** (+7.60) |

### Problemas detectados

- 500 `Feedback_ID` duplicados (duplicado intencional del reto).
- `Rating_Producto` fuera de escala 1–5 (valores como 45 o 99).
- `Edad_Cliente` imposible (hasta 195 años).
- ~25 % de nulos en `Recomienda_Marca`.
- `Ticket_Soporte_Abierto` con codificación mixta (`Sí`/`No`/`1`/`0` y fallos de tilde).
- Placeholders en comentarios (`N/A`, `---`, vacíos).

### Transformaciones aplicadas

| Paso | Columna | Acción | Registros afectados |
|------|---------|--------|---------------------|
| 1 | `Feedback_ID` | Eliminación de duplicados exactos y por ID | **500** |
| 2 | `Rating_Producto` | Fuera de [1, 5] → NaN → mediana (3) | 27 |
| 3 | `Edad_Cliente` | > 100 años → NaN → mediana (50) | 20 |
| 4 | `Recomienda_Marca` | Mapeo a Si/No/Talvez; NaN → `Sin_Respuesta` | 999 |
| 5 | `Ticket_Soporte_Abierto` | Normalización a booleano (primera letra: `s`/`1` → True, `n`/`0` → False) | 955 |
| 6 | `Comentario_Texto` | Placeholders y NaN → `Sin_Comentario` | 1.152 |

### Variables derivadas

- `Segmento_NPS` — Promotor (≥50), Pasivo [0, 50), Detractor (<0)
- `Rating_Promedio` — media de rating producto y logística
- `Registro_Confiable` — consolida imputaciones de rating y edad

**Registros confiables finales:** ~99.0 % (40 registros con imputación).

---

## Comparativa de Health Score

| Dataset | Antes | Después | Δ | Dimensión más mejorada |
|---------|-------|---------|---|------------------------|
| Inventario | 93.48 | 99.46 | +5.98 | Validez (+24.41) |
| Transacciones | 94.43 | 99.69 | +5.26 | Validez (+24.18) |
| Feedback | 92.40 | 100.00 | +7.60 | Unicidad (+22.22) |

---

## Artefactos generados

| Dataset | Dataset limpio | Log de limpieza | Health Score |
|---------|----------------|-----------------|--------------|
| Inventario | `data/interim/inventario_limpio.csv` | `reports/quality/log_limpieza_inventario.csv` | `reports/quality/health_score_inventario.csv` |
| Transacciones | `data/interim/transacciones_logistica_limpio.csv` | `reports/quality/log_limpieza_transacciones.csv` | `reports/quality/health_score_transacciones.csv` |
| Feedback | `data/interim/feedback_limpio.csv` | `reports/quality/log_limpieza_feedback.csv` | `reports/quality/health_score_feedback.csv` |

---

## Criterios de diseño transversales

- **No eliminar filas salvo duplicados** — Los valores extremos se tratan con winsorización, imputación o categorías explícitas, no con borrado masivo.
- **Imputación contextual** — Mediana por categoría, ciudad o canal según la variable; moda solo para categóricas corruptas con baja tasa de error.
- **Trazabilidad** — Cada imputación deja una bandera (`*_Imputado`, `*_Winsorizado`) y el campo `Registro_Confiable` permite filtrar análisis sensibles.
- **Claves de integración** — `SKU_ID` normalizado (`strip + upper`) en inventario y transacciones para el merge posterior en `src/techlogistics/pipelines/integration.py`.
