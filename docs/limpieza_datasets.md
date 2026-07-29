# Resumen de limpieza de datasets

**Challenge 02 — Fundamentos en Ciencia de Datos (EAFIT 2026-1)**

Este documento resume las transformaciones aplicadas a los tres datasets del reto y **justifica columna por columna** por qué se tomó cada decisión, qué alternativas existían y por qué no se eligieron.

Cada dataset fue procesado con un pipeline propio (`src/techlogistics/pipelines/`) que comparte la infraestructura de auditoría y trazabilidad de `src/techlogistics/quality/`.

---

## Metodología común

Todos los pipelines siguen la misma estructura:

1. **Auditoría inicial** — Health Score (Completitud, Unicidad, Consistencia, Validez).
2. **Limpieza y estandarización** — Con log de trazabilidad por transformación.
3. **Feature engineering** — Variables derivadas para el dashboard y las 5 preguntas del reto.
4. **Auditoría final** — Comparación antes/después del Health Score.
5. **Exportación** — `data/interim/` y `reports/quality/`.

### Principios que guían todas las decisiones

| Principio | Qué implica | Por qué |
|-----------|-------------|---------|
| **Conservar filas** | No borrar registros salvo duplicados reales | Perder filas oculta volumen de negocio (ingresos, ventas fantasma, NPS) |
| **Imputación contextual** | Mediana/mod por categoría, ciudad o canal | Respeta que smartphones ≠ accesorios, Bogotá ≠ Bucaramanga |
| **Mediana sobre media** | Para variables con outliers o centinelas | Un solo valor extremo no arrastra todo el imputado |
| **Categoría explícita vs. moda** | Cuando los nulos son >15–20 % | Imputar con moda fabricaría opiniones o estados logísticos inexistentes |
| **Trazabilidad** | Bandera `*_Imputado`, `Registro_Confiable` | Permite filtrar análisis sensibles y auditar ante la junta |

---

## 1. Inventario Central

| | |
|---|---|
| **Archivo origen** | `data/raw/inventario_central_v2.csv` |
| **Archivo limpio** | `data/interim/inventario_limpio.csv` |
| **Registros** | 2.500 (sin eliminación de filas) |
| **Health Score** | 93.48 → **99.46** (+5.98) |

### Resumen de transformaciones

| Paso | Columna | Acción | Registros afectados |
|------|---------|--------|---------------------|
| 1 | `SKU_ID` | Eliminación de duplicados exactos y por ID | 0 |
| 2 | `SKU_ID` | Normalización `strip + upper` | 0 |
| 3–4 | `Categoria` | Mapeo canónico + `???` → moda | 308 |
| 5 | `Bodega_Origen` | Unificación de capitalización | 1 |
| 6 | `Stock_Actual` | Negativos/nulos → mediana por categoría | 160 |
| 7 | `Costo_Unitario_USD` | Winsorización IQR por categoría | 1 |
| 8–9 | `Lead_Time_Dias` | Parseo texto → numérico + mediana por categoría | 2.500 |
| 10 | `Ultima_Revision` | Datetime + `Antiguedad_Revision_Dias` | 0 |

**Registros confiables finales:** ~69.7 % (757 con al menos una imputación).

---

### Justificación por columna — Inventario

#### `SKU_ID`

| | |
|---|---|
| **Problema** | Clave primaria del maestro; debe coincidir exactamente con transacciones para el JOIN. |
| **Decisión** | Eliminar duplicados por ID (`keep='first'`) + `strip + upper`. |
| **Por qué es la mejor opción** | Un SKU duplicado infla stock y valor de inventario. La normalización evita "SKU fantasma" por espacios o minúsculas en el merge. |
| **Alternativas descartadas** | **Eliminar duplicados sin normalizar** → el merge seguiría fallando. **Fusionar stock de duplicados** → no hay evidencia de que sean el mismo producto mal cargado; arriesga doble conteo. |

#### `Categoria`

| | |
|---|---|
| **Problema** | Variantes tipográficas (`smart-phone`, `Smartphones`) y 305 valores corruptos (`???`). |
| **Decisión** | Mapeo a nombres canónicos; `???` → NaN → imputación con **moda** (`Laptops`). |
| **Por qué es la mejor opción** | La moda preserva la distribución de clases sin inventar una categoría arbitraria. Unificar variantes evita fragmentar el análisis por categoría (Pregunta 4). |
| **Alternativas descartadas** | **Eliminar filas con `???`** → se pierde ~12 % del maestro. **Imputar con "Desconocida"** → válido para auditoría, pero rompe agregaciones que esperan categorías reales del negocio. **Modelo de clasificación** → overkill para 305 filas con contexto disponible. |

#### `Stock_Actual`

| | |
|---|---|
| **Problema** | Existencias negativas (imposibles contablemente) y nulos. |
| **Decisión** | Tratar negativos como NaN → **mediana por `Categoria`**. |
| **Por qué es la mejor opción** | El stock varía mucho entre categorías (smartphones vs. accesorios); la mediana global distorsionaría categorías de bajo volumen. La mediana es robusta ante outliers de stock en la misma categoría. |
| **Alternativas descartadas** | **Poner 0** → subestima capital inmovilizado. **Media por categoría** → sensible a SKUs con stock extremo. **Eliminar filas** → pierde SKUs válidos con un solo campo erróneo. |

#### `Costo_Unitario_USD`

| | |
|---|---|
| **Problema** | Rango documentado: $0.01 – $850.000; valores fuera de IQR por categoría. |
| **Decisión** | **Winsorización** (`clip` por IQR dentro de cada categoría). No eliminación. |
| **Por qué es la mejor opción** | El producto sigue siendo una venta válida; lo que falló fue la captura del precio. Winsorizar acota el extremo sin perder la fila (Guía Senior Toolkit). Por categoría porque un monitor y un accesorio tienen escalas distintas. |
| **Alternativas descartadas** | **Eliminar outliers** → pierde SKUs reales caros/baratos. **Mediana global** → aplana diferencias entre categorías. **Dejar NaN** → rompe `Valor_Inventario` y el merge con transacciones. |

#### `Punto_Reorden`

| | |
|---|---|
| **Problema** | Sin errores detectados en auditoría. |
| **Decisión** | **Sin transformación.** |
| **Por qué** | Valores consistentes; se usa directamente en `Ratio_Stock_Reorden` para detectar sobre-stock (Pregunta 4). |
| **Alternativas** | Ninguna necesaria en esta fase. |

#### `Lead_Time_Dias`

| | |
|---|---|
| **Problema** | Texto mixto (`Inmediato`, `25-30 días`) y ~16 % nulos. |
| **Decisión** | Parseo: `Inmediato` → 0; rangos → promedio; nulos → **mediana por categoría**. Se conserva `Lead_Time_Dias_Original`. |
| **Por qué es la mejor opción** | Convierte a numérico sin perder semántica. El lead time depende del tipo de producto (importado vs. accesorio local). Eliminar 16 % del maestro sería inaceptable. |
| **Alternativas descartadas** | **Moda del texto** → no permite comparar ni promediar. **Mediana global** → mezcla categorías con dinámicas distintas. **Eliminar nulos** → −403 SKUs (~16 %). |

#### `Bodega_Origen`

| | |
|---|---|
| **Problema** | Capitalización inconsistente (`norte` vs `Norte`). |
| **Decisión** | Mapeo de variantes conocidas; **preservar** códigos externos (`ZONA_FRANCA`, `BOD-EXT-99`). |
| **Por qué es la mejor opción** | Norte/Sur/Occidente son la misma bodega con distinta captura. Los códigos externos son bodegas tercerizadas reales — colapsarlos ocultaría operación relevante (Pregunta 5). |
| **Alternativas descartadas** | **Mapear todo a Norte/Sur/Occidente** → oculta operación en zona franca. **Eliminar códigos raros** → pierde señal de riesgo operativo. |

#### `Ultima_Revision`

| | |
|---|---|
| **Problema** | Fecha en texto; necesaria para análisis temporal. |
| **Decisión** | `to_datetime` + derivar `Antiguedad_Revision_Dias`. |
| **Por qué es la mejor opción** | Insumo directo de Pregunta 5 (bodegas que "operan a ciegas"). Sin datetime no hay antigüedad. |
| **Alternativas descartadas** | **Imputar fecha** → inventaría cuándo se revisó el stock. **Ignorar columna** → pierde KPI clave del reto. |

---

## 2. Transacciones Logísticas

| | |
|---|---|
| **Archivo origen** | `data/raw/transacciones_logistica_v2.csv` |
| **Archivo limpio** | `data/interim/transacciones_logistica_limpio.csv` |
| **Registros** | 10.000 (sin eliminación de filas) |
| **Health Score** | 94.43 → **99.69** (+5.26) |

### Resumen de transformaciones

| Paso | Columna | Acción | Registros afectados |
|------|---------|--------|---------------------|
| 1 | `Transaccion_ID` | Eliminación de duplicados | 0 |
| 2–3 | `Fecha_Venta` | Datetime + calendario derivado | 10.000 |
| 4 | `SKU_ID` | `strip + upper` | 0 |
| 5–6 | `Ciudad_Destino` | Mapeo canónico + canales → NaN | 1.293 |
| 7 | `Cantidad_Vendida` | Centinela −5 → mediana global | 100 |
| 8 | `Costo_Envio` | Mediana por `Canal_Venta` | 834 |
| 9 | `Estado_Envio` | NaN → `Sin_Informacion` | 1.683 |
| 10–11 | `Tiempo_Entrega_Real` | Centinela 999/negativos → mediana por ciudad | 50 |

**Registros confiables finales:** ~79.4 % (2.065 con al menos una bandera).

---

### Justificación por columna — Transacciones

#### `Transaccion_ID`

| | |
|---|---|
| **Problema** | Clave atómica de la venta; duplicados inflarían ingresos. |
| **Decisión** | `drop_duplicates` exactos + por ID (`keep='first'`). |
| **Por qué es la mejor opción** | Una transacción = una línea de ingreso. Duplicar ID duplica revenue en Preguntas 1 y 3. |
| **Alternativas descartadas** | **Conservar duplicados** → KPIs monetarios inflados. **Promediar montos** → no tiene sentido de negocio para IDs repetidos. |

#### `SKU_ID`

| | |
|---|---|
| **Problema** | Llave del JOIN con inventario. |
| **Decisión** | `strip + upper` (igual que inventario). |
| **Por qué es la mejor opción** | Un espacio o minúscula genera falso SKU fantasma en el merge. |
| **Alternativas descartadas** | **No normalizar** → ~17 % de SKUs fantasma artificial. |

#### `Fecha_Venta`

| | |
|---|---|
| **Problema** | Texto `dd/mm/yyyy`; impide análisis temporal. |
| **Decisión** | `to_datetime` con formato fijo + fallback flexible; derivar año/mes/trimestre. |
| **Por qué es la mejor opción** | Habilita estacionalidad, antigüedad y Pregunta 3 (ingreso en riesgo por mes). |
| **Alternativas descartadas** | **Dejar como string** → no se puede ordenar ni agrupar por tiempo. **Eliminar fechas inválidas** → 0 irrecuperables aquí, pero el enfoque `coerce` es más seguro a futuro. |

#### `Cantidad_Vendida`

| | |
|---|---|
| **Problema** | 100 registros con valor **−5** idéntico; distribución uniforme entre estados de envío (solo 11 son `Devuelto`). |
| **Decisión** | Tratar −5 como **centinela de error** → NaN → **mediana global (7)** + bandera `Cantidad_Sospechosa_Origen`. |
| **Por qué es la mejor opción** | Una devolución real se concentraría en `Devuelto`, no repartida uniformemente. −5 es código ERP ("cantidad no registrada"), no evento de negocio. La mediana es robusta; la bandera permite excluir del análisis. |
| **Alternativas descartadas** | **Interpretar como devolución** → sesga estados logísticos. **Poner 1** → arbitrario. **Eliminar 100 filas** → pierde ~1 % de ingresos. **Mediana por canal** → pocos casos por segmento; mediana global más estable con n=100. |

#### `Precio_Venta_Final`

| | |
|---|---|
| **Problema** | Sin violaciones de reglas de negocio en auditoría (todos > 0). |
| **Decisión** | **Sin transformación.** |
| **Por qué** | Valores válidos; base de `Ingreso_Bruto` y margen. |
| **Alternativas** | Winsorización posible si hubiera outliers extremos, pero no se detectaron reglas rotas. |

#### `Costo_Envio`

| | |
|---|---|
| **Problema** | ~8.3 % nulos (834 filas). |
| **Decisión** | **Mediana por `Canal_Venta`** + fallback mediana global. |
| **Por qué es la mejor opción** | Online, retail y mayorista tienen estructuras de flete distintas. La mediana resiste outliers de envío express o gratis. |
| **Alternativas descartadas** | **Mediana global** → mezcla canales con costos muy distintos. **Cero** → subestima costo logístico en Pregunta 1. **Media** → sensible a envíos atípicos caros. **Eliminar nulos** → pierde 834 ventas. |

#### `Tiempo_Entrega_Real`

| | |
|---|---|
| **Problema** | Centinela **999** ("sin dato") y algunos negativos. |
| **Decisión** | Tratar 999 y negativos como NaN → **mediana por `Ciudad_Destino`**. |
| **Por qué es la mejor opción** | 999 días multiplicaría por ~50 el promedio y destruiría Pregunta 2 (correlación tiempo–NPS). El tiempo de entrega es geográfico: Medellín ≠ Barranquilla. |
| **Alternativas descartadas** | **Dejar 999** → KPI logístico irreparable. **Mediana global** → ignora distancia real. **Cap en 15 días (SLA)** → oculta retrasos reales legítimos. **Eliminar filas** → pierde ventas válidas. |

#### `Estado_Envio`

| | |
|---|---|
| **Problema** | **16.8 % nulos** (1.683 filas). |
| **Decisión** | NaN → categoría explícita **`Sin_Informacion`** (no moda). |
| **Por qué es la mejor opción** | Con casi 1 de cada 6 envíos sin estado, imputar `Entregado` o `Retrasado` inventaría desempeño logístico. La ausencia de registro es en sí un hallazgo de invisibilidad operativa. |
| **Alternativas descartadas** | **Moda (`Entregado`)** → inflaría tasa de éxito ~17 pp. **Eliminar nulos** → pierde 16.8 % de transacciones. **Imputar por canal** → sigue siendo inventar un estado no medido. |

#### `Ciudad_Destino`

| | |
|---|---|
| **Problema** | Variantes (`BOG`, `Bogotá`) y **contaminación por canal** (`Ventas_Web`, `online`) en columna geográfica. |
| **Decisión** | Mapeo canónico; canales inválidos → **NaN** + bandera `Ciudad_Invalida_Origen`. **No imputar** ciudad. |
| **Por qué es la mejor opción** | Imputar `Ventas_Web` como ciudad fabricaría un mercado inexistente. Mejor excluir del geo-análisis que mentir. Unificar BOG/Bogotá evita subestimar correlación por ciudad (Pregunta 2). |
| **Alternativas descartadas** | **Imputar moda (Bogotá)** → 1.290 filas con ciudad falsa. **Categoría "Online"** → válida como alternativa, pero elegimos NaN + bandera para no mezclar geografía con canal. **Eliminar filas** → −13 % del dataset. |

#### `Canal_Venta`

| | |
|---|---|
| **Problema** | Sin inconsistencias detectadas. |
| **Decisión** | **Sin transformación**; se usa como segmento para imputar `Costo_Envio`. |
| **Por qué** | Valores estables (`Online`, `Retail`, etc.). |
| **Alternativas** | Ninguna necesaria. |

---

## 3. Feedback de Clientes

| | |
|---|---|
| **Archivo origen** | `data/raw/feedback_clientes_v2.csv` |
| **Archivo limpio** | `data/interim/feedback_limpio.csv` |
| **Registros** | 4.500 → **4.000** (−500 duplicados intencionales) |
| **Health Score** | 92.40 → **100.00** (+7.60) |

### Resumen de transformaciones

| Paso | Columna | Acción | Registros afectados |
|------|---------|--------|---------------------|
| 1 | `Feedback_ID` | Eliminación de duplicados | **500** |
| 2 | `Rating_Producto` | Fuera [1,5] → mediana (3) | 27 |
| 3 | `Edad_Cliente` | > 100 → mediana (50) | 20 |
| 4 | `Recomienda_Marca` | NaN → `Sin_Respuesta` | 999 |
| 5 | `Ticket_Soporte_Abierto` | Normalización a booleano | 955 |
| 6 | `Comentario_Texto` | Placeholders → `Sin_Comentario` | 1.152 |

**Registros confiables finales:** ~99.0 % (40 con imputación numérica).

---

### Justificación por columna — Feedback

#### `Feedback_ID`

| | |
|---|---|
| **Problema** | 500 IDs duplicados (trampa intencional del reto). |
| **Decisión** | **Eliminar duplicados** por ID (`keep='first'`). |
| **Por qué es la mejor opción** | Un mismo formulario contado dos veces duplica la voz del cliente en NPS agregado y en ratios de soporte. |
| **Alternativas descartadas** | **Conservar duplicados** → NPS y conteos inflados +11 %. **Promediar ratings del ID** → asume que ambos formularios son la misma opinión con ruido; menos transparente que `keep='first'`. |

#### `Transaccion_ID`

| | |
|---|---|
| **Problema** | Clave de enlace con transacciones; sin errores de formato. |
| **Decisión** | **Sin transformación.** |
| **Por qué** | Se usa en el merge; no requiere limpieza adicional. |
| **Alternativas** | Validar existencia en transacciones (left join) — eso ocurre en integración, no aquí. |

#### `Rating_Producto`

| | |
|---|---|
| **Problema** | 27 valores fuera de escala 1–5 (ej. 45, 99) — errores de captura. |
| **Decisión** | Fuera de rango → NaN → **mediana (3)** + bandera `Rating_Producto_Imputado`. |
| **Por qué es la mejor opción** | Escala ordinal 1–5; 45 no es "muy insatisfecho", es basura de teclado. La mediana no se arrastra por outliers extremos. |
| **Alternativas descartadas** | **Clip a [1,5]** → 99 se convertiría en 5, fabricando un promotor. **Eliminar filas** → pierde el resto de la opinión (NPS, logística). **Media** → sensible a valores imposibles antes de limpiar. |

#### `Rating_Logistica`

| | |
|---|---|
| **Problema** | En auditoría: 0 fuera de rango; sin nulos reportados. |
| **Decisión** | **Sin transformación.** |
| **Por qué** | Datos limpios; entra en `Rating_Promedio` y análisis de Pregunta 2. |
| **Alternativas** | Misma lógica que `Rating_Producto` si aparecieran outliers en otro lote. |

#### `Satisfaccion_NPS`

| | |
|---|---|
| **Problema** | Rango −100 a 100 es el **estándar NPS**; no es error. |
| **Decisión** | **Sin recorte.** Derivar `Segmento_NPS` (Promotor/Pasivo/Detractor). |
| **Por qué es la mejor opción** | "Normalizar" recortando destruiría la métrica. La normalización correcta es **categorizar**, no escalar. |
| **Alternativas descartadas** | **Min-max a [0,1]** → pierde interpretabilidad ejecutiva. **Eliminar negativos** → sesga NPS hacia arriba. |

#### `Edad_Cliente`

| | |
|---|---|
| **Problema** | Hasta 195 años (23 outliers IQR + 20 imposibles > 100). |
| **Decisión** | > 100 → NaN → **mediana (50)** + bandera. |
| **Por qué es la mejor opción** | 195 no es edad plausible; es error de captura. Mediana robusta ante extremos. La edad no es KPI central del reto — no justifica modelo complejo. |
| **Alternativas descartadas** | **Clip a 100** → concentra valores falsos en un techo artificial. **Eliminar filas** → pierde feedback válido en otras columnas. **Imputar por segmento NPS** → sobreingeniería para variable secundaria. |

#### `Recomienda_Marca`

| | |
|---|---|
| **Problema** | ~25 % nulos + variantes (`SI`, `Sí`, `Maybe`). |
| **Decisión** | Mapeo a Si/No/Talvez; NaN → **`Sin_Respuesta`**. |
| **Por qué es la mejor opción** | Imputar moda inventaría la opinión de 1/4 de clientes. `Sin_Respuesta` preserva el patrón de no respuesta como hallazgo. |
| **Alternativas descartadas** | **Moda (`Si`)** → +25 pp de recomendación ficticia. **Eliminar nulos** → −999 opiniones. **Imputar aleatorio según distribución** → no auditable. |

#### `Ticket_Soporte_Abierto`

| | |
|---|---|
| **Problema** | Mezcla `Sí`/`No`/`1`/`0` + fallos de encoding en tildes. |
| **Decisión** | Mapeo por **primera letra** en minúscula → booleano; sin mapear → `False`. |
| **Por qué es la mejor opción** | Recupera registros corruptos por tilde (`S`/`s` de Sí). Unificado para KPI de soporte (Pregunta 4 y 5). |
| **Alternativas descartadas** | **Solo mapeo exacto Sí/No** → pierde filas por encoding. **Dejar como string** → no permite `.mean()` por categoría. **Eliminar ambiguos** → pierde señal de riesgo. |

#### `Comentario_Texto`

| | |
|---|---|
| **Problema** | Placeholders (`N/A`, `---`, vacíos) mezclados con nulos reales. |
| **Decisión** | Unificar placeholders y NaN → **`Sin_Comentario`**. |
| **Por qué es la mejor opción** | Texto libre de baja relevancia cuantitativa en el reto; no se hace NLP. Evita fragmentar "sin opinión" en 4 categorías distintas. |
| **Alternativas descartadas** | **Eliminar filas sin comentario** → pierde ratings numéricos valiosos. **Dejar N/A y --- separados** → ruido en conteos. **Imputar "Sin datos" vs "Sin_Comentario"** → diferencia sin valor analítico aquí. |

---

## 4. Integración (Fuente Única de Verdad)

Además de la limpieza por dataset, el merge en `integration.py` toma decisiones que afectan columnas cruzadas:

| Columna / operación | Decisión | Por qué es la mejor opción | Alternativas descartadas |
|---------------------|----------|----------------------------|--------------------------|
| **Merge transacciones + inventario** (`SKU_ID`, `how='left'`) | Conservar **todas** las ventas aunque el SKU no exista en inventario | Eliminarlas subestimaría ingreso real; Pregunta 3 exige cuantificar venta invisible | **Inner join** → oculta 17.5 % del ingreso. **Eliminar SKU fantasma** → contradice el objetivo del reto |
| **`Costo_Unitario_USD` en SKU fantasma** | Imputar con precio × (1 − margen % mediano observado) | Permite calcular margen sin excluir 1.751 filas; imputación **flexible por transacción**, no valor fijo | **Dejar NaN** → excluye 17.5 % del análisis de rentabilidad. **Costo = 0** → margen 100 % ficticio. **Costo fijo promedio** → menos preciso que tasa de margen |
| **Feedback por `Transaccion_ID`** | Agregar a 1 fila/transacción (mean NPS, any ticket) + merge `left` | Evita multiplicar filas; deja visible que solo ~40 % tiene feedback | **Merge sin agregar** → duplica transacciones. **Imputar NPS** → fabricar satisfacción |
| **`Ingreso_En_Riesgo`** | `Ingreso_Bruto` donde `SKU_Fantasma` | Separa ingreso controlado vs. en riesgo sin mezclar | **Sumar al ingreso normal** → oculta el problema |
| **`Margen_Utilidad`** | `(Precio − Costo) × Cantidad` a nivel línea | Consistente con `Ingreso_Bruto`; corrige subestimación ~7× vs. margen unitario | **Solo margen unitario** → subestima pérdida total |

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
| Integración | `data/processed/fuente_unica_verdad.csv` | `reports/integration/log_integracion.csv` | — |

---

## Criterios de diseño transversales

- **No eliminar filas salvo duplicados** — Los extremos se winsorizan, imputan o categorizan explícitamente.
- **Imputación contextual** — Mediana/mod por categoría, ciudad o canal según la variable.
- **Alta tasa de nulos → categoría explícita, no moda** — Aplica a `Estado_Envio` (16.8 %) y `Recomienda_Marca` (25 %).
- **Trazabilidad** — Bandera `*_Imputado` / `Registro_Confiable` para filtrar análisis auditable.
- **Claves de integración** — `SKU_ID` normalizado en inventario y transacciones.

---

## Guía rápida: ¿qué imputación usar y cuándo?

| Situación | Estrategia elegida | Cuándo considerar otra |
|-----------|-------------------|------------------------|
| Variable numérica con outliers | Mediana (por grupo) | Media si distribución simétrica y sin centinelas |
| Variable categórica corrupta (<15 % error) | Moda | Categoría "Desconocido" si se quiere auditar aparte |
| Variable categórica con >15 % nulos | Categoría explícita (`Sin_*`) | No imputar con moda |
| Centinela de sistema (−5, 999) | NaN + imputación + bandera | Nunca interpretar como valor real |
| Precio/costo extremo | Winsorización IQR | Eliminar solo si se confirma error de captura al 100 % |
| Texto libre no analizado | Unificar placeholders | NLP/sentimiento si el reto lo exigiera |
| Clave de merge | Normalizar, nunca eliminar | — |

---

## Solución al dilema del SKU Fantasma (resumen)

Al integrar transacciones con inventario, **~17.5 % de las ventas** (~1.751 filas) tienen un `SKU_ID` que no existe en el maestro. Sin contexto del ERP no se puede saber si son productos nuevos no catalogados o errores de digitación.

**Decisión adoptada:**

1. **Conservar todas las ventas** — merge `left` sobre `SKU_ID`; no se eliminan filas ni se hace inner join.
2. **Marcar el fenómeno** — `SKU_Fantasma = True` cuando no hay match en inventario (`Categoria` nula).
3. **Cuantificar el riesgo financiero** — `Ingreso_En_Riesgo = Ingreso_Bruto` solo donde `SKU_Fantasma` (Pregunta 3).
4. **Estimar margen sin inventar certeza** — para SKU fantasma se imputa costo con la **tasa de margen mediana** (~23.6 %) de las ventas sí controladas: `Costo ≈ Precio × (1 − margen_mediano)`. La bandera `Costo_Fantasma_Imputado` distingue costo real vs. estimado.

**En una frase:** no se oculta la venta invisible, se mide su ingreso en riesgo por separado, y el margen total incluye una estimación prudente — siempre auditable — en lugar de excluir el 17.5 % del análisis o fingir margen del 100 %.
