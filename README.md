# TechLogistics S.A.S. — Data Hub & Sistema de Soporte a la Decisión (DSS)

Challenge 02 — Curso Fundamentos en Ciencia de Datos (Maestría), Universidad EAFIT, 2026-1.

TechLogistics S.A.S. (ficticio) tiene tres sistemas que no hablan el mismo idioma: ERP de
Inventarios, Logística y Feedback de clientes. Este proyecto hace la curaduría de los tres
datasets, los integra en una **Sola Fuente de Verdad**, responde 5 preguntas de alta gerencia
y genera recomendaciones estratégicas con IA (Groq / Llama 3.3), todo dentro de un dashboard
en Streamlit.

## Estructura del repositorio

```
├── app.py                          # Dashboard Streamlit (DSS)
├── generar_informe.py              # Genera el PDF de hallazgos para la junta
├── requirements.txt
├── data/
│   ├── inventario_central_v2.csv       # Datos crudos (entregados por el reto)
│   ├── transacciones_logistica_v2.csv
│   ├── feedback_clientes_v2.csv
│   ├── *_limpio.csv                    # Salidas curadas de cada pipeline
│   └── fuente_unica_verdad.csv         # Merge integrado + variables derivadas
├── processing/
│   ├── common.py                   # Infraestructura compartida: LogLimpieza, Health Score
│   ├── lg_transactions.py          # Pipeline de transacciones logísticas
│   ├── inventory.py                # Pipeline de inventario central
│   ├── feedback.py                 # Pipeline de feedback de clientes
│   └── integracion.py              # Merge estratégico + feature engineering (Fase 2.2)
└── reports/
    ├── health_score_*.csv          # Health Score antes/después por dataset
    ├── log_limpieza_*.csv          # Trazabilidad de cada transformación
    ├── log_integracion.csv
    └── Informe_Hallazgos_TechLogistics.pdf   # Documento de hallazgos para la junta
```

## Instalación

Requiere Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # En Windows (PowerShell/CMD)
pip install -r requirements.txt
```

## Cómo replicar el análisis

1. **Ejecutar los pipelines de curación por separado** (opcional — `app.py` los invoca
   automáticamente, pero puedes correrlos sueltos para ver el Health Score en consola):

   ```bash
   python processing/lg_transactions.py
   python processing/inventory.py
   python processing/feedback.py
   python processing/integracion.py
   ```

   Cada uno imprime el Health Score antes/después y exporta a `data/` y `reports/`.

2. **Levantar el dashboard:**

   ```bash
   streamlit run app.py
   ```

   La primera carga ejecuta los 3 pipelines y construye la Fuente Única de Verdad
   (`@st.cache_data`, así que las siguientes interacciones son instantáneas).

3. **Generar el PDF de hallazgos** (se regenera con los datos más recientes):

   ```bash
   python generar_informe.py
   ```

   Escribe `reports/Informe_Hallazgos_TechLogistics.pdf`.

## Groq API Key (módulo de IA)

La pestaña **🤖 Recomendaciones IA** del dashboard llama al modelo `llama-3.3-70b-versatile`
de Groq para generar 3 párrafos de recomendación estratégica a partir del resumen
estadístico del subconjunto filtrado.

1. Crea una cuenta gratuita en https://console.groq.com/
2. Genera una API Key en la sección *API Keys*.
3. Pégala en el campo **Groq API Key** de la barra lateral del dashboard (no se persiste en
   disco; solo vive en la sesión de Streamlit).

Sin API Key, el resto del dashboard funciona con normalidad — solo esa pestaña queda
inactiva con un aviso.

## Decisiones de limpieza (resumen)

El detalle línea por línea, con justificación, está en `reports/log_limpieza_*.csv`
(descargable también desde el dashboard). Las decisiones más relevantes:

- **SKU Fantasma** (`processing/integracion.py`): 480 de 2,889 SKUs de transacciones
  (~17%) no existen en el inventario. Se conservan todas las ventas (`merge how='left'`)
  para no ocultar el fenómeno y se cuantifican aparte en `Ingreso_En_Riesgo` (Pregunta 3).
  Para el margen, en vez de dejarlas en `NaN` (lo que las excluía del 100% del análisis de
  rentabilidad), se les imputa un costo estimado con la tasa de margen **mediana** de las
  transacciones con costo real, aplicada al precio de cada línea — una imputación flexible
  que varía por transacción en vez de un valor fijo, y queda marcada con la bandera
  `Costo_Fantasma_Imputado` para auditar qué filas usan costo real vs. estimado.
- **Margen a nivel de línea, no por unidad**: `Margen_Utilidad` es
  `(Precio_Venta_Final - Costo_Unitario_USD) x Cantidad_Vendida`, igual que `Ingreso_Bruto`.
  La primera versión calculaba solo el margen unitario, lo que subestimaba la pérdida total
  por un factor de ~7x al no multiplicar por la cantidad vendida de cada línea.
- **Centinelas de error**: `Cantidad_Vendida = -5` y `Tiempo_Entrega_Real = 999` son
  códigos de error del sistema origen, no eventos de negocio; se tratan como nulos y se
  imputan con la mediana (robusta ante outliers), segmentada por canal o ciudad según el
  caso.
- **Duplicados intencionales de Feedback**: 500 `Feedback_ID` repetidos se eliminan
  (`keep='first'`) para no inflar el NPS agregado.
- **Escala de NPS**: el rango observado (-100 a 100) **es** el estándar de NPS: no se
  recorta, se deriva un `Segmento_NPS` (Promotor/Pasivo/Detractor) para volverlo legible.
- **Nulos de alto porcentaje** (`Estado_Envio` 16.8%, `Recomienda_Marca` 25%): no se
  imputan con la moda — se marcan como categoría explícita ("Sin_Informacion" /
  "Sin_Respuesta"), porque inventar una respuesta a esa escala sesgaría el KPI de negocio.
- **Costos atípicos de inventario** ($0.01–$850k): se winsorizan (`clip`) por IQR dentro
  de cada categoría, no se eliminan filas — el producto sigue siendo una venta válida.

## Variables derivadas (Feature Engineering)

`Margen_Utilidad(_Pct)`, `Brecha_Entrega_vs_Prometido` (vs. SLA de 15 días),
`Ingreso_En_Riesgo`, `Ratio_Soporte_Categoria`, `Antiguedad_Revision_Dias`,
`Valor_Inventario`, `Alta_Disponibilidad`, `Segmento_NPS`, entre otras — ver
`processing/integracion.py` y los pipelines individuales para el detalle completo.

## Autor

Santiago Betancur — Maestría en Ciencia de Datos, Universidad EAFIT.
