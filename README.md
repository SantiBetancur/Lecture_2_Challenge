# TechLogistics S.A.S. — Data Hub & Sistema de Soporte a la Decisión (DSS)

Challenge 02 — Curso Fundamentos en Ciencia de Datos (Maestría), Universidad EAFIT, 2026-1.

TechLogistics S.A.S. (ficticio) tiene tres sistemas que no hablan el mismo idioma: ERP de
Inventarios, Logística y Feedback de clientes. Este proyecto hace la curaduría de los tres
datasets, los integra en una **Sola Fuente de Verdad**, responde 5 preguntas de alta gerencia
y genera recomendaciones estratégicas con IA (Groq / Llama 3.3), todo dentro de un dashboard
en Streamlit.

## Estructura del repositorio

```
├── app.py                              # Dashboard Streamlit (DSS)
├── pyproject.toml                      # Paquete Python instalable
├── Makefile                            # Atajos: make pipeline, make app, make report
├── requirements.txt
│
├── data/
│   ├── raw/                            # Datos crudos (solo lectura)
│   ├── interim/                        # Datasets limpios por pipeline
│   └── processed/                      # Fuente única de verdad
│
├── src/techlogistics/                  # Paquete principal
│   ├── config.py                       # Rutas y constantes centralizadas
│   ├── io.py                           # Carga de CSV
│   ├── quality/                        # Health Score y LogLimpieza
│   └── pipelines/                      # Curación e integración
│       ├── inventory.py
│       ├── transactions.py
│       ├── feedback.py
│       └── integration.py
│
├── scripts/
│   ├── run_pipeline.py                 # Ejecuta todos los pipelines
│   └── generate_report.py              # Genera PDF de hallazgos
│
├── reports/
│   ├── quality/                        # health_score_*.csv, log_limpieza_*.csv
│   ├── integration/                    # log_integracion.csv
│   └── deliverables/                   # Informe_Hallazgos_TechLogistics.pdf
│
└── docs/
    ├── limpieza_datasets.md            # Resumen de limpieza por dataset
    ├── health_score_report.md
    └── references/                     # Material de referencia del curso
```

## Instalación

Requiere Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell/CMD)
pip install -e .
# o: pip install -r requirements.txt
```

## Cómo replicar el análisis

1. **Ejecutar los pipelines de curación** (opcional — `app.py` los invoca automáticamente):

   ```bash
   make pipeline
   # o: python scripts/run_pipeline.py
   # o por etapa: python scripts/run_pipeline.py --stage inventory transactions
   ```

   Cada pipeline imprime el Health Score antes/después y exporta a `data/` y `reports/`.

2. **Levantar el dashboard:**

   ```bash
   make app
   # o: streamlit run app.py
   ```

   La primera carga ejecuta los 3 pipelines y construye la Fuente Única de Verdad
   (`@st.cache_data`, así que las siguientes interacciones son instantáneas).

3. **Generar el PDF de hallazgos:**

   ```bash
   make report
   # o: python scripts/generate_report.py
   ```

   Escribe `reports/deliverables/Informe_Hallazgos_TechLogistics.pdf`.

## Groq API Key (módulo de IA)

La pestaña **Recomendaciones IA** del dashboard llama al modelo `llama-3.3-70b-versatile`
de Groq para generar 3 párrafos de recomendación estratégica a partir del resumen
estadístico del subconjunto filtrado.

1. Crea una cuenta gratuita en https://console.groq.com/
2. Genera una API Key en la sección *API Keys*.
3. Pégala en el campo **Groq API Key** de la barra lateral del dashboard (no se persiste en
   disco; solo vive en la sesión de Streamlit).

Sin API Key, el resto del dashboard funciona con normalidad — solo esa pestaña queda
inactiva con un aviso.

## Decisiones de limpieza (resumen)

El detalle línea por línea, con justificación, está en `reports/quality/log_limpieza_*.csv`
(descargable también desde el dashboard) y en `docs/limpieza_datasets.md`. Las decisiones
más relevantes:

- **SKU Fantasma** (`integration.py`): 480 de 2,889 SKUs de transacciones
  (~17%) no existen en el inventario. Se conservan todas las ventas (`merge how='left'`)
  para no ocultar el fenómeno y se cuantifican aparte en `Ingreso_En_Riesgo` (Pregunta 3).
- **Centinelas de error**: `Cantidad_Vendida = -5` y `Tiempo_Entrega_Real = 999` son
  códigos de error del sistema origen, no eventos de negocio.
- **Duplicados intencionales de Feedback**: 500 `Feedback_ID` repetidos se eliminan
  (`keep='first'`) para no inflar el NPS agregado.
- **Nulos de alto porcentaje**: no se imputan con la moda — se marcan como categoría
  explícita (`Sin_Informacion` / `Sin_Respuesta`).

## Autor

Santiago Betancur — Maestría en Ciencia de Datos, Universidad EAFIT.
