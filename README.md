# TechLogistics S.A.S. — Data Hub & Sistema de Soporte a la Decisión (DSS)

Challenge 02 — Curso **Fundamentos en Ciencia de Datos** (Maestría), Universidad **EAFIT**, 2026-1.

TechLogistics S.A.S. (ficticio) opera con tres sistemas desconectados: **Inventario**, **Logística** y **Feedback**. Este repositorio:

1. **Curaduría trazable** de los tres datasets (Health Score, logs de limpieza, decisiones éticas).
2. **Integración** en una Sola Fuente de Verdad.
3. **Dashboard Streamlit** con sidebar, pestañas y descargas.
4. **5 preguntas estratégicas** de alta gerencia con gráficos y conclusiones.
5. **Módulo IA** (Llama-3 vía Groq) sobre datos filtrados.
6. **Informe PDF** de consultoría para la junta directiva.

---

## Requisitos

| Requisito | Versión |
|-----------|---------|
| Python | 3.10 o superior |
| pip | reciente |
| SO | Windows, macOS o Linux |

Opcional: cuenta gratuita en [Groq](https://console.groq.com/) para la pestaña **Recomendaciones IA**.

---

## Instalación (usuario externo)

```bash
# 1. Clonar o descomprimir el repositorio
cd Lecture_2_Challenge

# 2. Entorno virtual (recomendado)
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# 3. Instalar dependencias y paquete local
pip install -e .
# Alternativa: pip install -r requirements.txt
```

---

## Cómo replicar el análisis completo

### Paso 1 — Ejecutar pipelines de limpieza e integración

Genera datasets limpios, Health Score, logs de trazabilidad y la Fuente Única de Verdad.

```bash
make pipeline
# equivalente:
python scripts/run_pipeline.py
```

Salidas principales:

- `data/interim/*.csv` — datasets limpios  
- `data/processed/fuente_unica_verdad.csv` — merge estratégico  
- `reports/quality/health_score_*.csv` — Health Score antes/después  
- `reports/quality/log_limpieza_*.csv` — cada transformación documentada  

Ejecutar etapas individuales:

```bash
python scripts/run_pipeline.py --stage inventory transactions feedback integration
```

### Paso 2 — Levantar el dashboard Streamlit

```bash
make app
# equivalente:
streamlit run app.py
```

- **Sidebar:** navegación por secciones, filtros (Preguntas Estratégicas / IA), **descargas** (log CSV, trazabilidad TXT, PDF).
- **Pestañas:** Dashboard (Resumen / Calidad / Datos), Transacciones, Inventario, Feedback, Preguntas Estratégicas, etc.
- La primera carga ejecuta los pipelines (`@st.cache_data`); las siguientes interacciones son rápidas.

### Paso 3 — Generar el informe PDF para la junta directiva

```bash
make report
# equivalente:
python scripts/generate_report.py
```

Genera en la **raíz del proyecto**:

`Informe_Consultoria_TechLogistics_Junta_Directiva_Hallazgos_Estrategicos.pdf`

Incluye gráficas alineadas al dashboard, Health Score, las 5 preguntas estratégicas y narrativa ejecutiva. También se copia a `reports/deliverables/`.

### Paso 4 (opcional) — Recomendaciones con IA

1. Obtener API Key en https://console.groq.com/  
2. Pegarla en el sidebar **IA (Groq · Llama-3)**  
   - O definir `GROQ_API_KEY` en `.streamlit/secrets.toml`  
3. Ir a **Recomendaciones IA**, ajustar filtros y generar 3 párrafos en streaming.

---

## Estructura del repositorio

```
├── app.py                              # Dashboard Streamlit (DSS)
├── Informe_Consultoria_TechLogistics_Junta_Directiva_Hallazgos_Estrategicos.pdf
├── pyproject.toml
├── Makefile
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                            # CSV originales (solo lectura)
│   ├── interim/                        # Datasets limpios (generados)
│   └── processed/                      # Fuente única de verdad (generada)
│
├── src/techlogistics/
│   ├── config.py                       # Rutas centralizadas
│   ├── quality/                        # Health Score, LogLimpieza
│   └── pipelines/                      # inventory, transactions, feedback, integration
│
├── scripts/
│   ├── run_pipeline.py
│   └── generate_report.py
│
├── reports/
│   ├── quality/                        # health_score_*, log_limpieza_* (generados)
│   └── deliverables/                   # copia del PDF
│
└── docs/
    ├── limpieza_datasets.md            # Justificación ética columna por columna
    └── health_score_report.md
```

---

## Streamlit — criterios de usabilidad

| Elemento | Dónde |
|----------|--------|
| `st.sidebar` | Navegación, filtros globales, descargas, API Groq |
| `st.tabs` | Dashboard principal, auditoría por dataset, análisis por fuente |
| Descarga log de limpieza | Sidebar **Descargas** + pestaña **Calidad y trazabilidad** |
| Descarga trazabilidad TXT | Incluye Health Score, nulidad, outliers, anexo `limpieza_datasets.md` |
| Descarga PDF | Sidebar (si existe; generar con `make report`) |

---

## Trazabilidad y decisión ética (resumen)

Documentación completa: `docs/limpieza_datasets.md` y logs CSV.

| Decisión | Criterio |
|----------|----------|
| **Eliminar** | Solo duplicados reales (500 feedback duplicados) |
| **Mediana** | Outliers, centinelas (-5, 999), variables asimétricas |
| **Moda** | Categorías corruptas en bajo % |
| **Categoría explícita** | Nulos masivos (>15 %): `Sin_Informacion`, `Sin_Respuesta` |
| **Winsorización IQR** | Costos extremos por categoría (sin borrar SKUs) |
| **Conservar filas** | Transacciones e inventario: 0 filas eliminadas por limpieza |

**Health Score (ejemplo):**

| Dataset | Antes | Después |
|---------|-------|---------|
| Inventario | 93.5 | 99.5 |
| Transacciones | 94.4 | 99.7 |
| Feedback | 92.4 | 100.0 |

---

## Buenas prácticas de código

- **PEP 8:** nombres en `snake_case`, funciones con docstrings, módulos separados por responsabilidad.
- **Config centralizado:** `src/techlogistics/config.py` (sin rutas hardcodeadas dispersas).
- **Manejo de excepciones:** pipelines y scripts capturan `FileNotFoundError`, `ValueError`, `PermissionError`; la app muestra mensajes accionables al usuario.
- **Trazabilidad:** clase `LogLimpieza` registra cada transformación con justificación.
- **Reproducibilidad:** `make pipeline && make app && make report` desde cero.

---

## Solución de problemas

En esta tabla se detallan las acciones a realizar en caso tal de que al intentar ejecutar o replicar el proyecto suceda algun error.

| Problema | Causa probable | Acción |
|----------|----------------|--------|
| `FileNotFoundError` al correr el pipeline o al abrir la app | Faltan los CSV de entrada en `data/raw/` (o se ejecutó el comando fuera de la raíz del proyecto) | Compruebe que existan estos tres archivos: `inventario_central_v2.csv`, `transacciones_logistica_v2.csv`, `feedback_clientes_v2.csv`. Luego ejecute desde la raíz: `make pipeline` |
| Health Score vacío / sin datos de calidad en el dashboard | Aún no se generaron `reports/quality/health_score_*.csv` ni los limpios en `data/interim/` | Ejecutar `make pipeline` (o `python scripts/run_pipeline.py`) |
| El PDF no aparece en el sidebar (Descargas) | El informe aún no se ha generado en la raíz del proyecto | Ejecutar `make report` (o `python scripts/generate_report.py`) |
| Error en la pestaña Recomendaciones IA (Groq) | API Key ausente, inválida o sin cuota | Revisar la key en el sidebar o en `.streamlit/secrets.toml` y la cuota en [console.groq.com](https://console.groq.com/) |
| Streamlit tarda mucho en la primera carga | Comportamiento esperado: la app ejecuta los pipelines y cachea el resultado | Esperar la primera carga; las siguientes interacciones son rápidas |

---

## Autores

- **Santiago Betancur** 
- **Santiago Acevedo urrego** 
- **Jeronimo Acosta Acevedo** 


Maestría en Ciencia de Datos, Universidad EAFIT.
