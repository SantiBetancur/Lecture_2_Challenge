.PHONY: install pipeline app report clean

install:
	pip install -e .

pipeline:
	python scripts/run_pipeline.py

app:
	streamlit run app.py

report:
	python scripts/generate_report.py
	@echo Informe PDF en la raiz: Informe_Consultoria_TechLogistics_Junta_Directiva_Hallazgos_Estrategicos.pdf

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
