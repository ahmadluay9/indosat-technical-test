# ==============================================================================
# IOH Customer Churn, Survival Hazard & Agentic Retention Intelligence
# Makefile — Pipeline, Streamlit App (ChatGoogleGenerativeAI gemini-3.8-flash),
#            Notebooks, Executive Deck & Packaging
# ==============================================================================

PYTHON ?= D:/Projects/.venv/Scripts/python.exe
STREAMLIT ?= D:/Projects/.venv/Scripts/streamlit.exe
PORT ?= 8501

.PHONY: help install run app streamlit pipeline secrets notebook presentation test zip clean

help:
	@echo "=============================================================================="
	@echo "IOH Customer Retention Intelligence — Available Make Targets"
	@echo "=============================================================================="
	@echo "  make run          Launch the Streamlit App (requires ?apikey=API_KEY in URL)"
	@echo "  make app          Alias for 'make run'"
	@echo "  make streamlit    Alias for 'make run'"
	@echo "  make secrets      Generate GCP service-account.json & .streamlit/secrets.toml"
	@echo "  make pipeline     Run the full 10-stage analytical & LangGraph pipeline (main.py)"
	@echo "  make install      Install all dependencies from requirements.txt via uv/pip"
	@echo "  make test         Run end-to-end verification & Streamlit URL auth tests"
	@echo "  make zip          Build and verify IOH_Data_Scientist_Submission.zip"
	@echo "=============================================================================="

install:
	uv pip install --python "$(PYTHON)" -r requirements.txt

secrets:
	"$(PYTHON)" generate_service_account_secrets.py --create-gcp-key

pipeline:
	"$(PYTHON)" main.py

run:
	@echo "Launching Streamlit App on http://localhost:$(PORT)/?apikey=$$(grep '^API_KEY=' .env | cut -d'=' -f2)"
	"$(STREAMLIT)" run streamlit_app.py --server.port=$(PORT)

app: run

streamlit: run

test:
	"$(PYTHON)" -c "import src.retention_agent as ra; llm = ra.create_gemini_chat_model(); print('LLM Ready:', llm.model)"
	"$(PYTHON)" main.py

zip:
	"$(PYTHON)" -c "import zipfile, pathlib; excl = {'IOH_Data_Scientist_Submission.zip', '__pycache__', '.env', 'secrets.toml', 'service-account.json', 'service_account.json'}; zf = zipfile.ZipFile('IOH_Data_Scientist_Submission.zip', 'w', zipfile.ZIP_DEFLATED); [zf.write(p, p.as_posix()) for p in pathlib.Path('.').rglob('*') if p.is_file() and not any(part in excl or (part.startswith('.') and part != '.env.example') or 'secrets.toml' in part or 'service-account' in part for part in p.parts)]; zf.close(); print('Built IOH_Data_Scientist_Submission.zip')"
