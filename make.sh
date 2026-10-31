#!/usr/bin/env bash
# ==============================================================================
# Git Bash / Linux / macOS Wrapper for Makefile Targets (make.sh)
# Usage in Git Bash:
#   ./make.sh secrets
#   ./make.sh run
#   ./make.sh pipeline
#   ./make.sh test
#   ./make.sh zip
# ==============================================================================
set -e

PYTHON="D:/Projects/.venv/Scripts/python.exe"
STREAMLIT="D:/Projects/.venv/Scripts/streamlit.exe"
PORT=8501

TARGET="${1:-help}"

case "$TARGET" in
  install)
    uv pip install --python "$PYTHON" -r requirements.txt
    ;;
  secrets)
    "$PYTHON" generate_service_account_secrets.py --create-gcp-key
    ;;
  pipeline)
    "$PYTHON" main.py
    ;;
  run|app|streamlit)
    API_KEY_VAL=$(grep '^API_KEY=' .env | cut -d'=' -f2 | tr -d '\r')
    echo "=============================================================================="
    echo "Open Authenticated URL: http://localhost:${PORT}/?apikey=${API_KEY_VAL}"
    echo "=============================================================================="
    "$STREAMLIT" run streamlit_app.py --server.port="$PORT"
    ;;
  test)
    "$PYTHON" -c "import src.retention_agent as ra; llm = ra.create_gemini_chat_model(); print('LLM Ready:', llm.model)"
    "$PYTHON" main.py
    ;;
  zip)
    "$PYTHON" -c "import zipfile, pathlib; excl = {'IOH_Data_Scientist_Submission.zip', '__pycache__', '.env', 'secrets.toml', 'service-account.json', 'service_account.json'}; zf = zipfile.ZipFile('IOH_Data_Scientist_Submission.zip', 'w', zipfile.ZIP_DEFLATED); [zf.write(p, p.as_posix()) for p in pathlib.Path('.').rglob('*') if p.is_file() and not any(part in excl or (part.startswith('.') and part != '.env.example') or 'secrets.toml' in part or 'service-account' in part for part in p.parts)]; zf.close(); print('Built IOH_Data_Scientist_Submission.zip')"
    ;;
  *)
    echo "=============================================================================="
    echo "IOH Customer Retention Intelligence — Available Targets (Git Bash ./make.sh)"
    echo "=============================================================================="
    echo "  ./make.sh run          Launch the Streamlit App (requires ?apikey=API_KEY in URL)"
    echo "  ./make.sh app          Alias for 'run'"
    echo "  ./make.sh streamlit    Alias for 'run'"
    echo "  ./make.sh secrets      Generate GCP service-account.json & .streamlit/secrets.toml"
    echo "  ./make.sh pipeline     Run the full 10-stage analytical & LangGraph pipeline (main.py)"
    echo "  ./make.sh install      Install all dependencies from requirements.txt via uv"
    echo "  ./make.sh test         Run end-to-end verification tests"
    echo "  ./make.sh zip          Build and verify IOH_Data_Scientist_Submission.zip"
    echo "=============================================================================="
    ;;
esac
