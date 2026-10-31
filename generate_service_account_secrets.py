"""
GCP Service Account Key & Streamlit `.toml` Secrets Generator (`generate_service_account_secrets.py`).

Capabilities:
  1. Reads `.env` variables (`API_KEY`, `GEMINI_MODEL`, `GOOGLE_GENAI_USE_VERTEXAI`,
     `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`).
  2. Optionally creates a GCP Service Account & downloads `service-account.json` via `gcloud`
     (with `--create-gcp-key`), or reads an existing `service-account.json` key file.
  3. Generates `.streamlit/secrets.toml` and `secrets.toml` in valid TOML format ready to paste
     directly into the Streamlit Community Cloud "Secrets" settings box.
  4. All generated key and TOML files are git-ignored via `.gitignore` so credentials are never
     pushed to GitHub.

Usage:
  # Generate TOML from .env (and existing service-account.json if present):
  python generate_service_account_secrets.py

  # Provision a new GCP Service Account key via gcloud CLI and embed it into secrets.toml:
  python generate_service_account_secrets.py --create-gcp-key

  # Embed a specific existing service-account JSON key file into secrets.toml:
  python generate_service_account_secrets.py --key-file path/to/service-account.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DEFAULT_KEY_PATH = BASE_DIR / "service-account.json"
STREAMLIT_SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"
ROOT_SECRETS_PATH = BASE_DIR / "secrets.toml"


def _toml_escape(val: str) -> str:
    """Escape a string for a TOML double-quoted scalar value."""
    return (
        val.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "")
        .replace("\n", "\\n")
    )


def create_gcp_service_account_key(
    project_id: str,
    sa_name: str,
    key_output_path: Path,
) -> Path:
    """
    Create (if needed) a GCP Service Account with Vertex AI User (`roles/aiplatform.user`)
    permissions and export its JSON key to `key_output_path` using the `gcloud` CLI.
    """
    gcloud_bin = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not gcloud_bin:
        raise RuntimeError(
            "`gcloud` CLI was not found on PATH. Install Google Cloud SDK or pass an existing "
            "JSON key via `--key-file service-account.json`."
        )

    sa_email = f"{sa_name}@{project_id}.iam.gserviceaccount.com"
    print(f"[1/3] Ensuring service account `{sa_email}` exists in project `{project_id}`...")
    describe_res = subprocess.run(
        [
            gcloud_bin,
            "iam",
            "service-accounts",
            "describe",
            sa_email,
            f"--project={project_id}",
            "--quiet",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if describe_res.returncode != 0:
        subprocess.run(
            [
                gcloud_bin,
                "iam",
                "service-accounts",
                "create",
                sa_name,
                f"--project={project_id}",
                "--display-name=IOH Streamlit Gemini Vertex AI Agent",
                "--quiet",
            ],
            check=True,
        )
    else:
        print(f"      Service account `{sa_email}` already exists.")

    print(f"[2/3] Binding least-privilege role `roles/aiplatform.user` to `{sa_email}`...")
    subprocess.run(
        [
            gcloud_bin,
            "projects",
            "add-iam-policy-binding",
            project_id,
            f"--member=serviceAccount:{sa_email}",
            "--role=roles/aiplatform.user",
            "--condition=None",
            "--quiet",
        ],
        check=True,
    )

    print(f"[3/3] Creating JSON key file at `{key_output_path}`...")
    subprocess.run(
        [
            gcloud_bin,
            "iam",
            "service-accounts",
            "keys",
            "create",
            str(key_output_path),
            f"--iam-account={sa_email}",
            f"--project={project_id}",
            "--quiet",
        ],
        check=True,
    )
    return key_output_path


def build_secrets_toml_content(
    env_vars: Dict[str, str],
    sa_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Format environment variables and optional `[gcp_service_account]` into valid TOML."""
    lines = [
        "# ==============================================================================",
        "# Streamlit Secrets Configuration (.streamlit/secrets.toml)",
        "# Paste the contents of this file into the Streamlit Cloud 'Secrets' UI box.",
        "# DO NOT COMMIT THIS FILE TO GITHUB (Protected by .gitignore)",
        "# ==============================================================================",
        "",
        f'API_KEY = "{_toml_escape(env_vars["API_KEY"])}"',
        f'GEMINI_MODEL = "{_toml_escape(env_vars["GEMINI_MODEL"])}"',
        f'GOOGLE_GENAI_USE_VERTEXAI = "{_toml_escape(env_vars["GOOGLE_GENAI_USE_VERTEXAI"])}"',
        f'GOOGLE_CLOUD_PROJECT = "{_toml_escape(env_vars["GOOGLE_CLOUD_PROJECT"])}"',
        f'GOOGLE_CLOUD_LOCATION = "{_toml_escape(env_vars["GOOGLE_CLOUD_LOCATION"])}"',
    ]

    if env_vars.get("GOOGLE_API_KEY"):
        lines.append(f'GOOGLE_API_KEY = "{_toml_escape(env_vars["GOOGLE_API_KEY"])}"')
    if env_vars.get("GEMINI_API_KEY"):
        lines.append(f'GEMINI_API_KEY = "{_toml_escape(env_vars["GEMINI_API_KEY"])}"')

    if sa_info:
        lines.extend(
            [
                "",
                "[gcp_service_account]",
            ]
        )
        ordered_keys = [
            "type",
            "project_id",
            "private_key_id",
            "private_key",
            "client_email",
            "client_id",
            "auth_uri",
            "token_uri",
            "auth_provider_x509_cert_url",
            "client_x509_cert_url",
            "universe_domain",
        ]
        for k in ordered_keys:
            if k in sa_info and sa_info[k] is not None:
                lines.append(f'{k} = "{_toml_escape(str(sa_info[k]))}"')
        for k, v in sa_info.items():
            if k not in ordered_keys and v is not None:
                lines.append(f'{k} = "{_toml_escape(str(v))}"')
    else:
        lines.extend(
            [
                "",
                "# Optional: Run `python generate_service_account_secrets.py --create-gcp-key`",
                "# or `python generate_service_account_secrets.py --key-file service-account.json`",
                "# to automatically populate the [gcp_service_account] block below.",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GCP Service Account JSON key and Streamlit secrets.toml"
    )
    parser.add_argument(
        "--create-gcp-key",
        action="store_true",
        help="Create a GCP Service Account & download service-account.json using gcloud CLI",
    )
    parser.add_argument(
        "--sa-name",
        type=str,
        default="ioh-streamlit-agent",
        help="Service account name when using --create-gcp-key (default: ioh-streamlit-agent)",
    )
    parser.add_argument(
        "--key-file",
        type=str,
        default=str(DEFAULT_KEY_PATH),
        help="Path to service-account.json key file (default: ./service-account.json)",
    )
    args = parser.parse_args()

    import secrets as py_secrets

    load_dotenv(dotenv_path=ENV_PATH, override=True)
    api_key_val = os.getenv("API_KEY", "").strip()
    if not api_key_val:
        api_key_val = f"ioh_ret_{py_secrets.token_hex(16)}"

    env_vars = {
        "API_KEY": api_key_val,
        "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        "GOOGLE_GENAI_USE_VERTEXAI": os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true"),
        "GOOGLE_CLOUD_PROJECT": os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id"),
        "GOOGLE_CLOUD_LOCATION": os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
    }

    key_path = Path(args.key_file).resolve()
    if args.create_gcp_key:
        create_gcp_service_account_key(
            project_id=env_vars["GOOGLE_CLOUD_PROJECT"],
            sa_name=args.sa_name,
            key_output_path=key_path,
        )

    sa_info: Optional[Dict[str, Any]] = None
    if key_path.exists():
        with open(key_path, "r", encoding="utf-8") as f:
            sa_info = json.load(f)
        print(f"Loaded GCP Service Account key from: {key_path}")

    toml_content = build_secrets_toml_content(env_vars, sa_info=sa_info)

    STREAMLIT_SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STREAMLIT_SECRETS_PATH.write_text(toml_content, encoding="utf-8")
    ROOT_SECRETS_PATH.write_text(toml_content, encoding="utf-8")

    print(f"Generated Streamlit TOML secrets at:")
    print(f"  - {STREAMLIT_SECRETS_PATH}")
    print(f"  - {ROOT_SECRETS_PATH}")
    print(f"\nAuthorized Streamlit App URL:")
    print(f"  https://example-url.streamlit.app?apikey={env_vars['API_KEY']}")


if __name__ == "__main__":
    main()
