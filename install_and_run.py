#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
SERVER_SCRIPT = ROOT / "start-llm.py"
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"


def _print_step(message: str) -> None:
    print(f"[setup] {message}")


def _run(cmd, cwd: Path = ROOT) -> None:
    print("$", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _resolve_python312() -> str:
    python312 = shutil.which("python3.12")
    if python312:
        return python312
    if sys.version_info[:2] == (3, 12):
        return sys.executable
    raise RuntimeError(
        "Python 3.12 is required. Install python3.12 and re-run this script."
    )


def _ensure_env_file() -> None:
    if ENV_FILE.exists():
        return
    if ENV_EXAMPLE.exists():
        ENV_FILE.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
        _print_step("Created .env from .env.example")
        return
    _print_step("No .env found and no .env.example available. Continuing without copy.")


def _ensure_venv(python312: str) -> None:
    if VENV_DIR.exists():
        _print_step(f"Using existing virtualenv at {VENV_DIR}")
        return
    _print_step("Creating virtualenv with Python 3.12")
    _run([python312, "-m", "venv", str(VENV_DIR)])


def _venv_bin(name: str) -> str:
    return str(VENV_DIR / "bin" / name)


def main() -> int:
    if not REQUIREMENTS.exists():
        raise RuntimeError(f"Missing requirements file: {REQUIREMENTS}")
    if not SERVER_SCRIPT.exists():
        raise RuntimeError(f"Missing server script: {SERVER_SCRIPT}")

    python312 = _resolve_python312()
    _print_step(f"Using Python 3.12 at {python312}")

    _ensure_env_file()
    _ensure_venv(python312)

    venv_python = _venv_bin("python")
    venv_pip = _venv_bin("pip")

    _print_step("Upgrading pip")
    _run([venv_python, "-m", "pip", "install", "--upgrade", "pip"])

    _print_step("Installing dependencies")
    _run([venv_pip, "install", "-r", str(REQUIREMENTS)])

    _print_step("Starting MLX server")
    os.execv(venv_python, [venv_python, str(SERVER_SCRIPT)])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
