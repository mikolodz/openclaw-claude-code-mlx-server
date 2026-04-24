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


def _assert_venv_is_312(venv_python: str) -> None:
    """
    Refuse to proceed if the venv's python is not 3.12, or if a parallel
    python3.14 / lib/python3.14 contamination is present.  A prior incident
    (2026-04-24) showed that running a Python-3.14-based `venv` or `pip`
    against an existing 3.12 venv silently overlays 3.14 entry-points and
    site-packages; subsequent `pip install` then downloaded cp314 wheels
    into a parallel tree while the server kept loading from the 3.12 tree,
    making debugging hell.  Detect and stop instead of muddling through.
    """
    if not Path(venv_python).exists():
        raise RuntimeError(f"Venv python missing: {venv_python}")
    result = subprocess.run(
        [venv_python, "-c", "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
        capture_output=True, text=True, check=True,
    )
    actual = result.stdout.strip()
    if actual != "3.12":
        raise RuntimeError(
            f"Venv python reports version {actual!r}, expected 3.12. "
            f"Something overwrote the venv — delete {VENV_DIR} and re-run."
        )
    # Check for contamination symptoms.
    contaminants = []
    for name in ("python3.13", "python3.14", "python3.15", "pip3.13", "pip3.14", "pip3.15"):
        p = VENV_DIR / "bin" / name
        if p.exists() or p.is_symlink():
            contaminants.append(str(p))
    for libdir in VENV_DIR.glob("lib/python3.*"):
        if libdir.name != "python3.12":
            contaminants.append(str(libdir))
    if contaminants:
        raise RuntimeError(
            "Venv contamination detected (non-3.12 artefacts present):\n  - "
            + "\n  - ".join(contaminants)
            + f"\n\nRemove them or delete {VENV_DIR} entirely and re-run this script."
        )
    # Verify pip's shebang points at the 3.12 venv python.  If someone ran
    # `python3.14 -m pip install --upgrade pip`, this symlink gets rewritten
    # with a 3.14 shebang and subsequent `pip install` lands in the wrong tree.
    pip_path = VENV_DIR / "bin" / "pip"
    if pip_path.exists() and not pip_path.is_symlink():
        try:
            first_line = pip_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        except Exception:
            first_line = ""
        if "python3.12" not in first_line and first_line.startswith("#!"):
            raise RuntimeError(
                f"{pip_path} has shebang {first_line!r} — expected python3.12. "
                "Delete the venv and re-run, or recreate the pip entry-point."
            )


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
    _assert_venv_is_312(venv_python)

    _print_step("Upgrading pip")
    _run([venv_python, "-m", "pip", "install", "--upgrade", "pip"])

    _print_step("Installing dependencies")
    # Always go through `python -m pip` — never the `pip` console script —
    # so a swapped pip shebang cannot silently divert installs to a wrong
    # Python.  Pairs with `_assert_venv_is_312` above.
    _run([venv_python, "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    _print_step("Starting MLX server")
    os.execv(venv_python, [venv_python, str(SERVER_SCRIPT)])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
