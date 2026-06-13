import os
from pathlib import Path


def load_env_file(env_path=".env"):
    path = Path(env_path)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Add it to .env or set it in your terminal.")
    return value
