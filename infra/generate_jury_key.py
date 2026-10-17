"""Generate and store a jury API key in the ignored repository-root .env file."""

from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
KEY_NAME = "FORTE_API_KEY"
PLACEHOLDERS = ("replace-with-", "your-", "change-me")


def _read_key(line: str) -> str:
    value = line.split("=", 1)[1].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def _write_env(lines: list[str]) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=ROOT, prefix=".env-key-", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write("\n".join(lines).rstrip() + "\n")
        if os.name != "nt":
            temporary_path.chmod(0o600)
        os.replace(temporary_path, ENV_FILE)
        if os.name != "nt":
            ENV_FILE.chmod(0o600)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--rotate", action="store_true", help="replace the currently configured key")
    action.add_argument("--show", action="store_true", help="print the current key from .env")
    args = parser.parse_args()

    if not ENV_FILE.exists():
        if not ENV_EXAMPLE.is_file():
            parser.error(".env and .env.example are both missing")
        ENV_FILE.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")

    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    key_indexes = [
        i for i, line in enumerate(lines)
        if "=" in line and line.partition("=")[0].strip() == KEY_NAME
    ]
    if len(key_indexes) > 1:
        parser.error(f".env contains more than one {KEY_NAME} entry; remove the duplicate first")

    existing_key = _read_key(lines[key_indexes[0]]) if key_indexes else ""
    placeholder = existing_key.casefold().startswith(PLACEHOLDERS)
    if args.show:
        if not existing_key or placeholder:
            parser.error(".env has no generated jury key; run this command without --show")
        print("Current jury key (share only with approved judges over a private channel):")
        print(existing_key)
        return 0

    if existing_key and not placeholder and not args.rotate:
        print("A jury key is already configured; it was not printed or changed.")
        print("Use --show to display it locally or --rotate to replace it.")
        return 0

    jury_key = secrets.token_urlsafe(32)
    new_line = f"{KEY_NAME}={jury_key}"
    if key_indexes:
        lines[key_indexes[0]] = new_line
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(new_line)
    _write_env(lines)

    print("Jury key generated and saved to the ignored .env file.")
    print("Copy it now and share only with approved judges over a private channel; never commit or log it.")
    print(jury_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
