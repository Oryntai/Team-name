"""Local-only JSON integration harness. Media extraction belongs to backend."""

import argparse
import json
import os
from pathlib import Path
import tempfile

from .contracts import Observation, Rule, strict_json_loads
from .engine import decide


def load_rules(paths):
    """Load one or more array catalogs (or a single-key {'rules': [...]} wrapper)."""
    rules = []
    for path in paths:
        catalog = strict_json_loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(catalog, dict):
            if set(catalog) != {"rules"}:
                raise ValueError("catalog wrapper must contain only rules")
            catalog = catalog["rules"]
        if not isinstance(catalog, list):
            raise ValueError("catalog must be an array")
        for entry in catalog:
            if not isinstance(entry, dict):
                raise ValueError("rule must be an object")
            rules.append(Rule(**entry))
    return rules


def write_result(path, content):
    """Replace an explicit output atomically using private filesystem permissions."""
    output = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=".forte-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--rules", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-anchor-delta-sec", type=float, default=1.0)
    parser.add_argument("--mode", choices=("review", "strict"), default="review")
    args = parser.parse_args(argv)
    try:
        inputs = [args.observation, *args.rules]
        if args.output.resolve() in {path.resolve() for path in inputs}:
            raise ValueError("output cannot replace an input")
        observation = Observation.from_json(
            args.observation.read_text(encoding="utf-8")
        )
        rules = load_rules(args.rules)
        result = decide(
            observation,
            rules,
            max_anchor_delta_sec=args.max_anchor_delta_sec,
            mode=args.mode,
        )
        write_result(args.output, result.to_json())
    except (ValueError, TypeError, KeyError, OSError, UnicodeError):
        # Exception messages can contain user observations. Never echo them.
        print(json.dumps({"status": "error", "event_count": 0}))
        return 1
    print(
        json.dumps(
            {
                "status": result.status,
                "event_count": len(result.events),
            }
        )
    )
    return 0 if result.status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
