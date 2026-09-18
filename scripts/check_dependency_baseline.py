"""Check locked exports without requiring Git metadata or changing tracked files."""
from pathlib import Path
import difflib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = ("uv.lock", "constraints/light-dev.txt", "constraints/full-dev.txt")


def main():
    try:
        original = {name: (ROOT / name).read_bytes() for name in FILES}
    except FileNotFoundError as exc:
        print(f"Missing dependency baseline file: {exc.filename}", file=sys.stderr)
        return 1

    try:
        for profile in ("light", "full"):
            subprocess.run(
                ["uv", "export", "--locked", "--extra", profile, "--extra", "dev",
                 "--no-emit-project", "--no-hashes", "-o", f"constraints/{profile}-dev.txt"],
                cwd=ROOT, stdout=subprocess.DEVNULL, check=True,
            )
        changed = []
        for name, before in original.items():
            after = (ROOT / name).read_bytes()
            # Git checkout may convert LF to CRLF on Windows.
            before = before.decode("utf-8").replace("\r\n", "\n")
            after = after.decode("utf-8").replace("\r\n", "\n")
            if before != after:
                changed.append(name)
                sys.stderr.writelines(difflib.unified_diff(
                    before.splitlines(keepends=True), after.splitlines(keepends=True),
                    fromfile=name, tofile=name + " (exported)",
                ))
        if changed:
            print("Dependency baseline is stale. Regenerate and commit both exports.",
                  file=sys.stderr)
            return 1
        print("Dependency baseline exports match uv.lock.")
        return 0
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"Dependency baseline validation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for name, content in original.items():
            (ROOT / name).write_bytes(content)


if __name__ == "__main__":
    raise SystemExit(main())
