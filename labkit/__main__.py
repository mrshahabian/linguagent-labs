"""`python -m labkit` — the labs from a terminal. See labkit/__init__.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import UNITS, find, report, run_tests, send, units, _last, _save


def _list() -> int:
    print("Units (python -m labkit check <number>):\n")
    for u in units():
        last = _last(u["id"])
        mark = ("✅" if last and last["passed"] == last["total"] else f"{last['passed']}/{last['total']}") if last else "  "
        print(f"  {mark:>6}  {Path(u['_dir']).name:<28} {u.get('title', '')}")
    return 0


def _check(unit: str, target: str = "") -> int:
    folder = find(unit)
    if not (folder / "test_exercises.py").exists():
        # A project unit: its own test layout, run as its README says.
        return subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=folder)
    result = run_tests(folder, target=target)
    report(result)
    if not target:
        _save(result)
    return 0 if result.ok else 1


def _sync(unit: str) -> int:
    meta = json.loads((find(unit) / "unit.json").read_text())
    last = _last(meta["id"])
    if not last:
        print(f"Run `python -m labkit check {unit}` first.")
        return 1
    print(send(meta["id"], last["passed"], last["total"], last.get("failing", [])))
    return 0


def _verify() -> int:
    """Every reference solution passes its tests; every stub doesn't (yet).
    Each unit in its own process, so their modules never meet."""
    bad = 0
    for u in units():
        folder = Path(u["_dir"])
        if not (folder / "test_exercises.py").exists():
            continue
        code = ("import sys, json; from pathlib import Path; from labkit import run_tests; "
                f"r = run_tests(Path({str(folder)!r}), use='solutions'); "
                "print(json.dumps([r.passed, r.total, r.failures[:3]]))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=str(UNITS.parent))
        try:
            passed, total, fails = json.loads(out.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            passed, total, fails = 0, 0, [["(crashed)", out.stderr[-300:]]]
        ok = total > 0 and passed == total
        bad += not ok
        print(f"{'✅' if ok else '❌'} {u['id']:<34} {passed}/{total}" + ("" if ok else f"  {fails}"))
    print("\nAll reference solutions pass." if not bad else f"\n{bad} unit(s) failing.")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    if not argv:
        return _list()
    cmd, rest = argv[0], argv[1:]
    if cmd == "check" and rest:
        return _check(rest[0], rest[1] if len(rest) > 1 else "")
    if cmd == "sync" and rest:
        return _sync(rest[0])
    if cmd == "verify":
        return _verify()
    print(__doc__ or "", (Path(__file__).parent / "__init__.py").read_text().split('"""')[1])
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
