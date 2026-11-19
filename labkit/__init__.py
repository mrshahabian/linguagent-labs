"""labkit — open, check and sync a LinguAgent lab.

The same few calls work in Google Colab, in a Codespace and on your own
machine. Standard library only: nothing to install.

In a notebook (each lab's first cell does this for you):

    import labkit
    lab = labkit.open("ai-dl/attention")
    ...write a function in a cell...
    lab.check("scaled_dot_product_attention", globals())   # just that function's tests
    lab.check_all(globals())                               # every test in the unit
    lab.sync()                                             # record it in LinguAgent

From a terminal, working in a unit's exercises.py:

    python -m labkit                    list the units and where you are
    python -m labkit check 26           run unit 26's tests (or: check ai-dl/attention)
    python -m labkit check 26 causal_mask
    python -m labkit sync 26            record your last result in LinguAgent
    python -m labkit verify             prove every reference solution passes
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import time
import traceback
import types
import unittest
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__version__ = "1.0.0"

ROOT = Path(__file__).resolve().parent.parent
UNITS = ROOT / "units"
API = os.environ.get("LINGUAGENT_API", "https://api.linguagent.com").rstrip("/")
APP = os.environ.get("LINGUAGENT_APP", "https://linguagent.com").rstrip("/")
_STATE = ROOT / ".labkit" / "results.json"


# ---------------------------------------------------------------------------
# Finding units
# ---------------------------------------------------------------------------

def units() -> list[dict[str, Any]]:
    """Every unit's metadata, in course order."""
    out = []
    for meta in sorted(UNITS.glob("*/*/unit.json")):
        data = json.loads(meta.read_text(encoding="utf-8"))
        data["_dir"] = str(meta.parent)
        out.append(data)
    return sorted(out, key=lambda u: u.get("order", 0))


def find(unit: str) -> Path:
    """A unit's folder, by its id ("ai-dl/attention"), folder ("26-attention",
    "ai-dl/26-attention") or number ("26")."""
    key = str(unit).strip().strip("/")
    for u in units():
        folder = Path(u["_dir"])
        names = {u["id"], folder.name, f"{folder.parent.name}/{folder.name}", folder.name.split("-", 1)[0]}
        if key in names:
            return folder
    raise LookupError(f"No unit called {unit!r}. Run `python -m labkit` to list them.")


def where() -> str:
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("CODESPACES"):
        return "codespaces"
    return "local"


# ---------------------------------------------------------------------------
# Running tests
# ---------------------------------------------------------------------------

@dataclass
class Result:
    unit: str
    target: str                       # "" for the whole unit
    passed: int = 0
    total: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)   # (test, first line of why)
    # Tests of a section that also use something from a LATER section, not
    # written yet: (test, "causal_mask (§2)"). Not failures — not yet.
    waiting: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.passed == self.total

    @property
    def ready_ok(self) -> bool:
        """Everything this section can check yet passes."""
        return self.total > 0 and not self.failures and bool(self.passed or self.waiting)

    def __repr__(self) -> str:        # what a notebook shows under the cell
        return f"<{self.passed}/{self.total} tests pass>"


class _Collect(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.passed: list[str] = []

    def addSuccess(self, test: unittest.TestCase) -> None:  # noqa: N802 — unittest's name
        super().addSuccess(test)
        self.passed.append(test.id())


def _why(err: str) -> str:
    """The line of a traceback worth reading: the error itself."""
    lines = [ln for ln in err.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        if not ln.startswith(" ") and ":" in ln:
            return ln.strip()[:300]
    return (lines[-1] if lines else "failed").strip()[:300]


def _fresh(name: str, path: Path) -> types.ModuleType:
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _from_namespace(ns: dict[str, Any]) -> types.ModuleType:
    """The functions and classes written in a notebook, as the module the tests
    import (`exercises`)."""
    mod = types.ModuleType("exercises")
    mod.__dict__.update({k: v for k, v in ns.items() if not (k.startswith("__") and k.endswith("__"))})
    return mod


def run_tests(unit_dir: Path, *, target: str = "", ns: Optional[dict] = None, use: str = "exercises") -> Result:
    """Run a unit's tests (or one exercise's) against the notebook's code (`ns`),
    the unit's exercises.py, or its solutions.py (`use="solutions"`)."""
    meta = json.loads((unit_dir / "unit.json").read_text(encoding="utf-8"))
    result = Result(unit=meta["id"], target=target)
    here = str(unit_dir)
    if here not in sys.path:
        sys.path.insert(0, here)
    old_cwd = os.getcwd()
    os.chdir(unit_dir)
    try:
        if ns is not None:
            sys.modules["exercises"] = _from_namespace(ns)
        else:
            _fresh("exercises", unit_dir / f"{use}.py")
        os.environ["PY_TARGET"] = "exercises"
        tests = _fresh("test_exercises", unit_dir / "test_exercises.py")
        loader = unittest.TestLoader()
        names = []
        if isinstance(target, (list, tuple)):
            wanted = [e for e in meta.get("exercises", []) if e["name"] in target]
            names = sorted({t for e in wanted for t in e["test_targets"]})
            result.target = ", ".join(target)
            target = ""
        elif target.startswith("§"):
            # A section of the lesson: the tests its "Test it" line names.
            names = meta.get("checks", {}).get(target[1:], [])
            if not names:
                raise LookupError(f"{meta['id']} has no checks for section {target}.")
            done = meta.get("sections", {}).get(target[1:], [])
            result.target = f"§{target[1:]}" + (f" ({', '.join(done)})" if done else "")
        elif target:
            ex = next((e for e in meta.get("exercises", []) if e["name"] == target), None)
            if ex is None:
                raise LookupError(f"{meta['id']} has no exercise called {target!r}. "
                                  f"Try one of: {', '.join(e['name'] for e in meta.get('exercises', []))}")
            names = ex["test_targets"]
        suite = _select(loader.loadTestsFromModule(tests), names) if names else loader.loadTestsFromModule(tests)
        collected = _Collect()
        suite.run(collected)
        result.total = collected.testsRun
        result.passed = len(collected.passed)
        later = _later_names(meta, target)
        for test, err in collected.failures + collected.errors:
            name = test.id().split(".", 1)[-1]
            needs = _needs_later(err, later)
            if needs:
                result.waiting.append((name, needs))
            else:
                result.failures.append((name, _why(err)))
    except Exception:                                         # the code didn't even import
        result.failures.append(("(loading the code)", _why(traceback.format_exc())))
    finally:
        os.chdir(old_cwd)
    return result


def _later_names(meta: dict, target: str) -> dict[str, str]:
    """For a section check: exercises written in LATER sections (or after the
    lesson), name -> where. Empty for anything else."""
    if not target.startswith("§"):
        return {}
    here = int(target[1:])
    out = {}
    for sec, names in meta.get("sections", {}).items():
        if sec.isdigit() and int(sec) > here:
            out.update({n: f"§{sec}" for n in names})
    placed = {n for names in meta.get("sections", {}).values() for n in names}
    out.update({e["name"]: "the exercises at the end" for e in meta.get("exercises", []) if e["name"] not in placed})
    return out


def _needs_later(err: str, later: dict[str, str]) -> str:
    """If a test failed only because it uses an exercise from a later section
    that isn't written yet, that exercise ("causal_mask (§2)"); else ""."""
    if not later:
        return ""
    import re
    m = re.search(r"has no attribute '(\w+)'", err) or re.search(r"name '(\w+)' is not defined", err)
    if m and m.group(1) in later:
        return f"{m.group(1)} ({later[m.group(1)]})"
    if "NotImplementedError" in err:
        frames = re.findall(r'File "[^"]*", line \d+, in (\w+)', err)
        for fn in reversed(frames):
            if fn in later:
                return f"{fn} ({later[fn]})"
            if fn not in ("__init__", "forward", "__call__"):
                break
    return ""


def _flat(suite):
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            yield from _flat(t)
        else:
            yield t


def _select(suite: unittest.TestSuite, names: list[str]) -> unittest.TestSuite:
    """The tests a check names: whole classes ("TestCausalMask"), or unittest's
    `-k` patterns ("k:repr": a substring, or fnmatch with *)."""
    import fnmatch
    classes = {n for n in names if not n.startswith("k:")}
    patterns = [n[2:] for n in names if n.startswith("k:")]

    def keep(t) -> bool:
        tid = t.id()
        if type(t).__name__ in classes:
            return True
        return any(fnmatch.fnmatchcase(tid, f"*{p}*" if "*" not in p else p) for p in patterns)
    return unittest.TestSuite(t for t in _flat(suite) if keep(t))


def report(result: Result, out=None) -> None:
    out = out or sys.stdout
    what = result.target or result.unit
    if result.ok:
        print(f"✅ {what}: all {result.total} tests pass.", file=out)
        return
    if result.ready_ok:
        names = sorted({n for _, n in result.waiting})
        print(f"✅ {what}: all {result.passed} tests that need only this part pass.", file=out)
        print(f"   {len(result.waiting)} more also use {', '.join(names)} — they'll pass once that's written.",
              file=out)
        return
    print(f"{'❌' if result.passed == 0 else '🟡'} {what}: {result.passed}/{result.total} tests pass.", file=out)
    for name, why in result.failures[:8]:
        print(f"   ✗ {name}\n       {why}", file=out)
    if len(result.failures) > 8:
        print(f"   … and {len(result.failures) - 8} more.", file=out)


# ---------------------------------------------------------------------------
# Results, kept so `sync` can send the latest
# ---------------------------------------------------------------------------

def _save(result: Result) -> None:
    try:
        _STATE.parent.mkdir(exist_ok=True)
        data = json.loads(_STATE.read_text()) if _STATE.exists() else {}
        data[result.unit] = {"passed": result.passed, "total": result.total, "at": time.time(),
                             "failing": [n for n, _ in result.failures][:30]}
        _STATE.write_text(json.dumps(data, indent=1))
    except OSError:
        pass


def _last(unit_id: str) -> Optional[dict]:
    try:
        return json.loads(_STATE.read_text()).get(unit_id)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Syncing to LinguAgent
# ---------------------------------------------------------------------------

def _token(token: Optional[str]) -> str:
    """Your lab token: given, or LINGUAGENT_TOKEN in the environment, or the
    Colab secret of that name, or asked for (never stored in the notebook)."""
    if token:
        return token.strip()
    if os.environ.get("LINGUAGENT_TOKEN"):
        return os.environ["LINGUAGENT_TOKEN"].strip()
    try:
        from google.colab import userdata  # type: ignore
        value = userdata.get("LINGUAGENT_TOKEN")
        if value:
            return value.strip()
    except Exception:  # noqa: BLE001 — not in Colab, or the secret isn't shared with this notebook
        pass
    import getpass
    return getpass.getpass(f"Your LinguAgent lab token ({APP}/settings#labs): ").strip()


def send(unit_id: str, passed: int, total: int, failing: list[str], token: Optional[str] = None) -> str:
    body = json.dumps({"block_id": unit_id, "passed": passed, "total": total, "failing": failing[:30],
                       "where": where(), "labkit": __version__}).encode()
    req = urllib.request.Request(f"{API}/api/labs/sync", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "X-Lab-Token": _token(token),
                                          "User-Agent": f"labkit/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read() or b"{}").get("message") or "Recorded in LinguAgent."
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read() or b"{}").get("detail")
        except ValueError:
            detail = ""
        if e.code in (401, 403):
            return f"LinguAgent didn't accept the token ({detail or e.code}). Make a new one at {APP}/settings#labs."
        return f"LinguAgent said no ({e.code}): {detail}"
    except urllib.error.URLError as e:
        return f"Couldn't reach LinguAgent ({e.reason}). Your result is kept; try `sync` again later."


# ---------------------------------------------------------------------------
# The notebook object
# ---------------------------------------------------------------------------

class Lab:
    def __init__(self, unit_dir: Path):
        self.dir = unit_dir
        self.meta = json.loads((unit_dir / "unit.json").read_text(encoding="utf-8"))
        self.id = self.meta["id"]
        self.last: Optional[Result] = None

    def __repr__(self) -> str:
        n = len(self.meta.get("exercises", []))
        return (f"<lab {self.id}: {self.meta.get('title', '')} — {n} exercises, "
                f"{self.meta.get('tests', 0)} tests>")

    def check(self, name: str = "", ns: Optional[dict] = None) -> Result:
        """Run the tests for one exercise (or all, with no name) against the
        code in this notebook. Pass globals()."""
        ns = ns if ns is not None else _caller_globals()
        result = run_tests(self.dir, target=name, ns=ns)
        report(result)
        if not name:
            self.last = result
            _save(result)
            if result.ok:
                print(f"   Run lab.sync() to record it in LinguAgent.")
        return result

    def check_all(self, ns: Optional[dict] = None) -> Result:
        return self.check("", ns if ns is not None else _caller_globals())

    def solution(self, name: str) -> None:
        """Print the reference solution for one exercise. Try first: it's worth it."""
        ex = next((e for e in self.meta.get("exercises", []) if e["name"] == name), None)
        if not ex:
            raise LookupError(f"No exercise called {name!r}.")
        print(ex.get("solution") or "(no reference for this one)")

    def sync(self, token: Optional[str] = None) -> None:
        """Record this unit's latest full check in LinguAgent."""
        r = self.last
        last = {"passed": r.passed, "total": r.total, "failing": [n for n, _ in r.failures]} if r else _last(self.id)
        if not last:
            print("Run lab.check_all(globals()) first — sync sends your latest full result.")
            return
        print(send(self.id, last["passed"], last["total"], last.get("failing", []), token))


def _caller_globals() -> dict:
    frame = sys._getframe(2)
    return frame.f_globals


def open(unit: str) -> Lab:  # noqa: A001 — `labkit.open(...)` reads well in a notebook
    """The lab for a unit; moves into its folder so its files are found."""
    folder = find(unit)
    os.chdir(folder)
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    lab = Lab(folder)
    print(f"📘 {lab.meta.get('title', lab.id)} — {len(lab.meta.get('exercises', []))} exercises, "
          f"{lab.meta.get('tests', 0)} tests. Write each function in its cell, then run the check under it.")
    return lab
