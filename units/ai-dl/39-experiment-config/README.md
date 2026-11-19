# 39 — Experiment configuration management

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/39-experiment-config/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/experiment-config?utm_source=github&utm_medium=labs)

*~50 minutes · 18 tests*

Every real training or inference pipeline eventually needs to swap a whole
component — a different pose extractor, a different model, a different
dataset — from the command line, without touching code. Hydra and
OmegaConf are the library most ML repos reach for; this module reimplements
their core mechanics from scratch, in plain dicts, no dependency required:
`_target_` + `instantiate` for building objects from config, config-group
composition, `${a.b.c}` interpolation, and CLI-style overrides.

Work in `exercises.py`. Check your work with `python -m labkit check 39`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/39-experiment-config/`.

## What you'll build

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `instantiate` | the function | `hydra.utils.instantiate` — building an object from a `_target_` path |
| `resolve_interpolations` | `_lookup`, `resolve_interpolations` | OmegaConf's `${a.b.c}` string interpolation |
| `merge_config_groups` | the function | Hydra's `defaults:` list — composing config groups |
| `apply_overrides` | `_coerce`, `apply_overrides` | Hydra's CLI override syntax — `pose=mediapipe`, `pose.confidence=0.9` |

---

## The running example

Everything below builds toward the same scenario: a pipeline with a
swappable pose extractor, defined as two interchangeable YAML-shaped
configs:

```python
config_groups = {
    "pose": {
        "yolo":      {"_target_": "mypackage.PoseExtractor", "name": "yolo",      "confidence": 0.5, "device": "cpu"},
        "mediapipe": {"_target_": "mypackage.PoseExtractor", "name": "mediapipe", "confidence": 0.8, "device": "cpu"},
    },
}
defaults = {"pose": "yolo"}
```

## 1. `_target_` + `instantiate`: config describes, code never imports

```python
def instantiate(config):
    config = dict(config)
    target = config.pop("_target_")
    module_path, _, attr_name = target.rpartition(".")
    module = importlib.import_module(module_path)
    target_callable = getattr(module, attr_name)
    return target_callable(**config)
```

`"_target_"` is a dotted string — everything before the last dot is a
**module** to import, everything after is the **class or function** to
pull out of it. Every other key in the config becomes a keyword argument
to that callable. The config file never contains an `import` statement;
it only *names* what to build. That is what makes swapping an entire
implementation — YOLO-based pose extraction for MediaPipe-based — a
one-line config change instead of an edit to the code that constructs it.

```python
>>> instantiate({"_target_": "datetime.date", "year": 2024, "month": 1, "day": 1})
datetime.date(2024, 1, 1)
```

**In `exercises.py`:** `instantiate`.

**Test it:** `python3 -m unittest test_exercises.TestInstantiate -v`

## 2. `${a.b.c}`: interpolation against the final merged config

```python
run_name: demo_${pose.name}
```

A string containing `${a.b.c}` is resolved by looking up `a.b.c` in the
**config it lives inside** and substituting the result — so
`run_name` can reference `pose.name` as a sibling field, without knowing
in advance whether `pose` resolved to `yolo` or `mediapipe`. This only
makes sense to resolve **after** every default and override has already
been applied: interpolation reads the final tree, not whatever was in
scope when the string was originally written in some far-away YAML file.

```python
>>> resolve_interpolations({"pose": {"name": "yolo"}, "run_name": "demo_${pose.name}"})
{'pose': {'name': 'yolo'}, 'run_name': 'demo_yolo'}
```

**In `exercises.py`:** `_lookup`, `resolve_interpolations`.

**Test it:** `python3 -m unittest test_exercises.TestResolveInterpolations -v`

## 3. Config groups: one slot, several interchangeable options

```python
def merge_config_groups(defaults, config_groups):
    return {
        group: copy.deepcopy(config_groups[group][defaults[group]])
        for group in defaults
    }
```

`config_groups["pose"]` holds every *option* for the pose slot; `defaults`
picks exactly one per group. This is Hydra's `defaults:` list — the
declaration in `config.yaml` that says "load `pose/yolo.yaml` unless told
otherwise" for each group it names. The **deep copy** matters: the merged
result gets mutated next (by overrides), and a shallow copy would let
that mutation corrupt the shared `config_groups` dict every other
composed config still reads from.

```python
>>> merge_config_groups({"pose": "yolo"}, config_groups)
{'pose': {'_target_': 'mypackage.PoseExtractor', 'name': 'yolo', 'confidence': 0.5, 'device': 'cpu'}}
```

**In `exercises.py`:** `merge_config_groups`.

**Test it:** `python3 -m unittest test_exercises.TestMergeConfigGroups -v`

## 4. CLI overrides: swap a whole group, or one key inside it

```python
key, _, raw_value = override.partition("=")
if "." not in key and key in config_groups:
    config[key] = copy.deepcopy(config_groups[key][raw_value])   # whole-group swap
else:
    ...                                                          # walk to the leaf and set it
```

Two override shapes, told apart by whether the left side has a dot:
`pose=mediapipe` replaces the **entire** `pose` group with a different
named option; `pose.confidence=0.9` walks into the nested dict and
changes **only that leaf**, leaving every other key in `pose` untouched.
Values are coerced — `"0.9"` becomes a `float`, `"50"` an `int`, anything
that doesn't parse as a number (like `"cuda:0"`) stays a string.

**Overrides apply left to right, mutating the same config in sequence** —
`pose=mediapipe pose.device=cuda:0` swaps the whole group first, *then*
overrides a key on top of the freshly-swapped-in config. That ordering
is what makes combining a group swap with a key override inside that
same group actually work, rather than the two being computed
independently and merged at the end.

```python
>>> merged = merge_config_groups({"pose": "yolo"}, config_groups)
>>> apply_overrides(merged, config_groups, ["pose=mediapipe", "pose.device=cuda:0"])
{'pose': {'_target_': 'mypackage.PoseExtractor', 'name': 'mediapipe', 'confidence': 0.8, 'device': 'cuda:0'}}
```

This is the exact shape behind a real training script's CLI:
`python train.py model=vit dataset=rhm_har trainer.max_epochs=50` — two
whole-group swaps and one dotted key override, applied in the order
given, on top of whatever `defaults:` already assembled.

**In `exercises.py`:** `_coerce`, `apply_overrides`.

**Test it:** `python3 -m unittest test_exercises.TestApplyOverrides -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `instantiate` | Building an object from a `_target_` path — config describes, never imports |
| `resolve_interpolations` | `${a.b.c}` resolves against the final tree, not write-time scope |
| `merge_config_groups` | `defaults:` composition; deep-copying to protect shared config |
| `apply_overrides` | Group swap vs. dotted key override; sequential application; value coercion |

Every function in this table has a stub in `exercises.py` and test
coverage in `test_exercises.py` — a clean 1:1 mapping.

## Check your understanding

1. **Why does `_target_` name a dotted import path instead of the config
   just embedding the class directly somehow?**
   <details><summary>Answer</summary>A config file is data, not code — it
   can't "embed a class." Naming the class by its import path lets a
   plain dict fully describe *what to build* without the config system
   itself needing to know anything about the classes involved; swapping
   implementations becomes editing a string, not writing an
   import.</details>

2. **Why must interpolation (`${a.b.c}`) be resolved after every default
   and override has already been applied, not before?**
   <details><summary>Answer</summary>An interpolation reads from the
   FINAL merged config — if `pose` hasn't been resolved yet (or gets
   swapped by a later override), `${pose.name}` would either fail to
   resolve or resolve against a value that's about to be replaced.
   Resolving last guarantees it sees the config exactly as the run will
   actually use it.</details>

3. **Why does `merge_config_groups` deep-copy the selected option instead
   of just referencing it directly?**
   <details><summary>Answer</summary>The merged result gets mutated next,
   by `apply_overrides`. A shallow reference would mean overriding
   `pose.confidence` for one run also silently changed the `confidence`
   value inside the shared `config_groups["pose"]["yolo"]` dict, which
   every other composed config might still be reading from.</details>

4. **In `pose=mediapipe pose.device=cuda:0`, why does the group swap have
   to happen before the key override, not after or independently?**
   <details><summary>Answer</summary>The key override targets whatever is
   currently in <code>config["pose"]</code> at the moment it runs. If the
   group swap happened after, the override would apply to the
   about-to-be-replaced <code>yolo</code> config and then get discarded
   entirely when the swap overwrote the whole group afterward. Applying
   overrides strictly left to right against the same mutable config is
   what makes "swap, then tweak" behave as one coherent operation.</details>

5. **Why does `pose.confidence=0.9` end up as a `float` but
   `pose.device=cuda:0` stays a `str`?**
   <details><summary>Answer</summary>Override values arrive from the
   command line as plain text, so they're coerced by trying number
   parsing first: <code>"0.9"</code> successfully parses as a float, but
   <code>"cuda:0"</code> fails both int and float parsing (the colon
   isn't valid in either), so it's left as a string — which happens to
   be exactly the type a device string needs to be.</details>

## Interview questions this module answers

- *How would you let someone swap an entire model or dataset from the
  command line without touching code?*
- *What does `hydra.utils.instantiate` actually do, mechanically?*
- *Design a config system where one field can reference another field's
  value.*
- *Someone runs `python train.py model=vit trainer.max_epochs=50` — walk
  through what happens to the config, step by step.*
