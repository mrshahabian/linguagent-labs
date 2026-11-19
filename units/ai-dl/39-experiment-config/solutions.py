"""Reference answers for 39 - experiment configuration management."""

import copy
import importlib
import re


def instantiate(config):
    """Build an object from a config dict via its "_target_" dotted path.

    This is Hydra's `hydra.utils.instantiate` in miniature: "_target_" is a
    string like "collections.OrderedDict" - everything before the last dot
    is a MODULE to import, everything after is the attribute (a class or
    function) to pull out of it. Every other key in the config becomes a
    keyword argument to that callable. The config never imports anything
    itself; it just names what to import, which is what lets swapping an
    entire implementation be a one-line config change instead of an edit
    to the code that constructs it.
    """
    config = dict(config)
    target = config.pop("_target_")
    module_path, _, attr_name = target.rpartition(".")
    module = importlib.import_module(module_path)
    target_callable = getattr(module, attr_name)
    return target_callable(**config)


_INTERPOLATION = re.compile(r"\$\{([\w.]+)\}")


def _lookup(config, dotted_path):
    value = config
    for part in dotted_path.split("."):
        value = value[part]
    return value


def resolve_interpolations(config):
    """Replace every "${a.b.c}" in a string value with the value it names.

    OmegaConf (the config library under Hydra) resolves these lazily and
    against the FINAL merged config, not against whatever was in scope
    when the string was written - which is exactly why interpolation only
    makes sense to resolve after every default and override has already
    been applied, never before. A value can reference a sibling field
    (run_name referencing pose.name) precisely because the whole config
    is one tree by the time this runs.

    Returns a new config with every string value's "${...}" references
    substituted; non-string values pass through unchanged.
    """
    def resolve_value(value):
        if isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        if isinstance(value, str):
            def replace(match):
                return str(_lookup(config, match.group(1)))
            return _INTERPOLATION.sub(replace, value)
        return value

    return resolve_value(config)


def merge_config_groups(defaults, config_groups):
    """Assemble a config from named GROUPS, each with several options.

    config_groups looks like {"pose": {"yolo": {...}, "mediapipe": {...}}}
    - a family of interchangeable options for the same slot. defaults
    picks exactly one option per group, e.g. {"pose": "yolo"}. The result
    has one key per group, holding a COPY of the selected option's config
    - a copy, so later mutating the merged result (via overrides) never
    corrupts the original config_groups dict that other composed configs
    might still read from.
    """
    return {
        group: copy.deepcopy(config_groups[group][defaults[group]])
        for group in defaults
    }


def _coerce(raw_value):
    for converter in (int, float):
        try:
            return converter(raw_value)
        except ValueError:
            pass
    if raw_value in ("true", "false"):
        return raw_value == "true"
    return raw_value


def apply_overrides(config, config_groups, overrides):
    """Apply Hydra-style CLI overrides, in order, to a merged config.

    Two shapes, distinguished by whether the left side has a dot:

      "pose=mediapipe"        -> swap the ENTIRE pose group for a
                                  different option from config_groups
      "pose.confidence=0.9"   -> walk into the nested dict and set just
                                  that one leaf, leaving everything else
                                  in that group untouched

    Overrides apply left to right, so "pose=mediapipe pose.device=cuda:0"
    swaps the whole group first and then overrides a key ON TOP of the
    newly-swapped-in config - which only works because group swaps and
    key overrides are applied to the same mutable config in sequence,
    not computed independently and merged at the end.

    Values are coerced: "0.9" becomes a float, "50" an int, "true"/"false"
    a bool, anything else (like "cuda:0") stays a string.
    """
    config = copy.deepcopy(config)
    for override in overrides:
        key, _, raw_value = override.partition("=")
        if "." not in key and key in config_groups:
            config[key] = copy.deepcopy(config_groups[key][raw_value])
            continue

        *path, leaf = key.split(".")
        target = config
        for part in path:
            target = target[part]
        target[leaf] = _coerce(raw_value)

    return config
