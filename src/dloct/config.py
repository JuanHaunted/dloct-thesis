"""
YAML configs with inheritance and command-line overrides.

A config may name a parent with ``defaults: base.yaml`` (relative to its own directory); keys
are deep-merged, child wins. ``--set a.b=value`` overrides any key, with ``value`` parsed as
YAML (so numbers, booleans and lists work).
"""

import re
from pathlib import Path

import yaml


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads exponent floats without a decimal point (``1e-4``)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+$"),
    list("-+0123456789."),
)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, overrides: list[str] = ()) -> dict:
    path = Path(path)
    cfg = yaml.load(path.read_text(), Loader=_Loader) or {}
    parent = cfg.pop("defaults", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse(value)
    return cfg


def _parse(value: str):
    return yaml.load(value, Loader=_Loader)
