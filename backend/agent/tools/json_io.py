from __future__ import annotations

import json
import os
from typing import Any, Dict


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def apply_dotpatch(obj: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    patch 格式：{"a.b.c": 123, "x": "y"}
    会就地修改并返回 obj。
    """
    for k, v in patch.items():
        parts = k.split(".")
        cur: Any = obj
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = v
    return obj
