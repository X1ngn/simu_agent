from __future__ import annotations

import csv
import os
from typing import Any, Dict, List


def read_csv_head_row(path: str) -> Dict[str, Any]:
    """
    读取 CSV：跳过表头，读取第一行数据（最优结果假设在第一行）。
    返回 dict: {col: value}
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)
    return {}


def read_csv_all(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        # 写空文件也行，但至少要有表头：这里直接创建空文件
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
