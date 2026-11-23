"""Numeric grounding: every number in the reply must come from a tool result of the same turn
(or from the user's own question). Display rounding and percentages are accepted."""

from __future__ import annotations

import contextlib
import json
import re
from typing import Any

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")
NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
WARNING = "\n\n[grounding check] Some numbers above could not be traced to a tool result: {}. Treat them as unverified."


def numbers_in_text(text: str) -> list[tuple[str, float]]:
    out = []
    for m in NUM.finditer(text.translate(ARABIC_DIGITS)):
        raw = m.group(0)
        try:
            out.append((raw, float(raw.replace(",", ""))))
        except ValueError:
            continue
    return out


def _flatten(obj: Any) -> list[float]:
    vals: list[float] = []
    if isinstance(obj, bool):
        return vals
    if isinstance(obj, int | float):
        vals.append(float(obj))
    elif isinstance(obj, str):
        vals += [v for _, v in numbers_in_text(obj)]
        with contextlib.suppress(ValueError, TypeError):
            vals += _flatten(json.loads(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            vals += _flatten(k) if isinstance(k, str) else []
            vals += _flatten(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            vals += _flatten(v)
    return vals


def _decimals(raw: str) -> int:
    return len(raw.split(".")[1]) if "." in raw else 0


def _matches(raw: str, n: float, v: float) -> bool:
    tol = 0.5 * 10 ** (-_decimals(raw)) + 1e-9
    if abs(n - v) <= tol or abs(n - round(v)) <= tol and _decimals(raw) == 0 and abs(v) >= 1 and abs(n - v) < 1:
        return True
    if abs(n - v * 100) <= tol:  # shown as a percentage
        return True
    # shown in thousands or millions ("1.2M", "350K")
    return any(abs(n - v / s) <= tol for s in (1e3, 1e6))


def check(reply: str, tool_results: list[Any], question: str = "") -> dict[str, Any]:
    allowed = _flatten(tool_results) + [v for _, v in numbers_in_text(question)]
    found = numbers_in_text(reply)
    untraced = [raw for raw, n in found if not any(_matches(raw, n, v) for v in allowed)]
    return {"numbers": len(found), "untraced": untraced, "grounded": not untraced}


def enforce(reply: str, tool_results: list[Any], question: str = "") -> tuple[str, dict[str, Any]]:
    res = check(reply, tool_results, question)
    if not res["grounded"]:
        reply += WARNING.format(", ".join(res["untraced"]))
    return reply, res
