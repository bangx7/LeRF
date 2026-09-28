"""Answer reward for multiple-choice spatial questions.

1.0 iff the response contains exactly one ``\\boxed{...}`` and it matches the ground truth
(option key such as ``c`` / ``(C)``, the option text, or both, if consistent). Tool use is
not rewarded; it is only logged.

``ground_truth`` is a JSON string::

    {"answer_key": "c", "answer_text": "black laptop", "options": {"a": "...", ...}}
"""

from __future__ import annotations

import json
import numbers
import re

import numpy as np

_BOXED = "\\boxed{"
_TEXT_WRAPPER = re.compile(r"\\(?:text|mathrm|textbf|mathbf)\{([^{}]*)\}")
_ARTICLE = re.compile(r"^(?:the|a|an)\s+")
# "(c) text" / "c) text" / "c. text" / "c: text" / "[c] text"
_KEYED = re.compile(r"^[\(\[]?\s*([a-z]|\d{1,2})\s*(?:[\)\]]\s*[\.\:\-]?|[\.\:\-])\s*(.*)$")
_BARE_KEY = re.compile(r"^[\(\[]?\s*([a-z]|\d{1,2})\s*[\)\]]?$")
THINK_INTERRUPT_MARK = "Considering the limited time by the user"


def extract_boxed(text: str) -> list[str]:
    """All brace-balanced ``\\boxed{...}`` bodies, in order."""
    out, i = [], 0
    while True:
        start = text.find(_BOXED, i)
        if start < 0:
            return out
        j, depth = start + len(_BOXED), 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth:
            return out
        out.append(text[start + len(_BOXED) : j - 1])
        i = j


def normalize(s: str) -> str:
    s = _TEXT_WRAPPER.sub(r"\1", s)
    s = s.replace("$", " ").replace("*", " ").replace("`", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.rstrip(" .;,!")


def _norm_text(s: str) -> str:
    return _ARTICLE.sub("", normalize(s))


def judge(boxed: str, gt: dict) -> bool:
    options = {str(k).lower(): _norm_text(v) for k, v in gt["options"].items()}
    key_gt = str(gt["answer_key"]).lower()
    by_text: dict[str, list[str]] = {}
    for k, v in options.items():
        by_text.setdefault(v, []).append(k)

    b = normalize(boxed)
    if not b:
        return False
    m = _BARE_KEY.match(b)
    if m and m.group(1) in options:
        return m.group(1) == key_gt
    m = _KEYED.match(b)
    if m and m.group(1) in options:
        key, rest = m.group(1), _norm_text(m.group(2))
        if key != key_gt:
            return False
        return not (rest and rest in by_text and key not in by_text[rest])  # key and text must agree

    t = _norm_text(b)
    if t in by_text:
        return key_gt in by_text[t]
    m = re.match(r"^(.*?)\s*[\(\[]\s*([a-z]|\d{1,2})\s*[\)\]]$", t)  # "black laptop (c)"
    if m and m.group(2) in options:
        rest = m.group(1)
        return m.group(2) == key_gt and not (rest in by_text and key_gt not in by_text[rest])
    return False


def _flag(value) -> float | None:
    return float(value) if isinstance(value, (bool, np.bool_)) else None


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    text = solution_str or ""
    boxes = extract_boxed(text)
    score = float(len(boxes) == 1 and judge(boxes[0], gt))

    info = extra_info if isinstance(extra_info, dict) else {}
    tool_called = float("<tool_call>" in text)
    counts = info.get("tool_call_counts")
    if isinstance(counts, numbers.Real) and not isinstance(counts, (bool, np.bool_)):
        tool_called = float(counts >= 1)
    result = {
        "score": score,
        "acc": score,
        "n_boxed": float(len(boxes)),
        "no_answer": float(len(boxes) == 0),
        "tool_called": tool_called,
        "has_think_close": float("</think>" in text),
        "think_interrupted": float(THINK_INTERRUPT_MARK in text),
    }
    for key in ("hidden_tool_call", "no_tool_call", "think_interrupted"):
        if (flag := _flag(info.get(key))) is not None:
            result[key] = flag
    return result
