#!/usr/bin/env python3
"""LeRF inference with the ``draw_reference_frame`` tool, against a vLLM OpenAI-compatible server.

Same protocol as RL training (``verl/lerf/agent/frame_tool_agent_loop.py``):

    turn 1 (thinking off)  ->  <tool_call>draw_reference_frame(...)</tool_call>  or  NO_TOOL_CALL
    tool                   ->  frame rendered on the image
    turn 2 (thinking on)   ->  reasoning + \\boxed{answer}

Turn 2 is generated from a rebuilt context in which the tool call keeps only
``reference_object``; the rendered frame is the only geometry the model sees.

Single question:
    python inference.py --model lerf --image img.jpg --question "..." --options "left" "right"
Batch (JSONL with image / question / options [/ answer / id]):
    python inference.py --model lerf --input data.jsonl --output results.jsonl
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

import frame_tool
from prompts import (
    FRAME_TOOL,
    NO_TOOL_CALL_CONTINUATION,
    NO_TOOL_CALL_MARKER,
    SYSTEM_PROMPT,
    THINK_INTERRUPT_TEXT,
    TOOL_NAME,
    build_user_text,
)

KEEP_PARAMS = ("reference_object",)
_THINK_EXIT = THINK_INTERRUPT_TEXT.strip()[: -len("\n</think>")]

# ---------------------------------------------------------------- tool-call parsing (qwen3_coder format)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL)
_PARAM_TYPES = {k: v.get("type", "string") for k, v in FRAME_TOOL["function"]["parameters"]["properties"].items()}


def _convert_param(value: str, name: str):
    if value.lower() == "null":
        return None
    if _PARAM_TYPES.get(name, "string") == "string":
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def extract_first_call(text: str):
    """Return ``(name, params)`` of the first tool call in ``text``, or ``None``."""
    if "<tool_call>" not in text:
        return None
    raw_calls = [m[0] or m[1] for m in _TOOL_CALL_RE.findall(text)] or [text]
    for raw in raw_calls:
        for fm in _FUNCTION_RE.findall(raw):
            body = fm[0] or fm[1]
            end = body.find(">")
            if end == -1:
                continue
            params = {}
            for pm in _PARAMETER_RE.findall(body[end + 1 :]):
                ptext = pm[0] or pm[1]
                idx = ptext.find(">")
                if idx == -1:
                    continue
                value = ptext[idx + 1 :]
                value = value[1:] if value.startswith("\n") else value
                value = value[:-1] if value.endswith("\n") else value
                params[ptext[:idx]] = _convert_param(value, ptext[:idx])
            return body[:end], params
    return None


def placeholder_call(params: dict) -> str:
    """The turn-1 call as turn 2 sees it (same bytes the chat template renders from ``tool_calls``)."""
    body = ""
    for key in KEEP_PARAMS:
        if key in params:
            value = params[key]
            value = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
            body += f"<parameter={key}>\n{value}\n</parameter>\n"
    return f"<tool_call>\n<function={TOOL_NAME}>\n{body}</function>\n</tool_call>"


def extract_boxed(text: str) -> list[str]:
    out, i = [], 0
    while (start := text.find("\\boxed{", i)) >= 0:
        j, depth = start + len("\\boxed{"), 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth:
            break
        out.append(text[start + len("\\boxed{") : j - 1])
        i = j
    return out


# ---------------------------------------------------------------- messages / HTTP
def image_part(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def tool_message(image: Image.Image | None = None, text: str | None = None) -> dict:
    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    if image is not None:
        parts.append(image_part(image))
    return {"role": "tool", "content": parts, "tool_call_id": TOOL_NAME}


_local = threading.local()


def chat(args, messages, *, enable_thinking: bool, max_tokens: int, extra: dict | None = None, tries: int = 6):
    """One chat completion; returns ``(text, finish_reason)``.

    ``tool_choice="none"`` keeps the raw ``<tool_call>`` XML in the message content.
    """
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": max_tokens,
        "tools": [FRAME_TOOL],
        "tool_choice": "none",
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        **(extra or {}),
    }
    if getattr(_local, "session", None) is None:
        _local.session = requests.Session()
    url = args.base_url.rstrip("/") + "/chat/completions"
    last = None
    for attempt in range(tries):
        try:
            r = _local.session.post(url, json=payload, timeout=(10, args.timeout))
        except requests.RequestException as e:
            last = e
        else:
            if r.ok:
                choice = r.json()["choices"][0]
                msg = choice.get("message") or {}
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                if reasoning and "</think>" not in content:  # server-side reasoning parser enabled
                    content = f"{reasoning}\n</think>\n\n{content}"
                return content, choice.get("finish_reason")
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        time.sleep(min(8.0, 2.0**attempt))
    raise RuntimeError(f"request failed after {tries} tries: {last}")


def answer_turn(args, messages):
    """Thinking turn with a soft budget: stop thinking at ``think_budget``, then answer."""
    text, finish = chat(args, messages, enable_thinking=True, max_tokens=args.think_budget)
    if finish != "length":
        return text, finish, False

    scrub = lambda s: s.replace("<think>", "").replace("</think>", "")  # noqa: E731
    if "</think>" in text:
        thinking, answer = text.split("</think>", 1)
        prefix = scrub(thinking).strip() + "\n</think>\n\n" + scrub(answer).lstrip("\n")
        interrupted = False
    else:
        prefix = scrub(text).strip() + "\n\n" + _THINK_EXIT + "\n</think>"
        interrupted = True
    cont, finish = chat(
        args,
        [*messages, {"role": "assistant", "content": prefix}],
        enable_thinking=True,
        max_tokens=args.max_tokens - args.think_budget,
        extra={"add_generation_prompt": False, "continue_final_message": True},
    )
    return prefix + cont, finish, interrupted


# ---------------------------------------------------------------- episode
def run_episode(args, image: Image.Image, user_text: str) -> dict:
    """Run one question; returns every turn's full text, the rendered frame and the final answer."""
    base = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [image_part(image), {"type": "text", "text": "\n" + user_text}]},
    ]
    messages = list(base)
    turns: list[dict] = []
    params = render = None

    # turn 1; a rejected call is shown back to the model (thinking on) and retried
    for attempt in range(1 + args.max_retries):
        thinking = attempt > 0
        text, finish = chat(
            args, messages, enable_thinking=thinking, max_tokens=args.retry_max_tokens if thinking else args.turn1_max_tokens
        )
        turn = {"phase": "tool_call" if attempt == 0 else f"retry_{attempt}", "completion": text, "finish_reason": finish}
        turns.append(turn)
        call = extract_first_call(text)
        if call is None:
            break
        name, call_params = call
        turn["tool_call"] = {"name": name, "params": call_params}
        code = "params" if name != TOOL_NAME else frame_tool.validate_arguments(call_params)
        if code is None:
            params, render = call_params, frame_tool.execute(image, call_params)
            break
        turn["rejected"] = code
        messages += [{"role": "assistant", "content": text}, tool_message(text=frame_tool.RETRY_MESSAGES[code])]

    last = turns[-1]["completion"]
    if render is not None:
        messages2 = [*base, {"role": "assistant", "content": placeholder_call(params)}, tool_message(image=render.image)]
    elif len(turns) == 1 and NO_TOOL_CALL_MARKER in last and extract_first_call(last) is None:
        messages2 = [
            *base,
            {"role": "assistant", "content": NO_TOOL_CALL_MARKER},
            {"role": "user", "content": NO_TOOL_CALL_CONTINUATION},
        ]
    else:
        messages2 = None  # answered directly, or no valid call after retries

    if messages2 is not None:
        text, finish, interrupted = answer_turn(args, messages2)
        turns.append({"phase": "answer", "completion": text, "finish_reason": finish, "think_interrupted": interrupted})

    boxed = extract_boxed(turns[-1]["completion"])
    return {
        "turns": turns,
        "tool_called": render is not None,
        "no_tool_call": messages2 is not None and render is None,
        "tool_params": params,
        "prediction": boxed[-1] if boxed else None,
        "render": render.image if render is not None else None,
    }


def _norm(s) -> str:
    return re.sub(r"[\s\(\)\[\]\.:]", "", str(s)).lower()


def is_correct(prediction, answer, options: list[str]) -> bool:
    """``answer`` is an option index or letter; accepts the letter or the option text."""
    if prediction is None:
        return False
    key = chr(97 + answer) if isinstance(answer, int) else _norm(answer)
    pred = _norm(prediction)
    idx = ord(key) - 97 if len(key) == 1 and key.isalpha() else -1
    return pred == key or (0 <= idx < len(options) and pred == _norm(options[idx]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--image")
    ap.add_argument("--question")
    ap.add_argument("--options", nargs="*", default=[])
    ap.add_argument("--input", help="JSONL: image, question, options, optional answer/id")
    ap.add_argument("--output", default="results.jsonl")
    ap.add_argument("--save_renders", help="directory for rendered frames")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--turn1_max_tokens", type=int, default=4096)
    ap.add_argument("--retry_max_tokens", type=int, default=8192)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--think_budget", type=int, default=10240)
    ap.add_argument("--max_tokens", type=int, default=12288, help="total budget of the answer turn")
    ap.add_argument("--max_pixels", type=int, default=1024 * 1024)
    ap.add_argument("--timeout", type=float, default=2000.0)
    args = ap.parse_args()

    render_dir = Path(args.save_renders) if args.save_renders else None
    if render_dir:
        render_dir.mkdir(parents=True, exist_ok=True)

    def run(i: int, item: dict) -> dict:
        image = frame_tool.load_image(item["image"], args.max_pixels)
        out = run_episode(args, image, build_user_text(item["question"], item.get("options") or []))
        render = out.pop("render")
        rid = str(item.get("id", i))
        if render is not None and render_dir:
            render.save(render_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', rid)}.png")
        record = {"id": rid, **{k: item[k] for k in ("image", "question", "options") if k in item}, **out}
        if "answer" in item:
            record["correct"] = is_correct(out["prediction"], item["answer"], item.get("options") or [])
        return record

    if args.input:
        items = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    else:
        if not (args.image and args.question):
            ap.error("give --image and --question, or --input")
        items = [{"image": args.image, "question": args.question, "options": args.options}]

    results = [None] * len(items)
    with ThreadPoolExecutor(args.workers) as pool, open(args.output, "w") as f:
        futures = {pool.submit(run, i, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            rec = fut.result()
            results[futures[fut]] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    if len(items) == 1:
        for t in results[0]["turns"]:
            print(f"----- {t['phase']} -----\n{t['completion']}\n")
        print("prediction:", results[0]["prediction"])
    scored = [r for r in results if "correct" in r]
    if scored:
        print(f"accuracy: {sum(r['correct'] for r in scored) / len(scored):.4f} ({len(scored)} scored)")
    print(f"tool call rate: {sum(r['tool_called'] for r in results) / len(results):.4f}; wrote {args.output}")


if __name__ == "__main__":
    main()
