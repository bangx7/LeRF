"""Convert multiple-choice spatial QA (JSONL) into the parquet format used by ``examples/lerf/run_grpo.sh``.

Each input line::

    {"image": "path/to/img.jpg", "question": "...", "options": ["left", "right", ...],
     "answer": 1,                      # option index, or letter ("b")
     "id": "optional", "data_source": "optional tag, one validation curve per tag"}

Usage::

    python -m lerf.prepare_data --train train.jsonl --test test.jsonl --out_dir data/lerf
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image

import lerf  # noqa: F401  (puts LeRF/tool on sys.path)
from prompts import SYSTEM_PROMPT, build_user_text

AGENT_NAME = "frame_tool_agent"
TOOL_NAME = "draw_reference_frame"


def prepare_image(path: Path, out_dir: Path, max_pixels: int) -> tuple[str, int, int]:
    """Downscale to at most ``max_pixels`` (JPEG q95); returns the path used for training."""
    with Image.open(path) as im:
        w, h = im.size
        if max_pixels <= 0 or w * h <= max_pixels:
            return str(path.resolve()), w, h
        scale = math.sqrt(max_pixels / (w * h))
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        out = out_dir / f"{path.parent.name}_{path.stem}.jpg"
        if not out.exists():
            im.convert("RGB").resize(size, Image.LANCZOS).save(out, quality=95)
        return str(out.resolve()), *size


def build_row(item: dict, split: str, index: int, image_dir: Path, max_pixels: int) -> dict:
    options = [str(o) for o in item["options"]]
    answer = item["answer"]
    idx = answer if isinstance(answer, int) else ord(str(answer).strip().strip("()").lower()) - 97
    if not 0 <= idx < len(options):
        raise ValueError(f"answer {answer!r} out of range for {len(options)} options: {item}")
    keys = [chr(97 + i) for i in range(len(options))]
    ground_truth = {
        "answer_key": keys[idx],
        "answer_text": options[idx],
        "options": dict(zip(keys, options)),
        "key_style": "letter",
    }
    image_path, width, height = prepare_image(Path(item["image"]), image_dir, max_pixels)
    sample_id = str(item.get("id", f"{split}_{index}"))
    return {
        "data_source": item.get("data_source", "lerf"),
        "agent_name": AGENT_NAME,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<image>\n" + build_user_text(item["question"], options)},
        ],
        "images": [image_path],
        "ability": "spatial_reasoning",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
        "extra_info": {
            "split": split,
            "index": index,
            "id": sample_id,
            "image_path": image_path,
            "image_width": width,
            "image_height": height,
            "need_tools_kwargs": True,
            "tools_kwargs": {TOOL_NAME: {"create_kwargs": {"image_path": image_path}}},
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_pixels", type=int, default=1024 * 1024)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for split, src in (("train", args.train), ("test", args.test)):
        items = [json.loads(line) for line in Path(src).read_text().splitlines() if line.strip()]
        rows = [build_row(item, split, i, image_dir, args.max_pixels) for i, item in enumerate(items)]
        pd.DataFrame(rows).to_parquet(out_dir / f"{split}.parquet", index=False)
        print(f"{split}: {len(rows)} rows -> {out_dir / f'{split}.parquet'}")


if __name__ == "__main__":
    main()
