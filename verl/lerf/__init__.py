import sys
from pathlib import Path

# LeRF/tool (renderer + prompts) is shared with inference.
_TOOL_DIR = Path(__file__).resolve().parents[2] / "tool"
if _TOOL_DIR.is_dir() and str(_TOOL_DIR) not in sys.path:
    sys.path.append(str(_TOOL_DIR))
