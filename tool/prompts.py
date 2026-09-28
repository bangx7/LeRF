"""Prompts and constants shared by training (verl) and inference."""

from __future__ import annotations

from pathlib import Path

TOOL_NAME = "draw_reference_frame"

# Rendered by the chat template into the <tools> block of the system turn.
FRAME_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Draw the object-centered reference frame of one reference entity "
            "onto the image. The frame is the 2D perspective projection of "
            "its 3D coordinate frame: an origin at the entity's center or "
            "position, together with indicators for its intrinsic front, "
            "left, and up directions. The tool normally renders these "
            "directions as arrows (red = front, green = left, blue = up). "
            "When an axis is too strongly foreshortened to form a reliable "
            "arrow, the tool may render it at the origin using a depth marker "
            "instead. The tool returns the annotated image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reference_object": {
                    "type": "string",
                    "description": (
                        "A short, unambiguous description of the entity whose "
                        "perspective or orientation defines the frame, e.g. "
                        "'the red office chair by the window'."
                    ),
                },
                "origin": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": (
                        "[x, y] location of the frame origin. Coordinates are "
                        "integers normalized to [0, 1000]: x increases from "
                        "the left edge to the right edge, and y increases "
                        "from the top edge to the bottom edge."
                    ),
                },
                "axis_up": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": (
                        "[x, y] absolute endpoint of the axis that begins at "
                        "the origin and points toward the entity's intrinsic "
                        "up direction after projection onto the image. Uses "
                        "the same normalized coordinate system."
                    ),
                },
                "axis_front": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": (
                        "[x, y] absolute endpoint of the axis that begins at "
                        "the origin and points toward the entity's intrinsic "
                        "front direction after projection onto the image. "
                        "Uses the same normalized coordinate system."
                    ),
                },
                "axis_left": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": (
                        "[x, y] absolute endpoint of the axis that begins at "
                        "the origin and points toward the entity's intrinsic "
                        "left-hand side—not the viewer's left—after "
                        "projection onto the image. Uses the same normalized "
                        "coordinate system."
                    ),
                },
            },
            "required": [
                "reference_object", "origin",
                "axis_up", "axis_front", "axis_left",
            ],
        },
    },
}

SYSTEM_PROMPT = (Path(__file__).resolve().parent / "system_prompt.txt").read_text(encoding="utf-8")

ANSWER_INSTRUCTION = "Answer with the letter of the correct option."

# Turn 1 either calls the tool or emits this marker; the marker is answered by a continuation turn.
NO_TOOL_CALL_MARKER = "NO_TOOL_CALL"
NO_TOOL_CALL_CONTINUATION = (
    "Continue. Reason from the original image using the camera/image perspective and produce the final answer."
)

# Qwen3 thinking-budget sentence, appended when the thinking budget runs out (closes </think>).
THINK_INTERRUPT_TEXT = (
    "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
    "directly now.\n</think>\n\n"
)


def build_user_text(question: str, options: list[str]) -> str:
    """User turn text (without the image): question, ``(a) ...`` option lines, answer instruction."""
    lines = [f"({chr(97 + i)}) {opt}" for i, opt in enumerate(options)]
    return "\n".join([question.strip(), *lines, ANSWER_INSTRUCTION])
