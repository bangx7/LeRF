"""verl tool wrapper around ``LeRF/tool/frame_tool.py``."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from PIL import Image
from pydantic import BaseModel, Field

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)

import lerf  # noqa: F401  (puts LeRF/tool on sys.path)
import frame_tool
from prompts import FRAME_TOOL

logger = logging.getLogger(__name__)


# verl's default property schema drops items/minItems/maxItems; keep them so the rendered
# <tools> block is the full FRAME_TOOL.
class _PropertySchema(BaseModel):
    type: str | list[str]
    items: Optional[dict[str, Any]] = None
    minItems: Optional[int] = None
    maxItems: Optional[int] = None
    description: Optional[str] = None
    enum: Optional[list[Any]] = None


class _ParametersSchema(OpenAIFunctionParametersSchema):
    properties: dict[str, _PropertySchema]


class _FunctionSchema(OpenAIFunctionSchema):
    parameters: _ParametersSchema = Field(
        default_factory=lambda: _ParametersSchema(type="object", properties={}, required=[])
    )


class FrameToolSchema(OpenAIFunctionToolSchema):
    function: _FunctionSchema


class DrawReferenceFrameTool(BaseTool):
    """Render the model's predicted reference frame onto the prompt image."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        super().__init__(config or {}, tool_schema or self.get_openai_tool_schema())
        self._instances: dict[str, dict[str, Any]] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return FrameToolSchema.model_validate(FRAME_TOOL)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        create_kwargs = kwargs.get("create_kwargs") or {}
        self._instances[instance_id] = {"image_path": create_kwargs.get("image_path")}
        return instance_id, ToolResponse()

    def _source_image(self, instance_id: str, agent_data: Any) -> Optional[Image.Image]:
        images = getattr(agent_data, "image_data", None)
        if images and isinstance(images[0], Image.Image):
            return images[0]
        image_path = self._instances.get(instance_id, {}).get("image_path")
        if image_path and os.path.isfile(image_path):
            with Image.open(image_path) as opened:
                return opened.convert("RGB")
        return None

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = frame_tool.validate_arguments(parameters)
        if code is not None:
            return ToolResponse(text=frame_tool.RETRY_MESSAGES[code]), 0.0, {"rejected": code}
        source = self._source_image(instance_id, kwargs.get("agent_data"))
        if source is None:
            logger.warning("draw_reference_frame: no image for instance %s", instance_id)
            return ToolResponse(text=frame_tool.RETRY_MESSAGES["no_image"]), 0.0, {"rejected": "no_image"}
        result = frame_tool.execute(source, parameters)
        return ToolResponse(image=[result.image]), 0.0, {"degenerate_axes": len(result.warnings)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
