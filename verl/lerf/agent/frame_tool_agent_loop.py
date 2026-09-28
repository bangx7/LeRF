"""``FrameToolAgentLoop``: a ToolAgentLoop whose answer turn only sees the rendered frame.

Turn 1 (thinking off) either calls ``draw_reference_frame`` or emits ``NO_TOOL_CALL``. Turn 2
(thinking on) is generated from a rebuilt context instead of the append-only stream:

    tool path:     [system, user+image, assistant(call with reference_object only), tool(rendered image)]
    NO_TOOL_CALL:  [system, user+image, assistant(NO_TOOL_CALL), user(continuation)]

so the model cannot read its own coordinates back in turn 2. Each trajectory is returned as
two rows (turn 1, turn 2), each with the exact context it was generated under; the answer
reward is computed on the last row and broadcast. Rejected calls (text tool responses) take
the stock append path.

``think_budget`` caps turn-2 thinking: if ``</think>`` is not emitted within the budget, the
Qwen3 interrupt sentence is appended (masked out of the loss) and the model answers with the
remaining ``response_length - think_budget`` tokens.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Optional

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import (
    SPEC_DECODE_EXTRA_KEYS,
    AgentData,
    AgentState,
    ToolAgentLoop,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

import lerf  # noqa: F401  (puts LeRF/tool on sys.path)
from prompts import NO_TOOL_CALL_CONTINUATION, NO_TOOL_CALL_MARKER, THINK_INTERRUPT_TEXT

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

KEEP_PARAMS = ("reference_object",)
THINK_CLOSE_TOKEN = "</think>"


class FrameToolAgentLoop(ToolAgentLoop):
    def __init__(self, *args, think_budget: Optional[int | str] = None, name: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        after_tool = getattr(self.rollout_config.multi_turn, "enable_thinking_after_tool", None)
        self.turn2_template_kwargs: Optional[dict[str, Any]] = (
            {"enable_thinking": bool(after_tool)} if after_tool is not None else None
        )

        self.think_budget: Optional[int] = None
        self.think_interrupt_ids: list[int] = []
        self.think_close_id: Optional[int] = None
        if think_budget not in (None, "", "null", "None") and int(think_budget) > 0:
            budget = int(think_budget)
            if budget >= self.response_length:
                raise ValueError(
                    f"think_budget={budget} leaves nothing after </think>; it must be < "
                    f"rollout.response_length={self.response_length}."
                )
            close_id = self.tokenizer.convert_tokens_to_ids(THINK_CLOSE_TOKEN)
            if close_id is None or close_id == getattr(self.tokenizer, "unk_token_id", None):
                raise ValueError(f"{THINK_CLOSE_TOKEN!r} is not a single token of this tokenizer")
            self.think_budget = budget
            self.think_close_id = close_id
            self.think_interrupt_ids = self.tokenizer.encode(THINK_INTERRUPT_TEXT, add_special_tokens=False)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput | list[AgentLoopOutput]:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_multi_modal_info(messages)
        audios = multi_modal_data.get("audios")
        agent_data = AgentData(
            messages=messages,
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=audios,
            mm_processor_kwargs=self._get_mm_processor_kwargs(audios),
            metrics={},
            request_id=uuid.uuid4().hex,
            tools_kwargs=kwargs.get("tools_kwargs", {}),
        )
        agent_data.prompt_messages = list(messages)
        agent_data.tool_call_counts = 0
        agent_data._active_tools = self.tools
        agent_data._active_tool_schemas = self.tool_schemas

        earlier: list[AgentLoopOutput] = []
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
                if state == AgentState.TERMINATED and self._is_no_tool_call_turn(agent_data):
                    state, row = await self._handle_no_tool_call(agent_data, sampling_params)
                    if row is not None:
                        earlier.append(row)
            elif state == AgentState.PROCESSING_TOOLS:
                state, row = await self._handle_tools(agent_data, sampling_params)
                if row is not None:
                    earlier.append(row)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        final = self._finalize_output(agent_data)
        return [*earlier, final] if earlier else final

    def _finalize_output(self, agent_data: AgentData) -> AgentLoopOutput:
        """Package the current token stream as one training row."""
        n_resp = len(agent_data.response_mask)
        response_ids = agent_data.prompt_ids[-n_resp:] if n_resp else []
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - n_resp]
        multi_modal_data: dict[str, Any] = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = list(agent_data.image_data)
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        if agent_data.audio_data is not None:
            multi_modal_data["audios"] = agent_data.audio_data
        extra_fields = dict(agent_data.extra_fields)
        extra_fields.update(
            {
                "turn_scores": list(agent_data.turn_scores),
                "tool_rewards": list(agent_data.tool_rewards),
                "tool_call_counts": int(getattr(agent_data, "tool_call_counts", 0)),
                "hidden_tool_call": bool(getattr(agent_data, "hidden_tool_call", False)),
                "no_tool_call": bool(getattr(agent_data, "no_tool_call", False)),
                "think_interrupted": bool(getattr(agent_data, "think_interrupted", False)),
            }
        )
        return AgentLoopOutput(
            prompt_ids=list(prompt_ids),
            response_ids=list(response_ids[: self.response_length]),
            response_mask=list(agent_data.response_mask[: self.response_length]),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            response_logprobs=(
                list(agent_data.response_logprobs[: self.response_length]) if agent_data.response_logprobs else None
            ),
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=dict(agent_data.metrics),
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=extra_fields,
        )

    @staticmethod
    def _placeholder_message(tool_calls: list[FunctionCall]) -> dict[str, Any]:
        """The turn-1 call as turn 2 sees it: only ``KEEP_PARAMS`` are kept."""
        calls = []
        for call in tool_calls:
            try:
                arguments = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            kept = {k: arguments[k] for k in KEEP_PARAMS if k in arguments}
            calls.append({"type": "function", "function": {"name": call.name, "arguments": kept}})
        return {"role": "assistant", "content": "", "tool_calls": calls}

    @staticmethod
    def _tool_message(tool_response: ToolResponse) -> dict[str, Any]:
        if tool_response.image or tool_response.video:
            images = [img for img in tool_response.image or [] if img is not None]
            content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
            if tool_response.video:
                content.append({"type": "video"})
            if tool_response.text:
                content.append({"type": "text", "text": tool_response.text})
            return {"role": "tool", "content": content}
        return {"role": "tool", "content": tool_response.text or ""}

    def _merge_generation_extra_fields(self, agent_data: AgentData, output: TokenOutput) -> None:
        preempted = output.num_preempted if output.num_preempted is not None else 0
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += preempted
        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            if output.extra_fields.get("max_global_steps"):
                agent_data.extra_fields["max_global_steps"] = output.extra_fields["max_global_steps"]
            for key in SPEC_DECODE_EXTRA_KEYS:
                if key in output.extra_fields and key in agent_data.extra_fields:
                    agent_data.extra_fields[key] = int(agent_data.extra_fields[key]) + int(output.extra_fields[key])

    # ------------------------------------------------------------------ tool path
    async def _handle_tools(self, agent_data: AgentData, sampling_params: dict[str, Any]):
        calls = agent_data.tool_calls[: self.max_parallel_calls]
        tasks = [self._call_tool(call, agent_data.tools_kwargs, agent_data) for call in calls]
        agent_data.tool_call_counts += len(tasks)
        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)
        for _, tool_reward, _ in responses:
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        if not (len(responses) == 1 and responses[0][0].image):
            return await self._append_tool_messages(agent_data, responses), None

        tool_response = responses[0][0]
        new_images = [img for img in tool_response.image if img is not None]
        self._assert_mm_supported(bool(new_images))
        row1 = self._finalize_output(agent_data)
        messages2 = [*agent_data.prompt_messages, self._placeholder_message(calls), self._tool_message(tool_response)]
        images2 = [*(agent_data.image_data or []), *new_images]
        if not await self._generate_turn2(agent_data, messages2, images2, sampling_params):
            return AgentState.TERMINATED, None
        agent_data.hidden_tool_call = True
        agent_data.tool_calls = []
        return AgentState.TERMINATED, row1

    # ------------------------------------------------------------------ NO_TOOL_CALL path
    def _is_no_tool_call_turn(self, agent_data: AgentData) -> bool:
        if agent_data.tool_calls or getattr(agent_data, "hidden_tool_call", False):
            return False
        if getattr(agent_data, "no_tool_call", False):
            return False
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return False
        if len(agent_data.response_mask) >= self.response_length:
            return False
        ids = getattr(agent_data, "response_ids", None)
        return bool(ids) and NO_TOOL_CALL_MARKER in self.tokenizer.decode(ids)

    async def _handle_no_tool_call(self, agent_data: AgentData, sampling_params: dict[str, Any]):
        row1 = self._finalize_output(agent_data)
        messages2 = [
            *agent_data.prompt_messages,
            {"role": "assistant", "content": NO_TOOL_CALL_MARKER},
            {"role": "user", "content": NO_TOOL_CALL_CONTINUATION},
        ]
        if not await self._generate_turn2(agent_data, messages2, list(agent_data.image_data or []), sampling_params):
            return AgentState.TERMINATED, None
        agent_data.no_tool_call = True
        agent_data.tool_calls = []
        return AgentState.TERMINATED, row1

    # ------------------------------------------------------------------ turn 2
    async def _generate_turn2(self, agent_data: AgentData, messages2, images2, sampling_params) -> bool:
        """Render ``messages2`` as a fresh prompt, generate turn 2, and point the stream at it."""
        prefix = await self.loop.run_in_executor(
            None,
            lambda: self.continuous_token_builder.render_tokens_with_mm(
                messages2,
                images2,
                videos=agent_data.video_data,
                audios=agent_data.audio_data,
                add_generation_prompt=True,
                tools=agent_data._active_tool_schemas,
                chat_template_kwargs=self.turn2_template_kwargs,
            ),
        )
        max_model_len = self.rollout_config.get("max_model_len", None) or (self.prompt_length + self.response_length)
        if len(prefix) >= max_model_len - 1:
            logger.warning("turn-2 context (%d tokens) does not fit max_model_len=%d", len(prefix), max_model_len)
            return False
        if len(prefix) > self.prompt_length:
            logger.warning("turn-2 context (%d tokens) exceeds rollout.prompt_length=%d", len(prefix), self.prompt_length)

        ids, mask, logprobs, interrupted = await self._generate_with_budget(
            agent_data, prefix, images2, sampling_params, max_model_len
        )
        agent_data.prompt_ids = list(prefix) + ids
        agent_data.response_mask = mask
        agent_data.response_logprobs = logprobs
        agent_data.think_interrupted = interrupted
        agent_data.routed_experts = None
        agent_data.image_data = images2
        agent_data.messages = [*messages2, {"role": "assistant", "content": self.tokenizer.decode(ids)}]
        agent_data.assistant_turns += 1
        agent_data.user_turns += 1
        return True

    async def _generate_with_budget(self, agent_data, prefix, images, sampling_params, max_model_len):
        """Returns ``(ids, mask, logprobs, interrupted)``; the injected sentence has mask 0."""

        async def gen(prompt_ids: list[int], max_tokens: Optional[int]) -> TokenOutput:
            params = dict(sampling_params)
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            with simple_timer("generate_sequences", agent_data.metrics):
                out: TokenOutput = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=params,
                    image_data=images,
                    video_data=agent_data.video_data,
                    audio_data=agent_data.audio_data,
                    mm_processor_kwargs=agent_data.mm_processor_kwargs,
                )
            self._merge_generation_extra_fields(agent_data, out)
            return out

        if not self.think_budget:
            out = await gen(prefix, None)
            ids = list(out.token_ids)
            return ids, [1] * len(ids), list(out.log_probs) if out.log_probs else [], False

        cap = max(1, min(self.response_length, max_model_len - len(prefix)))
        think_max = max(1, min(self.think_budget, cap))
        out = await gen(prefix, think_max)
        ids = list(out.token_ids)
        mask = [1] * len(ids)
        logprobs = list(out.log_probs) if out.log_probs else []
        if len(ids) < think_max or len(ids) >= cap:  # stopped on its own, or no room left
            return ids, mask, logprobs, False

        # budget hit: close the thought unless </think> was already written
        fits = cap - len(ids) > len(self.think_interrupt_ids)
        injected = [] if (self.think_close_id in ids or not fits) else list(self.think_interrupt_ids)
        ids += injected
        mask += [0] * len(injected)
        if logprobs:
            logprobs += [0.0] * len(injected)

        room = cap - len(ids)
        if room > 0:
            out = await gen(list(prefix) + ids, room)
            tail = list(out.token_ids)
            ids += tail
            mask += [1] * len(tail)
            if logprobs:
                logprobs += list(out.log_probs) if out.log_probs else [0.0] * len(tail)
        return ids, mask, logprobs, bool(injected)

    # ------------------------------------------------------------------ rejected call: stock append path
    async def _append_tool_messages(self, agent_data: AgentData, responses) -> AgentState:
        previous_messages = list(agent_data.messages)
        new_images: list[Any] = []
        for i, (tool_response, _, _) in enumerate(responses):
            message = self._tool_message(tool_response)
            if agent_data.tool_calls[i].tool_call_id is not None:
                message["tool_call_id"] = agent_data.tool_calls[i].tool_call_id
            agent_data.messages.append(message)
            if tool_response.image:
                new_images.extend(img for img in tool_response.image if img is not None)
            if tool_response.video:
                raise NotImplementedError("Only image tool responses are supported.")
        self._assert_mm_supported(bool(new_images))

        merge_kwargs: dict[str, Any] = {"tools": agent_data._active_tool_schemas}
        if self.turn2_template_kwargs is not None:
            merge_kwargs["generation_chat_template_kwargs"] = dict(self.turn2_template_kwargs)
        merge_result, response_mask, response_logprobs = await self.ct_merge_non_assistant_msg(
            previous_messages,
            agent_data.messages,
            agent_data.prompt_ids,
            agent_data.response_mask,
            agent_data.response_logprobs if agent_data.response_logprobs else None,
            **merge_kwargs,
        )
        if len(response_mask) >= self.response_length:
            return AgentState.TERMINATED
        agent_data.prompt_ids = merge_result.token_ids
        agent_data.response_mask = response_mask
        if agent_data.response_logprobs:
            agent_data.response_logprobs = response_logprobs or []
        if new_images:
            agent_data.image_data = [*(agent_data.image_data or []), *new_images]
        agent_data.user_turns += 1
        return AgentState.GENERATING
