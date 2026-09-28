"""Agent-loop manager whose reward batch only holds each trajectory's final row.

The stock ``_compute_score`` pads all rows of a multi-row trajectory together, which fails for
VL models (M-RoPE position ids of shape ``(3, L)``). The reward only reads the final row anyway.

    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=lerf.agent.agent_loop_worker.FrameAgentLoopManagerTQ
"""

from __future__ import annotations

from typing import Any

import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.trainer.ppo.v1 import agent_loop_tq as _tq

# Subclass the class Ray instantiates, not the undecorated original.
_BaseWorkerTQ = _tq.AgentLoopWorkerTQ.__ray_metadata__.modified_class


class FrameAgentLoopWorkerTQ(_BaseWorkerTQ):
    async def _compute_score(self, outputs: list[AgentLoopOutput], kwargs: dict[str, Any]) -> None:
        await super()._compute_score(outputs[-1:], kwargs)


class FrameAgentLoopManagerTQ(_tq.AgentLoopManagerTQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(FrameAgentLoopWorkerTQ)
