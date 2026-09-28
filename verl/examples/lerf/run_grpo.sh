#!/usr/bin/env bash
# LeRF stage-2 GRPO with the draw_reference_frame tool.
#
#   turn 1  (thinking off)  draw_reference_frame(...) or NO_TOOL_CALL
#   turn 2  (thinking on)   reasoning over the rendered frame -> \boxed{answer}
#
# Usage:
#   MODEL_PATH=/path/to/sft_ckpt DATA_DIR=/path/to/data bash examples/lerf/run_grpo.sh [hydra overrides]
set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the SFT checkpoint}
DATA_DIR=${DATA_DIR:?set DATA_DIR to the output of lerf/prepare_data.py}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
TEST_FILE=${TEST_FILE:-${DATA_DIR}/test.parquet}

N_GPUS=${N_GPUS:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
# each trajectory is stored as two rows, so 32 keeps ~4 optimizer steps per training step
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-25}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4608}
# turn-2 prompt = dataset prompt + placeholder call + rendered image
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + 1280))}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-12288}
THINK_BUDGET=${THINK_BUDGET:-10240}
MAX_MODEL_LEN=$((ROLLOUT_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
SP_SIZE=${SP_SIZE:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
RAY_OBJECT_STORE_GB=${RAY_OBJECT_STORE_GB:-16}

PROJECT_NAME=${PROJECT_NAME:-lerf}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_$(date +%Y%m%d_%H%M)}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LOGGER=${LOGGER:-'["console","wandb"]'}
mkdir -p "${OUTPUT_DIR}"

export VLLM_USE_FLASHINFER_SAMPLER=0
unset ROCR_VISIBLE_DEVICES  # set by some SLURM setups; verl refuses it alongside CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LERF_THINK_BUDGET="${THINK_BUDGET}"

python -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init.object_store_memory=$((RAY_OBJECT_STORE_GB * 1024 * 1024 * 1024)) \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.LERF_THINK_BUDGET="'${LERF_THINK_BUDGET}'" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.advantage_final_row_only=True \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.image_key=images \
    data.return_raw_chat=True \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.dataloader_num_workers=0 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_MODEL_LEN} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","extra"]' \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_MODEL_LEN} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.prompt_length=${ROLLOUT_PROMPT_LENGTH} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.enable_thinking_after_tool=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${REPO_ROOT}/lerf/tools/tool_config.yaml" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${REPO_ROOT}/lerf/agent/agent_loop_config.yaml" \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=lerf.agent.agent_loop_worker.FrameAgentLoopManagerTQ \
    reward.custom_reward_function.path="${REPO_ROOT}/lerf/reward/reward.py" \
    reward.custom_reward_function.name=compute_score \
    reward.num_workers=8 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.logger="${LOGGER}" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
    "$@"
