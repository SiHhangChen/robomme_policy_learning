#!/usr/bin/env bash
set -euo pipefail

WA05_PID="${WA05_PID:-2186269}"
REPO_ROOT="${REPO_ROOT:-/data1/workspace/chensihang/membench/policy/robomme_policy_learning}"
OPENPI_ROOT="${OPENPI_ROOT:-/data1/workspace/chensihang/membench/policy/openpi}"
PYTHON_BIN="${PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
PYTHON_PATH="${REPO_ROOT}/src:${OPENPI_ROOT}/src"

WA05_RUN="/data1/shared_workspace/chensihang/ckpts/membench/pi05/wa05/checkpoints/pi05_membench_wa05_vggt_omega_memory_bank_lora/wa05-fulltask-vggt-omega-h8-b48-60k-fsdp1-final"
WA05_FINAL_METADATA="${WA05_RUN}/60000/_CHECKPOINT_METADATA"
TS01_DATASET="/data1/shared_workspace/chensihang/dataset/membench/ts01_200seeds_v061"
TS01_CACHE="/data1/shared_workspace/chensihang/dataset/baseline/vggt-omega/ts01_vggt_cache"
OMEGA_REPO="/data1/workspace/chensihang/pretrained/vggt-omega"
OMEGA_CHECKPOINT="/data1/shared_workspace/chensihang/pretrained/vggt_omega_1b_256_text.pt"
PI05_PARAMS="/data1/shared_workspace/tangzhipeng/ckpts/openpi/openpi_assets/pi05_base/params"

TRAIN_CONFIG="mme_vla_omega_ts01_action_modulation"
EXP_NAME="ts01_vggt_omega_h8_bs48_60k_g0123"
TRAIN_RUN="${REPO_ROOT}/runs/ckpts/${TRAIN_CONFIG}/${EXP_NAME}"
AUTOMATION_DIR="${REPO_ROOT}/runs/automation/ts01_after_wa05_context"
LOCK_FILE="${AUTOMATION_DIR}/queue.lock"

mkdir -p "${AUTOMATION_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "$(date --iso-8601=seconds) another TS01 queue watcher already holds ${LOCK_FILE}"
  exit 1
fi

log() {
  echo "$(date --iso-8601=seconds) $*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

gpu_is_busy() {
  local gpu="$1"
  nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$'
}

wait_for_gpus() {
  local busy
  while true; do
    busy=0
    for gpu in 0 1 2 3; do
      if gpu_is_busy "${gpu}"; then
        busy=1
      fi
    done
    if [[ "${busy}" == 0 ]]; then
      log "GPUs 0,1,2,3 are free"
      return
    fi
    log "waiting for GPUs 0,1,2,3 to become free"
    sleep 60
  done
}

[[ -x "${PYTHON_BIN}" ]] || fail "missing Python executable: ${PYTHON_BIN}"
[[ -d "${TS01_DATASET}" ]] || fail "missing TS01 dataset: ${TS01_DATASET}"
[[ -d "${OMEGA_REPO}" ]] || fail "missing VGGT-Omega repository: ${OMEGA_REPO}"
[[ -f "${OMEGA_CHECKPOINT}" ]] || fail "missing VGGT-Omega checkpoint: ${OMEGA_CHECKPOINT}"
[[ -d "${PI05_PARAMS}" ]] || fail "missing Pi05 params: ${PI05_PARAMS}"
[[ ! -e "${TRAIN_RUN}" ]] || fail "TS01 training directory already exists: ${TRAIN_RUN}"

log "watching WA05 context PID ${WA05_PID}"
while kill -0 "${WA05_PID}" 2>/dev/null; do
  sleep 60
done
log "WA05 context PID ${WA05_PID} exited"

[[ -f "${WA05_FINAL_METADATA}" ]] || fail "WA05 exited without a complete step 60000 checkpoint"
log "verified WA05 step 60000 checkpoint"
wait_for_gpus

cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export PI05_BASE_PARAMS_PATH="${PI05_PARAMS}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cache_args=(
  scripts/extract_lerobot_omega_cache.py
  --dataset-root "${TS01_DATASET}"
  --omega-repo "${OMEGA_REPO}"
  --checkpoint "${OMEGA_CHECKPOINT}"
  --output-dir "${TS01_CACHE}"
  --decision-stride-frames 50
)

if [[ ! -f "${TS01_CACHE}/metadata.json" ]]; then
  log "initializing TS01 Omega cache"
  "${PYTHON_BIN}" "${cache_args[@]}" --initialize-only
fi

if "${PYTHON_BIN}" "${cache_args[@]}" --verify-only; then
  log "TS01 Omega cache is already complete"
else
  log "extracting TS01 Omega cache with four GPU shards"
  cache_pids=()
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON_BIN}" "${cache_args[@]}" \
      --num-shards 4 --shard-index "${shard}" --device cuda:0 --batch-size 1 --resume \
      > "${AUTOMATION_DIR}/cache_gpu${shard}.log" 2>&1 &
    cache_pids+=("$!")
  done
  cache_failed=0
  for cache_pid in "${cache_pids[@]}"; do
    if ! wait "${cache_pid}"; then
      cache_failed=1
    fi
  done
  [[ "${cache_failed}" == 0 ]] || fail "one or more TS01 cache shards failed; see cache_gpu*.log"
  "${PYTHON_BIN}" "${cache_args[@]}" --verify-only
  log "verified complete TS01 Omega cache"
fi

log "running TS01 transformed-data smoke test"
"${PYTHON_BIN}" scripts/omega_data_smoke.py \
  --config-name "${TRAIN_CONFIG}" --batch-size 2 --num-workers 2

[[ ! -e "${TRAIN_RUN}" ]] || fail "TS01 training directory appeared before launch: ${TRAIN_RUN}"
sleep 15
wait_for_gpus
log "starting TS01 training on GPUs 0,1,2,3: global batch 48, 60000 steps"

export CUDA_VISIBLE_DEVICES=0,1,2,3
exec "${PYTHON_BIN}" scripts/train_mme_vla_omega.py "${TRAIN_CONFIG}" \
  --exp-name "${EXP_NAME}" \
  --batch-size 48 \
  --num-train-steps 60000 \
  --fsdp-devices 4 \
  --num-workers 4 \
  --no-wandb-enabled
