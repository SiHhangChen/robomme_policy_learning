#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data1/workspace/chensihang/membench/policy/robomme_policy_learning}"
OPENPI_ROOT="${OPENPI_ROOT:-/data1/workspace/chensihang/membench/policy/openpi}"
PYTHON_BIN="${PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
PYTHON_PATH="${REPO_ROOT}/src:${OPENPI_ROOT}/src"

WA01_FINAL_METADATA="${REPO_ROOT}/runs/ckpts/mme_vla_omega_wa01_action_condition/wa01_vggt_omega_h8_action_condition_bs48_60k_g0123/60000/_CHECKPOINT_METADATA"
WA05_DATASET="/data1/shared_workspace/chensihang/dataset/membench/wa05_200seeds_v061"
WA05_CACHE="/data1/shared_workspace/chensihang/dataset/baseline/vggt-omega/wa05_vggt_cache"
PI05_PARAMS="/data1/shared_workspace/tangzhipeng/ckpts/openpi/openpi_assets/pi05_base/params"

TRAIN_CONFIG="mme_vla_omega_wa05_action_modulation"
EXP_NAME="wa05_vggt_omega_h8_ah50_action_modulation_bs48_60k_g0123"
TRAIN_RUN="${REPO_ROOT}/runs/ckpts/${TRAIN_CONFIG}/${EXP_NAME}"
AUTOMATION_DIR="${REPO_ROOT}/runs/automation/wa05_after_wa01_action_modulation"
LOCK_FILE="${AUTOMATION_DIR}/queue.lock"
GPU_LIST=(0 1 2 3)
WAIT_SECONDS="${WAIT_SECONDS:-30}"

mkdir -p "${AUTOMATION_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "$(date --iso-8601=seconds) another WA05 queue watcher already holds ${LOCK_FILE}"
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
  local pids
  if ! pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
    return 2
  fi
  grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$' <<<"${pids}"
}

wait_for_gpus() {
  local busy unavailable gpu rc
  while true; do
    busy=0
    unavailable=0
    for gpu in "${GPU_LIST[@]}"; do
      if gpu_is_busy "${gpu}"; then
        busy=1
      else
        rc=$?
        if [[ "${rc}" == 2 ]]; then
          unavailable=1
        fi
      fi
    done
    if [[ "${busy}" == 0 && "${unavailable}" == 0 ]]; then
      log "GPUs ${GPU_LIST[*]} are free"
      return
    fi
    if [[ "${unavailable}" == 1 ]]; then
      log "GPU status unavailable; checking again in ${WAIT_SECONDS}s"
    else
      log "waiting for GPUs ${GPU_LIST[*]}; checking again in ${WAIT_SECONDS}s"
    fi
    sleep "${WAIT_SECONDS}"
  done
}

[[ -x "${PYTHON_BIN}" ]] || fail "missing Python executable: ${PYTHON_BIN}"
[[ -d "${WA05_DATASET}" ]] || fail "missing WA05 dataset: ${WA05_DATASET}"
[[ -d "${PI05_PARAMS}" ]] || fail "missing Pi05 params: ${PI05_PARAMS}"
[[ -f "${WA05_CACHE}/metadata.json" ]] || fail "missing WA05 Omega cache metadata"
[[ -f "${WA05_CACHE}/completed.u8.npy" ]] || fail "missing WA05 Omega cache completion bitmap"
[[ ! -e "${TRAIN_RUN}" ]] || fail "WA05 training directory already exists: ${TRAIN_RUN}"

cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export PI05_BASE_PARAMS_PATH="${PI05_PARAMS}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

log "validating WA05 horizon-50 action-modulation config"
"${PYTHON_BIN}" -c \
  'import sys; from mme_vla_omega_suite.training.config import get_config; c = get_config(sys.argv[1]); assert c.model.action_horizon == 50; assert c.batch_size == 48; assert c.num_train_steps == 60000; print(f"config={c.name} action_horizon={c.model.action_horizon} batch_size={c.batch_size} steps={c.num_train_steps}")' \
  "${TRAIN_CONFIG}"

log "verifying WA05 Omega cache completion bitmap"
"${PYTHON_BIN}" -c \
  'import sys, numpy as np; a = np.load(sys.argv[1], mmap_mode="r"); done = int(np.count_nonzero(a == 1)); total = len(a); print(f"completed={done}/{total} cache={sys.argv[1]}"); raise SystemExit(0 if done == total else 1)' \
  "${WA05_CACHE}/completed.u8.npy"

log "running WA05 transformed-data smoke test on CPU"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "${PYTHON_BIN}" scripts/omega_data_smoke.py \
  --config-name "${TRAIN_CONFIG}" --batch-size 2 --num-workers 2

log "watching WA01 step 60000 checkpoint every ${WAIT_SECONDS}s: ${WA01_FINAL_METADATA}"
while [[ ! -f "${WA01_FINAL_METADATA}" ]]; do
  log "WA01 is still training; checking again in ${WAIT_SECONDS}s"
  sleep "${WAIT_SECONDS}"
done
log "verified WA01 step 60000 checkpoint"

wait_for_gpus
[[ ! -e "${TRAIN_RUN}" ]] || fail "WA05 training directory appeared before launch: ${TRAIN_RUN}"
log "starting WA05 horizon-50 action-modulation training on GPUs 0,1,2,3: global batch 48, 60000 steps"

export CUDA_VISIBLE_DEVICES=0,1,2,3
exec "${PYTHON_BIN}" scripts/train_mme_vla_omega.py "${TRAIN_CONFIG}" \
  --exp-name "${EXP_NAME}" \
  --batch-size 48 \
  --num-train-steps 60000 \
  --fsdp-devices 1 \
  --num-workers 4 \
  --overwrite \
  --no-resume \
  --no-wandb-enabled
