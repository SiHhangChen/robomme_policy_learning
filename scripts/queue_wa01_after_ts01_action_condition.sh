#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data1/workspace/chensihang/membench/policy/robomme_policy_learning}"
OPENPI_ROOT="${OPENPI_ROOT:-/data1/workspace/chensihang/membench/policy/openpi}"
PYTHON_BIN="${PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
PYTHON_PATH="${REPO_ROOT}/src:${OPENPI_ROOT}/src"

TS01_FINAL_METADATA="${REPO_ROOT}/runs/ckpts/mme_vla_omega_ts01_action_modulation/ts01_vggt_omega_h8_bs48_60k_g0123/60000/_CHECKPOINT_METADATA"
WA01_DATASET="/data1/shared_workspace/chensihang/dataset/membench/wa01_200seeds_v061"
WA01_CACHE="/data1/shared_workspace/chensihang/dataset/baseline/vggt-omega/wa01_vggt_cache"
OMEGA_REPO="/data1/workspace/chensihang/pretrained/vggt-omega"
OMEGA_CHECKPOINT="/data1/shared_workspace/chensihang/pretrained/vggt_omega_1b_256_text.pt"
PI05_PARAMS="/data1/shared_workspace/tangzhipeng/ckpts/openpi/openpi_assets/pi05_base/params"

TRAIN_CONFIG="mme_vla_omega_wa01_action_condition"
EXP_NAME="wa01_vggt_omega_h8_action_condition_bs48_60k_g012"
TRAIN_RUN="${REPO_ROOT}/runs/ckpts/${TRAIN_CONFIG}/${EXP_NAME}"
AUTOMATION_DIR="${REPO_ROOT}/runs/automation/wa01_after_ts01_action_condition"
LOCK_FILE="${AUTOMATION_DIR}/queue.lock"
GPU_LIST=(0 1 2 3)
WAIT_SECONDS="${WAIT_SECONDS:-60}"

mkdir -p "${AUTOMATION_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "$(date --iso-8601=seconds) another WA01 queue watcher already holds ${LOCK_FILE}"
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
      log "GPU status unavailable; waiting before launch"
    else
      log "waiting for GPUs ${GPU_LIST[*]} to become free"
    fi
    sleep "${WAIT_SECONDS}"
  done
}

[[ -x "${PYTHON_BIN}" ]] || fail "missing Python executable: ${PYTHON_BIN}"
[[ -d "${WA01_DATASET}" ]] || fail "missing WA01 dataset: ${WA01_DATASET}"
[[ -d "${OMEGA_REPO}" ]] || fail "missing VGGT-Omega repository: ${OMEGA_REPO}"
[[ -f "${OMEGA_CHECKPOINT}" ]] || fail "missing VGGT-Omega checkpoint: ${OMEGA_CHECKPOINT}"
[[ -d "${PI05_PARAMS}" ]] || fail "missing Pi05 params: ${PI05_PARAMS}"
[[ -f "${WA01_CACHE}/metadata.json" ]] || fail "missing WA01 Omega cache metadata"
[[ -f "${WA01_CACHE}/completed.u8.npy" ]] || fail "missing WA01 Omega cache completion bitmap"
[[ ! -e "${TRAIN_RUN}" ]] || fail "WA01 training directory already exists: ${TRAIN_RUN}"

cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export PI05_BASE_PARAMS_PATH="${PI05_PARAMS}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

log "waiting for TS01 step 60000 checkpoint: ${TS01_FINAL_METADATA}"
while [[ ! -f "${TS01_FINAL_METADATA}" ]]; do
  sleep "${WAIT_SECONDS}"
done
log "verified TS01 step 60000 checkpoint"

cache_args=(
  scripts/extract_lerobot_omega_cache.py
  --dataset-root "${WA01_DATASET}"
  --omega-repo "${OMEGA_REPO}"
  --checkpoint "${OMEGA_CHECKPOINT}"
  --output-dir "${WA01_CACHE}"
  --decision-stride-frames 50
)

log "verifying WA01 Omega cache"
"${PYTHON_BIN}" "${cache_args[@]}" --verify-only

log "running WA01 transformed-data smoke test"
"${PYTHON_BIN}" scripts/omega_data_smoke.py \
  --config-name "${TRAIN_CONFIG}" --batch-size 2 --num-workers 2

[[ ! -e "${TRAIN_RUN}" ]] || fail "WA01 training directory appeared before launch: ${TRAIN_RUN}"
sleep 15
wait_for_gpus
log "starting WA01 action-condition training on GPUs 0,1,2,3: global batch 48, 60000 steps"

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
