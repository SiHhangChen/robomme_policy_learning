# MME-VLA Omega Suite

This suite is independent of the legacy `mme_vla_suite` pkl pipeline. It
trains the Pi05 action expert on local MemBench LeRobot v3 datasets while using
a frozen VGGT-Omega cache as strict-past conditioning.

The input contract is fixed to three synchronized views:

```text
agentview_left, agentview_right, eye_in_hand
```

Omega cache rows contain four fused layers (`4, 11, 17, 23`), the camera token
and 16 register tokens per view. The shared `8192 -> 1024` projection is
followed by action-hidden cross-attention and `MemoryRMSNorm` FFN modulation.

## Data

The WA05 configuration reads these existing files directly:

```text
dataset: /data1/shared_workspace/chensihang/dataset/membench/wa05_200seeds_v061
cache:   /data1/shared_workspace/chensihang/dataset/baseline/vggt-omega/wa05_vggt_cache
```

The dataset reader uses `data/chunk-*/file-*.parquet`, the three video trees
under `videos/`, and task text from `meta/tasks.parquet`. It does not require
Hugging Face network access or `tasks.jsonl`.

The model input follows the verified MemBench baseline contract:

- `observation.state` is 37-dimensional; the first 30 dimensions are used and
  padded to Pi05's 32-dimensional state input.
- `action` is the native 13-dimensional `pandaomron_hybrid13.v2` command and is
  padded to 32 for the model. Outputs are cropped back to 13 dimensions.
- WA05 uses a 20-frame action horizon. The comparable TS01 experiment uses a
  50-frame action horizon and LoRA on both Gemma experts.

Before initializing the full Pi05 model, validate the local parquet/video/cache
contract (including spawned workers) with:

```bash
PYTHONPATH=src:/data1/workspace/chensihang/membench/policy/openpi/src \
  /data1/workspace/chensihang/membench/policy/openpi/.venv/bin/python \
  scripts/omega_data_smoke.py --batch-size 2 --num-workers 2
```

For frame `k`, the memory slots contain cache decisions at
`k - 8*50, ..., k - 50`. The current frame is never inserted into its own
memory. Missing history at an episode start is zero-filled and masked, and
episode boundaries are never crossed.

### TS01 cache

The TS01 configuration uses:

```text
dataset: /data1/shared_workspace/chensihang/dataset/membench/ts01_200seeds_v061
cache:   /data1/shared_workspace/chensihang/dataset/baseline/vggt-omega/ts01_vggt_cache
```

Build the cache directly from the LeRobot parquet and MP4 files. Initialization
creates the `4520 x 3 x 4 x 17 x 2048` float16 memmap and exact
`[episode, frame, raw_index]` decision keys without loading VGGT-Omega:

```bash
OMEGA_PYTHON=/data1/workspace/chensihang/membench/policy/openpi/.venv/bin/python
OMEGA_PYTHONPATH=src:/data1/workspace/chensihang/membench/policy/openpi/src
TS01_DATASET=/data1/shared_workspace/chensihang/dataset/membench/ts01_200seeds_v061
TS01_CACHE=/data1/shared_workspace/chensihang/dataset/baseline/vggt-omega/ts01_vggt_cache
OMEGA_REPO=/data1/workspace/chensihang/pretrained/vggt-omega
OMEGA_CHECKPOINT=/data1/shared_workspace/chensihang/pretrained/vggt_omega_1b_256_text.pt

PYTHONPATH="$OMEGA_PYTHONPATH" "$OMEGA_PYTHON" scripts/extract_lerobot_omega_cache.py \
  --dataset-root "$TS01_DATASET" \
  --omega-repo "$OMEGA_REPO" \
  --checkpoint "$OMEGA_CHECKPOINT" \
  --output-dir "$TS01_CACHE" \
  --decision-stride-frames 50 \
  --initialize-only
```

Then run three non-overlapping resumable shards on GPUs 4, 5, and 6:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$OMEGA_PYTHONPATH" "$OMEGA_PYTHON" \
  scripts/extract_lerobot_omega_cache.py \
  --dataset-root "$TS01_DATASET" --omega-repo "$OMEGA_REPO" \
  --checkpoint "$OMEGA_CHECKPOINT" --output-dir "$TS01_CACHE" \
  --decision-stride-frames 50 --num-shards 3 --shard-index 0 \
  --device cuda:0 --batch-size 1 --resume > /tmp/ts01_omega_cache_gpu4.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 PYTHONPATH="$OMEGA_PYTHONPATH" "$OMEGA_PYTHON" \
  scripts/extract_lerobot_omega_cache.py \
  --dataset-root "$TS01_DATASET" --omega-repo "$OMEGA_REPO" \
  --checkpoint "$OMEGA_CHECKPOINT" --output-dir "$TS01_CACHE" \
  --decision-stride-frames 50 --num-shards 3 --shard-index 1 \
  --device cuda:0 --batch-size 1 --resume > /tmp/ts01_omega_cache_gpu5.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$OMEGA_PYTHONPATH" "$OMEGA_PYTHON" \
  scripts/extract_lerobot_omega_cache.py \
  --dataset-root "$TS01_DATASET" --omega-repo "$OMEGA_REPO" \
  --checkpoint "$OMEGA_CHECKPOINT" --output-dir "$TS01_CACHE" \
  --decision-stride-frames 50 --num-shards 3 --shard-index 2 \
  --device cuda:0 --batch-size 1 --resume > /tmp/ts01_omega_cache_gpu6.log 2>&1 &

wait
```

Interrupted shards can be launched again with the same commands; completed rows
are skipped. Verify all rows before training:

```bash
PYTHONPATH="$OMEGA_PYTHONPATH" "$OMEGA_PYTHON" scripts/extract_lerobot_omega_cache.py \
  --dataset-root "$TS01_DATASET" --omega-repo "$OMEGA_REPO" \
  --checkpoint "$OMEGA_CHECKPOINT" --output-dir "$TS01_CACHE" \
  --decision-stride-frames 50 --verify-only
```

After verification, check one transformed TS01 batch before model startup:

```bash
PYTHONPATH="$OMEGA_PYTHONPATH" "$OMEGA_PYTHON" scripts/omega_data_smoke.py \
  --config-name mme_vla_omega_ts01_action_modulation \
  --batch-size 2 --num-workers 2
```

## Training

### WA01 cache and action-condition variants

The WA01 LeRobot dataset is compatible with the same three-view Omega cache
contract. It contains 400,366 frames across 200 episodes. With the configured
50-frame decision stride there are 8,102 cache rows, requiring approximately
6.3 GiB for the float16 token bank.

Initialize the cache on the data disk. This command only creates the memmaps;
it does not load the Omega model:

```bash
OMEGA_PYTHON=/data1/workspace/chensihang/membench/policy/openpi/.venv/bin/python
OMEGA_PYTHONPATH=src:/data1/workspace/chensihang/membench/policy/openpi/src
WA01_DATASET=/data1/shared_workspace/chensihang/dataset/membench/wa01_200seeds_v061
WA01_CACHE=/data1/shared_workspace/chensihang/dataset/baseline/vggt-omega/wa01_vggt_cache
OMEGA_REPO=/data1/workspace/chensihang/pretrained/vggt-omega
OMEGA_CHECKPOINT=/data1/shared_workspace/chensihang/pretrained/vggt_omega_1b_256_text.pt

PYTHONPATH="$OMEGA_PYTHONPATH" "$OMEGA_PYTHON" scripts/extract_lerobot_omega_cache.py \
  --dataset-root "$WA01_DATASET" --omega-repo "$OMEGA_REPO" \
  --checkpoint "$OMEGA_CHECKPOINT" --output-dir "$WA01_CACHE" \
  --decision-stride-frames 50 --initialize-only
```

Fill the initialized cache with resumable GPU shards. Each shard owns a
disjoint set of decision rows, so interrupted workers can be restarted with
the same command:

```bash
for shard in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$shard" PYTHONPATH="$OMEGA_PYTHONPATH" "$OMEGA_PYTHON" \
    scripts/extract_lerobot_omega_cache.py \
    --dataset-root "$WA01_DATASET" --omega-repo "$OMEGA_REPO" \
    --checkpoint "$OMEGA_CHECKPOINT" --output-dir "$WA01_CACHE" \
    --decision-stride-frames 50 --num-shards 3 --shard-index "$shard" \
    --device cuda:0 --batch-size 1 --resume \
    > "$WA01_CACHE/shard-${shard}.log" 2>&1 &
done
wait
```

Verify every row before training, then inspect one transformed batch:

```bash
PYTHONPATH="$OMEGA_PYTHONPATH" "$OMEGA_PYTHON" scripts/extract_lerobot_omega_cache.py \
  --dataset-root "$WA01_DATASET" --omega-repo "$OMEGA_REPO" \
  --checkpoint "$OMEGA_CHECKPOINT" --output-dir "$WA01_CACHE" \
  --decision-stride-frames 50 --verify-only

PYTHONPATH="$OMEGA_PYTHONPATH" "$OMEGA_PYTHON" scripts/omega_data_smoke.py \
  --config-name mme_vla_omega_wa01_action_modulation \
  --batch-size 2 --num-workers 2
```

For the requested condition-injection experiment, use
`mme_vla_omega_wa01_action_condition`. It masked-pools the encoded strict-past
Omega tokens, projects the result to 1024 dimensions, and adds it to the Pi05
action expert's AdaRMS timestep condition. The connector is zero-initialized,
so loading `pi05_base` starts with the same behavior as the base policy. This
variant does not put Omega tokens in the VLM prefix and does not use per-layer
memory cross-attention.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
PYTHONPATH="$OMEGA_PYTHONPATH" "$OMEGA_PYTHON" \
  scripts/train_mme_vla_omega.py mme_vla_omega_wa01_action_condition \
  --exp-name wa01_vggt_omega_action_condition_h8_bs48_60k_g012 \
  --no-wandb-enabled
```

For the stronger per-layer cross-attention alternative, use
`mme_vla_omega_wa01_action_modulation` with the same cache and training
arguments. It is kept as a direct comparison rather than silently changing
the requested condition-injection experiment.

From this repository, run a short smoke run first:

```bash
PYTHONPATH=src:/data1/workspace/chensihang/membench/policy/openpi/src \
  /data1/workspace/chensihang/membench/policy/openpi/.venv/bin/python \
  scripts/train_mme_vla_omega.py \
  mme_vla_omega_wa05_action_modulation \
  --exp-name omega_smoke \
  --num-train-steps 10 \
  --batch-size 2 \
  --num-workers 0 \
  --no-wandb-enabled
```

Override the locations without editing the config:

```bash
OMEGA_WA05_DATASET_PATH=/path/to/wa05_200seeds_v061 \
OMEGA_WA05_CACHE_PATH=/path/to/wa05_vggt_cache \
  PYTHONPATH=src:/data1/workspace/chensihang/membench/policy/openpi/src \
  /data1/workspace/chensihang/membench/policy/openpi/.venv/bin/python \
  scripts/train_mme_vla_omega.py mme_vla_omega_wa05_action_modulation
```

The base Pi05 checkpoint defaults to
`/data1/shared_workspace/tangzhipeng/ckpts/openpi/openpi_assets/pi05_base/params`
and can be overridden with `PI05_BASE_PARAMS_PATH`. The Omega projection,
cross-attention and modulation parameters are newly initialized; the SigLIP
tower remains frozen.

### TS01 training

`mme_vla_omega_ts01_action_modulation` fixes the comparable experiment to
H8/stride50, three views, layers 4/11/17/23, a 50-frame action horizon, dual
Gemma LoRA, no EMA, global batch 48, 60,000 steps, and three FSDP devices. It
also points at the shared Pi05 base params path, so no command-line checkpoint
override is required:

```bash
CUDA_VISIBLE_DEVICES=4,5,6 \
PYTHONPATH=src:/data1/workspace/chensihang/membench/policy/openpi/src \
/data1/workspace/chensihang/membench/policy/openpi/.venv/bin/python \
scripts/train_mme_vla_omega.py mme_vla_omega_ts01_action_modulation \
  --exp-name ts01_vggt_omega_h8_bs48_60k_g456 \
  --no-wandb-enabled
```

For a run that already has a complete split-state checkpoint, use the same
command with `--resume`. If a failed first launch only created an empty run
directory, use `--overwrite` once instead.

### Resume training

Each newly saved Omega checkpoint splits the inference parameters from the
optimizer-bearing `TrainState`. Restoring the two items reconstructs model
parameters, optimizer state, EMA parameters when enabled, and the exact step
without writing the large inference parameters twice. To start a run in an
existing empty checkpoint directory, pass `--overwrite` once. After at least
one complete checkpoint has been written, use `--resume` to restore the latest
checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4,5,6 \
PYTHONPATH=src:/data1/workspace/chensihang/membench/policy/openpi/src \
/data1/workspace/chensihang/membench/policy/openpi/.venv/bin/python \
scripts/train_mme_vla_omega.py mme_vla_omega_wa05_action_modulation \
  --exp-name wa05_vggt_omega_h8_bs48_60k_g456 \
  --dataset-path /data1/shared_workspace/chensihang/dataset/membench/wa05_200seeds_v061 \
  --omega-cache-path /data1/shared_workspace/chensihang/dataset/baseline/vggt-omega/wa05_vggt_cache \
  --batch-size 48 --num-train-steps 60000 --fsdp-devices 3 --num-workers 4 \
  --no-wandb-enabled --resume \
  --weight-loader.params-path /data1/shared_workspace/tangzhipeng/ckpts/openpi/openpi_assets/pi05_base/params
```

`--resum-ckpt-id N` can be added with `--resume` to restore a specific saved
step instead of the latest one. On resume, the data iterator skips the batches
already represented by the restored step so shuffled sample order stays
aligned with an uninterrupted run.

## Online service

Serving still requires a trained checkpoint and, when encoding Omega online,
`OMEGA_REPO`, `OMEGA_CHECKPOINT`, and `OMEGA_DEVICE`. Every request must carry
all three synchronized camera images. The online policy appends the current
Omega scene to history only after its action is sampled, matching offline
strict-past semantics.
