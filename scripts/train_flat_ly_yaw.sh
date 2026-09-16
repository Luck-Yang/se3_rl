#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(git rev-parse --show-toplevel)
cd "$repo_dir"

if (( $# != 1 )); then
  echo "用法：bash scripts/train_flat_ly_yaw.sh <model_*.pt>" >&2
  exit 2
fi

if [[ ! -f "$1" ]]; then
  echo "[flat_ly yaw] checkpoint 不存在：$1" >&2
  exit 1
fi
source_checkpoint=$(realpath "$1")
source_run=$(dirname "$source_checkpoint")
source_run_name=$(basename "$source_run")
checkpoint_name=$(basename "$source_checkpoint")
experiment_dir=$(realpath "$repo_dir/logs/rsl_rl/se3_wheel_leg_flat_ly")
if [[ ! "$checkpoint_name" =~ ^model_[0-9]+\.pt$ ]]; then
  echo "[flat_ly yaw] checkpoint 名称必须符合 model_<迭代号>.pt：$checkpoint_name" >&2
  exit 2
fi
if [[ "$source_run" != "$experiment_dir"/* ]]; then
  echo "[flat_ly yaw] checkpoint 必须位于当前 flat_ly 实验目录：$experiment_dir" >&2
  exit 2
fi

num_envs=${SE3_FLAT_LY_YAW_NUM_ENVS:-4096}
iterations=${SE3_FLAT_LY_YAW_ITERATIONS:-3000}
run_suffix=${SE3_FLAT_LY_YAW_RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)_$$}
run_name="yaw_bidirectional_${num_envs}env_${iterations}iter_${run_suffix}"

for setting in "SE3_FLAT_LY_YAW_NUM_ENVS:$num_envs" "SE3_FLAT_LY_YAW_ITERATIONS:$iterations"; do
  name=${setting%%:*}
  value=${setting#*:}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[flat_ly yaw] $name 必须是正整数，实际为：$value" >&2
    exit 2
  fi
done
if [[ ! "$run_suffix" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "[flat_ly yaw] SE3_FLAT_LY_YAW_RUN_SUFFIX 只能包含字母、数字、下划线和连字符。" >&2
  exit 2
fi
existing_run=$(find "$experiment_dir" -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" -print)
if [[ -n "$existing_run" ]]; then
  echo "[flat_ly yaw] run-name 已存在，拒绝覆盖：$run_name" >&2
  exit 1
fi

estimated_hours=$(awk -v iterations="$iterations" 'BEGIN { printf "%.2f", iterations * 7.7 / 3600.0 }')
echo "[flat_ly yaw] 来源：$source_run_name/$checkpoint_name"
echo "[flat_ly yaw] 环境数：$num_envs"
echo "[flat_ly yaw] 训练轮数：$iterations（按 7.7 s/轮估算约 ${estimated_hours} h）"
echo "[flat_ly yaw] 课程：双向原地转向 ±0.3 -> ±6.0 rad/s，带静站与直行护栏"
echo "[flat_ly yaw] 输出名称：$run_name"

# warm-start 只加载 actor/critic；yaw 第一档不与 ±2 m/s 线速度组合。
SE3_FULL_RESUME=0 uv run se3-train "SE3-WheelLegged-Flat-LY-Turn-GRU" \
  --env.scene.num-envs "$num_envs" \
  --agent.resume True \
  --agent.load-run "^${source_run_name}$" \
  --agent.load-checkpoint "^${checkpoint_name%.pt}\\.pt$" \
  --agent.max-iterations "$iterations" \
  --agent.run-name "$run_name"
