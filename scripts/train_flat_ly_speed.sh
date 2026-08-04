#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(git rev-parse --show-toplevel)
cd "$repo_dir"

if (( $# != 1 )); then
  echo "用法：bash scripts/train_flat_ly_speed.sh <model_*.pt>" >&2
  exit 2
fi

if [[ ! -f "$1" ]]; then
  echo "[flat_ly speed] checkpoint 不存在：$1" >&2
  exit 1
fi
source_checkpoint=$(realpath "$1")
source_run=$(dirname "$source_checkpoint")
source_run_name=$(basename "$source_run")
checkpoint_name=$(basename "$source_checkpoint")
experiment_dir=$(realpath "$repo_dir/logs/rsl_rl/se3_wheel_leg_flat_ly")
if [[ ! "$checkpoint_name" =~ ^model_[0-9]+\.pt$ ]]; then
  echo "[flat_ly speed] checkpoint 名称必须符合 model_<迭代号>.pt：$checkpoint_name" >&2
  exit 2
fi
if [[ "$source_run" != "$experiment_dir"/* ]]; then
  echo "[flat_ly speed] checkpoint 必须位于当前 flat_ly 实验目录：$experiment_dir" >&2
  exit 2
fi

num_envs=${SE3_FLAT_LY_SPEED_NUM_ENVS:-4096}
iterations=${SE3_FLAT_LY_SPEED_ITERATIONS:-3700}
run_suffix=${SE3_FLAT_LY_SPEED_RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)_$$}
run_name="speed_bidirectional_${num_envs}env_${iterations}iter_${run_suffix}"

for setting in "SE3_FLAT_LY_SPEED_NUM_ENVS:$num_envs" "SE3_FLAT_LY_SPEED_ITERATIONS:$iterations"; do
  name=${setting%%:*}
  value=${setting#*:}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[flat_ly speed] $name 必须是正整数，实际为：$value" >&2
    exit 2
  fi
done
if [[ ! "$run_suffix" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "[flat_ly speed] SE3_FLAT_LY_SPEED_RUN_SUFFIX 只能包含字母、数字、下划线和连字符。" >&2
  exit 2
fi
existing_run=$(find "$experiment_dir" -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" -print)
if [[ -n "$existing_run" ]]; then
  echo "[flat_ly speed] run-name 已存在，拒绝覆盖：$run_name" >&2
  exit 1
fi

estimated_hours=$(awk -v iterations="$iterations" 'BEGIN { printf "%.2f", iterations * 7.7 / 3600.0 }')
echo "[flat_ly speed] 来源：$source_run_name/$checkpoint_name"
echo "[flat_ly speed] 环境数：$num_envs"
echo "[flat_ly speed] 训练轮数：$iterations（按 7.7 s/轮估算约 ${estimated_hours} h）"
echo "[flat_ly speed] 输出名称：$run_name"

# warm-start 只加载 actor/critic；新阶段从 0.2 m/s 重新验证，按双向成绩逐档扩到 2.0 m/s。
SE3_FULL_RESUME=0 uv run se3-train "SE3-WheelLegged-Flat-LY-Speed-GRU" \
  --env.scene.num-envs "$num_envs" \
  --agent.resume True \
  --agent.load-run "^${source_run_name}$" \
  --agent.load-checkpoint "^${checkpoint_name%.pt}\\.pt$" \
  --agent.max-iterations "$iterations" \
  --agent.run-name "$run_name"
