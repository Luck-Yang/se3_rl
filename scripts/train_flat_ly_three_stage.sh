#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(git rev-parse --show-toplevel)
cd "$repo_dir"

experiment_dir="$repo_dir/logs/rsl_rl/se3_wheel_leg_flat_ly"
num_envs=${SE3_FLAT_LY_NUM_ENVS:-2048}
base_iterations=${SE3_FLAT_LY_BASE_ITERATIONS:-5000}
stand_iterations=${SE3_FLAT_LY_STAND_ITERATIONS:-2000}
turn_iterations=${SE3_FLAT_LY_TURN_ITERATIONS:-2800}
arc_iterations=${SE3_FLAT_LY_ARC_ITERATIONS:-1500}
pipeline_id=${SE3_FLAT_LY_PIPELINE_ID:-$(date +%Y%m%d_%H%M%S)_$$}

base_run_name="fresh_base_${num_envs}env_${base_iterations}iter_${pipeline_id}"
stand_run_name="fresh_stand_${num_envs}env_${stand_iterations}iter_${pipeline_id}"
turn_run_name="fresh_turn_${num_envs}env_${turn_iterations}iter_${pipeline_id}"
arc_run_name="fresh_arc_${num_envs}env_${arc_iterations}iter_${pipeline_id}"

require_positive_integer() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[flat_ly pipeline] $name 必须是正整数，实际为：$value" >&2
    exit 2
  fi
}

for setting in \
  "SE3_FLAT_LY_NUM_ENVS:$num_envs" \
  "SE3_FLAT_LY_BASE_ITERATIONS:$base_iterations" \
  "SE3_FLAT_LY_STAND_ITERATIONS:$stand_iterations" \
  "SE3_FLAT_LY_TURN_ITERATIONS:$turn_iterations" \
  "SE3_FLAT_LY_ARC_ITERATIONS:$arc_iterations"; do
  require_positive_integer "${setting%%:*}" "${setting#*:}"
done

if [[ ! "$pipeline_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "[flat_ly pipeline] SE3_FLAT_LY_PIPELINE_ID 只能包含字母、数字、下划线和连字符。" >&2
  exit 2
fi

find_runs() {
  local run_name=$1
  if [[ ! -d "$experiment_dir" ]]; then
    return 0
  fi
  find "$experiment_dir" -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" -print
}

assert_run_name_unused() {
  local run_name=$1
  local existing
  existing=$(find_runs "$run_name")
  if [[ -n "$existing" ]]; then
    echo "[flat_ly pipeline] run-name 已存在，拒绝混用旧输出：$run_name" >&2
    echo "$existing" >&2
    exit 1
  fi
}

resolve_stage_output() {
  local run_name=$1
  local -a matches=()
  mapfile -t matches < <(find_runs "$run_name")
  if (( ${#matches[@]} != 1 )); then
    echo "[flat_ly pipeline] 阶段输出必须唯一：$run_name，实际找到 ${#matches[@]} 个。" >&2
    return 1
  fi

  resolved_run=${matches[0]}
  resolved_checkpoint=$(
    find "$resolved_run" -maxdepth 1 -type f -name 'model_[0-9]*.pt' -print \
      | sort -V \
      | tail -1
  )
  if [[ -z "$resolved_checkpoint" ]]; then
    echo "[flat_ly pipeline] 阶段没有生成 checkpoint：$resolved_run" >&2
    return 1
  fi
  if [[ ! "$(basename "$resolved_checkpoint")" =~ ^model_[0-9]+\.pt$ ]]; then
    echo "[flat_ly pipeline] checkpoint 名称不符合预期：$resolved_checkpoint" >&2
    return 1
  fi
}

run_fresh_base() {
  echo "[flat_ly pipeline] 启动随机初始化 base（不会读取任何旧 checkpoint）。"
  SE3_FULL_RESUME=0 uv run se3-train "SE3-WheelLegged-Flat-LY-GRU" \
    --env.scene.num-envs "$num_envs" \
    --agent.resume False \
    --agent.max-iterations "$base_iterations" \
    --agent.run-name "$base_run_name"
}

run_warm_start_stage() {
  local task_id=$1
  local iterations=$2
  local run_name=$3
  local source_run=$4
  local source_checkpoint=$5
  local checkpoint_name
  checkpoint_name=$(basename "$source_checkpoint")

  echo "[flat_ly pipeline] 启动 warm-start 阶段：$task_id"
  echo "[flat_ly pipeline] 只加载 actor/critic：$source_run/$checkpoint_name"
  SE3_FULL_RESUME=0 uv run se3-train "$task_id" \
    --env.scene.num-envs "$num_envs" \
    --agent.resume True \
    --agent.load-run "^$(basename "$source_run")$" \
    --agent.load-checkpoint "^${checkpoint_name%.pt}\\.pt$" \
    --agent.max-iterations "$iterations" \
    --agent.run-name "$run_name"
}

for run_name in "$base_run_name" "$stand_run_name" "$turn_run_name" "$arc_run_name"; do
  assert_run_name_unused "$run_name"
done

echo "[flat_ly pipeline] pipeline_id=$pipeline_id"
run_fresh_base
resolve_stage_output "$base_run_name"
base_run=$resolved_run
base_checkpoint=$resolved_checkpoint

run_warm_start_stage \
  "SE3-WheelLegged-Flat-LY-Stand-GRU" \
  "$stand_iterations" \
  "$stand_run_name" \
  "$base_run" \
  "$base_checkpoint"
resolve_stage_output "$stand_run_name"
stand_run=$resolved_run
stand_checkpoint=$resolved_checkpoint

run_warm_start_stage \
  "SE3-WheelLegged-Flat-LY-Turn-GRU" \
  "$turn_iterations" \
  "$turn_run_name" \
  "$stand_run" \
  "$stand_checkpoint"
resolve_stage_output "$turn_run_name"
turn_run=$resolved_run
turn_checkpoint=$resolved_checkpoint

run_warm_start_stage \
  "SE3-WheelLegged-Flat-LY-Arc-GRU" \
  "$arc_iterations" \
  "$arc_run_name" \
  "$turn_run" \
  "$turn_checkpoint"
resolve_stage_output "$arc_run_name"

echo "[flat_ly pipeline] fresh base + 三个 warm-start 阶段全部完成。"
echo "[flat_ly pipeline] 最终 checkpoint：$resolved_checkpoint"
