#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(git rev-parse --show-toplevel)
cd "$repo_dir"

experiment_dir="$repo_dir/logs/rsl_rl/se3_wheel_leg_flat_ly"
num_envs=${SE3_FLAT_LY_NUM_ENVS:-2048}
stand_iterations=${SE3_FLAT_LY_STAND_ITERATIONS:-2000}
turn_iterations=${SE3_FLAT_LY_TURN_ITERATIONS:-2800}
arc_iterations=${SE3_FLAT_LY_ARC_ITERATIONS:-1500}

v8_run_suffix="velocity_physics_recovery_yaw_learning_2048env_2000_v8"
stand_run_name="velocity_physics_recovery_stationkeeping_${num_envs}env_${stand_iterations}_v9"
turn_run_name="velocity_physics_recovery_yaw_spin_${num_envs}env_${turn_iterations}_v10"
arc_run_name="velocity_physics_recovery_arc_turn_${num_envs}env_${arc_iterations}_v11"

latest_run() {
  local suffix=$1
  find "$experiment_dir" -mindepth 1 -maxdepth 1 -type d -name "*${suffix}" \
    -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
}

latest_checkpoint() {
  local run_dir=$1
  find "$run_dir" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -1
}

wait_for_v8() {
  local pattern="se3-train.*--agent.run-name ${v8_run_suffix}"
  while pgrep -f "$pattern" >/dev/null; do
    echo "[flat_ly pipeline] V8 仍在训练，30 秒后再次检查。"
    sleep 30
  done
}

run_stage() {
  local task_id=$1
  local iterations=$2
  local run_name=$3
  local source_run=$4
  local source_checkpoint=$5

  echo "[flat_ly pipeline] 启动 $task_id"
  echo "[flat_ly pipeline] 来源：$source_run/$(basename "$source_checkpoint")"
  uv run se3-train "$task_id" \
    --env.scene.num-envs "$num_envs" \
    --agent.resume True \
    --agent.load-run "$(basename "$source_run")" \
    --agent.load-checkpoint "$(basename "$source_checkpoint")" \
    --agent.max-iterations "$iterations" \
    --agent.run-name "$run_name"
}

resolve_stage_output() {
  local run_name=$1
  resolved_run=$(latest_run "$run_name")
  if [[ -z "$resolved_run" ]]; then
    echo "[flat_ly pipeline] 找不到阶段输出：$run_name" >&2
    return 1
  fi
  resolved_checkpoint=$(latest_checkpoint "$resolved_run")
  if [[ -z "$resolved_checkpoint" ]]; then
    echo "[flat_ly pipeline] 阶段没有生成 checkpoint：$resolved_run" >&2
    return 1
  fi
}

wait_for_v8
base_run=$(latest_run "$v8_run_suffix")
if [[ -z "$base_run" ]]; then
  echo "[flat_ly pipeline] 找不到 V8 训练目录：*${v8_run_suffix}" >&2
  exit 1
fi
base_checkpoint=$(latest_checkpoint "$base_run")
if [[ -z "$base_checkpoint" ]]; then
  echo "[flat_ly pipeline] V8 没有可用 checkpoint：$base_run" >&2
  exit 1
fi

run_stage \
  "SE3-WheelLegged-Flat-LY-Stand-GRU" \
  "$stand_iterations" \
  "$stand_run_name" \
  "$base_run" \
  "$base_checkpoint"
resolve_stage_output "$stand_run_name"
stand_run=$resolved_run
stand_checkpoint=$resolved_checkpoint

run_stage \
  "SE3-WheelLegged-Flat-LY-Turn-GRU" \
  "$turn_iterations" \
  "$turn_run_name" \
  "$stand_run" \
  "$stand_checkpoint"
resolve_stage_output "$turn_run_name"
turn_run=$resolved_run
turn_checkpoint=$resolved_checkpoint

run_stage \
  "SE3-WheelLegged-Flat-LY-Arc-GRU" \
  "$arc_iterations" \
  "$arc_run_name" \
  "$turn_run" \
  "$turn_checkpoint"
resolve_stage_output "$arc_run_name"

echo "[flat_ly pipeline] 三阶段训练全部完成。"
echo "[flat_ly pipeline] 最终 checkpoint：$resolved_checkpoint"
