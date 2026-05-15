#!/usr/bin/env bash
# Robust RL launcher that works from any current directory.
# Usage:
#   bash /mnt/d/BSU/Course_Work/Course_work_part_2/run_rl_training.bash
# Optional env overrides:
#   COURSE_RL_TIMESTEPS=200000
#   COURSE_RL_MODEL_NAME=ppo_course_robot_200k
#   COURSE_RL_RUN_ID=run_custom_name

set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CR_DIR="$ROOT/course_robot"
RL_SCRIPT="$CR_DIR/rl_train.py"

if [[ ! -f "$RL_SCRIPT" ]]; then
  echo "RL script not found: $RL_SCRIPT" >&2
  exit 1
fi

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/jazzy/setup.bash
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/humble/setup.bash
else
  echo "ROS setup.bash not found in /opt/ros/<distro>/setup.bash" >&2
  exit 1
fi

if [[ -x "$CR_DIR/.venv_wsl/bin/python3" ]]; then
  PYTHON_BIN="$CR_DIR/.venv_wsl/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

COURSE_RL_TIMESTEPS="${COURSE_RL_TIMESTEPS:-100000}"
COURSE_RL_MODEL_NAME="${COURSE_RL_MODEL_NAME:-ppo_course_robot_100k}"
COURSE_RL_MAX_LINEAR_SPEED_MPS="${COURSE_RL_MAX_LINEAR_SPEED_MPS:-0.35}"
COURSE_RL_MAX_ANGULAR_SPEED_RADPS="${COURSE_RL_MAX_ANGULAR_SPEED_RADPS:-0.8}"
COURSE_RL_CONTROL_DT_SEC="${COURSE_RL_CONTROL_DT_SEC:-0.12}"
COURSE_RL_MAX_LINEAR_ACCEL_MPS2="${COURSE_RL_MAX_LINEAR_ACCEL_MPS2:-0.8}"
COURSE_RL_MAX_ANGULAR_ACCEL_RADPS2="${COURSE_RL_MAX_ANGULAR_ACCEL_RADPS2:-2.5}"
COURSE_RL_ANGULAR_DEADBAND="${COURSE_RL_ANGULAR_DEADBAND:-0.08}"
COURSE_RL_RESET_WORLD_ON_EPISODE="${COURSE_RL_RESET_WORLD_ON_EPISODE:-0}"
COURSE_RL_OBSTACLE_RANDOMIZE_EVERY="${COURSE_RL_OBSTACLE_RANDOMIZE_EVERY:-20}"
COURSE_RL_SAVE_DIR="${COURSE_RL_SAVE_DIR:-$CR_DIR/models}"
COURSE_RL_LOG_ROOT="${COURSE_RL_LOG_ROOT:-$CR_DIR/training_logs}"
COURSE_RL_RUN_ID="${COURSE_RL_RUN_ID:-}"

if ! mkdir -p "$COURSE_RL_SAVE_DIR" 2>/dev/null; then
  COURSE_RL_SAVE_DIR="/tmp/course_robot_models"
  mkdir -p "$COURSE_RL_SAVE_DIR"
  echo "WARN: save_dir on /mnt/d is not writable, fallback to $COURSE_RL_SAVE_DIR" >&2
fi

if ! mkdir -p "$COURSE_RL_LOG_ROOT" 2>/dev/null; then
  COURSE_RL_LOG_ROOT="/tmp/course_robot_training_logs"
  mkdir -p "$COURSE_RL_LOG_ROOT"
  echo "WARN: log root on /mnt/d is not writable, fallback to $COURSE_RL_LOG_ROOT" >&2
fi

if ! touch "$COURSE_RL_LOG_ROOT/.write_test" 2>/dev/null; then
  COURSE_RL_LOG_ROOT="/tmp/course_robot_training_logs"
  mkdir -p "$COURSE_RL_LOG_ROOT"
  echo "WARN: cannot write to selected log root, fallback to $COURSE_RL_LOG_ROOT" >&2
else
  rm -f "$COURSE_RL_LOG_ROOT/.write_test" 2>/dev/null || true
fi

cd "$ROOT"
echo "Project root: $ROOT"
echo "Python: $PYTHON_BIN"
echo "Timesteps: $COURSE_RL_TIMESTEPS"
echo "Model name: $COURSE_RL_MODEL_NAME"
echo "Save root: $COURSE_RL_SAVE_DIR"
echo "Log root: $COURSE_RL_LOG_ROOT"
if [[ -n "$COURSE_RL_RUN_ID" ]]; then
  echo "Run id: $COURSE_RL_RUN_ID"
fi
echo "Max linear speed: $COURSE_RL_MAX_LINEAR_SPEED_MPS"
echo "Max angular speed: $COURSE_RL_MAX_ANGULAR_SPEED_RADPS"
echo "Control dt: $COURSE_RL_CONTROL_DT_SEC"
echo "Max linear accel: $COURSE_RL_MAX_LINEAR_ACCEL_MPS2"
echo "Max angular accel: $COURSE_RL_MAX_ANGULAR_ACCEL_RADPS2"
echo "Angular deadband: $COURSE_RL_ANGULAR_DEADBAND"
echo "Reset world on episode: $COURSE_RL_RESET_WORLD_ON_EPISODE"
echo "Obstacle randomize every: $COURSE_RL_OBSTACLE_RANDOMIZE_EVERY"

cmd=(
  "$PYTHON_BIN" "$RL_SCRIPT"
  --total-timesteps "$COURSE_RL_TIMESTEPS"
  --model-name "$COURSE_RL_MODEL_NAME"
  --save-dir "$COURSE_RL_SAVE_DIR"
  --log-dir "$COURSE_RL_LOG_ROOT"
  --max-linear-speed-mps "$COURSE_RL_MAX_LINEAR_SPEED_MPS"
  --max-angular-speed-radps "$COURSE_RL_MAX_ANGULAR_SPEED_RADPS"
  --control-dt-sec "$COURSE_RL_CONTROL_DT_SEC"
  --max-linear-accel-mps2 "$COURSE_RL_MAX_LINEAR_ACCEL_MPS2"
  --max-angular-accel-radps2 "$COURSE_RL_MAX_ANGULAR_ACCEL_RADPS2"
  --angular-deadband "$COURSE_RL_ANGULAR_DEADBAND"
  --obstacle-randomize-every "$COURSE_RL_OBSTACLE_RANDOMIZE_EVERY"
)
if [[ "$COURSE_RL_RESET_WORLD_ON_EPISODE" -eq 1 ]]; then
  cmd+=(--reset-world-on-episode)
else
  cmd+=(--no-reset-world-on-episode)
fi
if [[ -n "$COURSE_RL_RUN_ID" ]]; then
  cmd+=(--run-id "$COURSE_RL_RUN_ID")
fi
cmd+=("$@")
exec "${cmd[@]}"
