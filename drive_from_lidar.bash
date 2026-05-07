#!/usr/bin/env bash
# One-command runner: bridge + lidar_drive controller.
# Usage (WSL/Ubuntu): bash drive_from_lidar.bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CR_DIR="$ROOT/course_robot"
BRIDGE_CFG_DEFAULT="$CR_DIR/ros_gz_course_robot.yaml"
NODE_DEFAULT="$CR_DIR/lidar_drive.py"

BRIDGE_CFG="$BRIDGE_CFG_DEFAULT"
NODE="$NODE_DEFAULT"
ROS_DISTRO="${ROS_DISTRO:-auto}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-15}"
NO_BRIDGE=0
NO_ROS_TF=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash drive_from_lidar.bash [options]

Options:
  --cfg PATH            Path to ros_gz_bridge yaml config
  --node PATH           Path to lidar_drive python node
  --ros-distro NAME     ROS 2 distro to source: jazzy|humble|auto (default: auto)
  --timeout SECONDS     Seconds to wait for bridge to appear (default: 15)
  --no-bridge           Do not start ros_gz_bridge (TF stack still starts — нужен для RViz/лидара)
  --no-ros-tf           Не поднимать мост и TF (только python-узел; мост/TF уже от start_everything и т.п.)
  --dry-run             Print commands instead of running them
  -h, --help            Show this help
EOF
}

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'DRY-RUN: %q' "$1"
    shift
    for a in "$@"; do printf ' %q' "$a"; done
    printf '\n'
    return 0
  fi
  "$@"
}

wait_for_bridge() {
  local timeout_s="$1"
  local start_s now_s elapsed_s
  start_s="$(date +%s)"

  while true; do
    if [[ -n "${BRIDGE_PID:-}" ]] && ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
      echo "ros_gz_bridge exited before it became ready." >&2
      return 1
    fi

    # Best-effort readiness: node appears in graph.
    if ros2 node list 2>/dev/null | grep -q 'parameter_bridge'; then
      return 0
    fi

    now_s="$(date +%s)"
    elapsed_s=$(( now_s - start_s ))
    if (( elapsed_s >= timeout_s )); then
      echo "Timed out waiting for ros_gz_bridge to appear in ROS graph (${timeout_s}s)." >&2
      return 1
    fi
    sleep 0.2
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg) BRIDGE_CFG="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    --ros-distro) ROS_DISTRO="$2"; shift 2 ;;
    --timeout) WAIT_TIMEOUT_S="$2"; shift 2 ;;
    --no-bridge) NO_BRIDGE=1; shift ;;
    --no-ros-tf) NO_ROS_TF=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$ROS_DISTRO" == "auto" ]]; then
  if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    ROS_DISTRO="jazzy"
  elif [[ -f /opt/ros/humble/setup.bash ]]; then
    ROS_DISTRO="humble"
  else
    echo "ROS 2 not found in /opt/ros (jazzy/humble). Set --ros-distro or install ROS 2." >&2
    exit 1
  fi
fi

case "$ROS_DISTRO" in
  jazzy|humble)
    # shellcheck source=/dev/null
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    ;;
  *)
    echo "Unsupported ROS distro: $ROS_DISTRO (expected jazzy|humble|auto)" >&2
    exit 2
    ;;
esac

for need in ros2 python3; do
  if ! command -v "$need" &>/dev/null; then
    echo "Missing command in PATH: $need" >&2
    exit 1
  fi
done

if [[ "$NO_ROS_TF" -eq 0 && "$NO_BRIDGE" -eq 0 ]] && [[ ! -f "$BRIDGE_CFG" ]]; then
  echo "Missing bridge config: $BRIDGE_CFG" >&2
  exit 1
fi
if [[ ! -f "$NODE" ]]; then
  echo "Missing node: $NODE" >&2
  exit 1
fi

cleanup() {
  if [[ "${NO_ROS_TF:-0}" -eq 0 ]] && [[ -f "$CR_DIR/ros_tf_stack.bash" ]]; then
    # shellcheck source=course_robot/ros_tf_stack.bash
    source "$CR_DIR/ros_tf_stack.bash"
    course_robot_stop_ros_tf_stack
  fi
}
trap cleanup EXIT INT TERM

URDF="$CR_DIR/course_robot.urdf"

if [[ "$NO_ROS_TF" -eq 0 ]]; then
  echo "== ros_gz_bridge + TF stack (как в start_everything; RViz увидит world -> ... -> lidar_3d) =="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'DRY-RUN: source %q && SKIP_BRIDGE=%q course_robot_launch_ros_tf_stack\n' "$CR_DIR/ros_tf_stack.bash" "$NO_BRIDGE"
  else
    # shellcheck source=course_robot/ros_tf_stack.bash
    source "$CR_DIR/ros_tf_stack.bash"
    export COURSE_ROBOT_ROS_BRIDGE_CFG="$BRIDGE_CFG"
    if [[ "$NO_BRIDGE" -eq 1 ]]; then
      SKIP_BRIDGE=1
    else
      SKIP_BRIDGE=0
    fi
    course_robot_launch_ros_tf_stack
  fi

  if [[ "$NO_BRIDGE" -eq 0 && "$DRY_RUN" -eq 0 ]]; then
    wait_for_bridge "$WAIT_TIMEOUT_S"
  fi

  if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "== Ожидание первого сообщения одометрии (TF odom->base) =="
    if timeout 25s ros2 topic echo /model/course_robot/odometry --qos-reliability best_effort 2>/dev/null | head -n 5 | grep -q .; then
      echo "Одометрия получена."
    else
      echo "Предупреждение: за 25 с не пришло ни одного /model/course_robot/odometry — TF до base_link может отсутствовать (проверьте Gazebo и мост)." >&2
    fi
  fi
else
  echo "== --no-ros-tf: мост и TF не запускаются (ожидается внешний стек ROS) =="
fi

echo "== lidar_drive (drives forward if no obstacles within 5m) =="
run python3 "$NODE" --ros-args -p use_sim_time:=true

