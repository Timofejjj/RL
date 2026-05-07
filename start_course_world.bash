#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLD="$ROOT/course_robot_world.sdf"
CR_DIR="$ROOT/course_robot"

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/jazzy/setup.bash
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/humble/setup.bash
fi

export GZ_SIM_RESOURCE_PATH="${ROOT}${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"

gz sim -r "$WORLD" &
SIM_PID=$!

cleanup() {
  kill "$SIM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2
ros2 run ros_gz_bridge parameter_bridge --ros-args -p "config_file:=${CR_DIR}/ros_gz_course_robot.yaml"

