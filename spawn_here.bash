#!/usr/bin/env bash
# Один вызов без ROS. Не передаём пустой второй аргумент — иначе Z в spawn ломался.
#   bash …/spawn_here.bash
#   bash …/spawn_here.bash diff_drive 1.2
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GZ_SIM_RESOURCE_PATH="${ROOT}${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"
W="${1:-diff_drive}"
if [[ $# -ge 2 ]]; then
  exec bash "$ROOT/spawn_course_robot.bash" "$W" "$2"
fi
exec bash "$ROOT/spawn_course_robot.bash" "$W"
