#!/usr/bin/env bash
# Проверка пунктов «одометрия + цепочка TF до лидара» (после запуска Gazebo и ROS-стека).
# Использование (из WSL, с тем же ROS_DISTRO, что и симуляция):
#   source /opt/ros/jazzy/setup.bash
#   bash course_robot/verify_course_robot_stack.bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CR_DIR="$ROOT/course_robot"

if ! command -v ros2 &>/dev/null; then
  echo "Нет ros2 в PATH — сначала: source /opt/ros/<distro>/setup.bash" >&2
  exit 1
fi

echo "== 1) Паблишеры /model/course_robot/odometry =="
if ros2 topic info /model/course_robot/odometry 2>/dev/null | grep -q "Publisher count"; then
  ros2 topic info /model/course_robot/odometry
else
  echo "Топик недоступен или нет паблишеров." >&2
fi

echo ""
echo "== 2) Первое сообщение одометрии (до 20 с) =="
if timeout 20s ros2 topic echo /model/course_robot/odometry --qos-reliability best_effort 2>/dev/null | head -n 8 | grep -q .; then
  echo "OK: одометрия приходит."
else
  echo "FAIL: одометрия нет за 20 с." >&2
fi

echo ""
echo "== 3) TF world -> course_robot/lidar_link/lidar_3d (tf2_echo, до 12 с, use_sim_time) =="
if timeout 12s ros2 run tf2_ros tf2_echo world course_robot/lidar_link/lidar_3d --ros-args -p use_sim_time:=true 2>&1 | head -n 15 | grep -qE 'At time|Translation|Rotation'; then
  echo "OK: цепочка TF до лидара найдена."
  exit 0
fi
echo "FAIL: tf2_echo не получил преобразование (проверьте ros_tf_stack / clock / RViz Fixed Frame=world)." >&2
echo "Подсказка: ros2 run tf2_tools view_frames  (нужен пакет tf2_tools)" >&2
exit 1
