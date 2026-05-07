#!/usr/bin/env bash
# RViz РєР°Рє РІ diff_drive + РјРѕРґРµР»СЊ course_robot Р±РµР· РІС‹Р±РѕСЂР° .urdf РІ РґРёР°Р»РѕРіРµ (С‚РѕРїРёРє /robot_description).
# Р—Р°РїСѓСЃРє: bash run_rviz_with_robot.bash   (РёР· WSL, РїРѕСЃР»Рµ source /opt/ros/jazzy/setup.bash)
set -eo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URDF="$DIR/course_robot.urdf"

# Р‘РµР· Gazebo: СЂРѕР±РѕС‚ РІ РЅР°С‡Р°Р»Рµ РєРѕРѕСЂРґРёРЅР°С‚. РЎ РїРѕР»РЅС‹Рј СЃС‚РµРєРѕРј СЃРј. start_everything.bash (world в†’ course_robot_odom + РјРѕСЃС‚).
: "${COURSE_ROBOT_RVIZ_WORLD_Z:=0.0}"
# Без Gazebo: корень URDF = course_robot_base_link (как у DiffDrive); TF Prefix в RViz пустой.
ros2 run tf2_ros static_transform_publisher --frame-id world --child-frame-id course_robot_base_link \
  --x 0 --y 0 --z "$COURSE_ROBOT_RVIZ_WORLD_Z" --qx 0 --qy 0 --qz 0 --qw 1 &
STF_PID=$!
python3 "$DIR/emit_robot_description.py" --ros-args -p urdf_path:="$URDF" &
DESC_PID=$!
cleanup() {
  kill "$STF_PID" "$DESC_PID" 2>/dev/null || true
}
trap cleanup EXIT
sleep 0.5
# В конфиге RViz включён sim time; без Gazebo /clock отключаем параметр узла.
rviz2 -d "$DIR/diff_drive_plus_course_robot.rviz" --ros-args -p use_sim_time:=false
