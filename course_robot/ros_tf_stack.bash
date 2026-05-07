#!/usr/bin/env bash
# Общий стек: ros_gz_bridge + статические TF (в т.ч. лидара) + odom_to_tf + robot_description + robot_state_publisher.
# Использование: после source /opt/ros/*/setup.bash выполнить:
#   source "$CR_DIR/ros_tf_stack.bash"
#   course_robot_launch_ros_tf_stack
# Переменные окружения:
#   ROOT, CR_DIR, URDF — обязательны (пути к корню репозитория, course_robot/, course_robot.urdf).
#   SKIP_BRIDGE=1 — не запускать ros_gz_bridge (уже запущен снаружи).
#   COURSE_ROBOT_ROS_BRIDGE_CFG — yaml моста (по умолчанию $CR_DIR/ros_gz_course_robot.yaml).
#   COURSE_ROBOT_SPAWN_X, COURSE_ROBOT_SPAWN_Y — смещение world→odom (как в start_everything.bash).

course_robot_launch_ros_tf_stack() {
  if [[ -z "${CR_DIR:-}" || -z "${URDF:-}" ]]; then
    echo "ros_tf_stack.bash: задайте CR_DIR и URDF перед вызовом course_robot_launch_ros_tf_stack" >&2
    return 1
  fi
  local BRIDGE_CFG="${COURSE_ROBOT_ROS_BRIDGE_CFG:-$CR_DIR/ros_gz_course_robot.yaml}"
  export COURSE_ROBOT_SPAWN_X="${COURSE_ROBOT_SPAWN_X:-0.0}"
  export COURSE_ROBOT_SPAWN_Y="${COURSE_ROBOT_SPAWN_Y:-0.0}"

  if [[ "${SKIP_BRIDGE:-0}" -ne 1 ]]; then
    if [[ ! -f "$BRIDGE_CFG" ]]; then
      echo "ros_tf_stack.bash: нет файла моста: $BRIDGE_CFG" >&2
      return 1
    fi
    ros2 run ros_gz_bridge parameter_bridge --ros-args -p use_sim_time:=true -p "config_file:=${BRIDGE_CFG}" &
    BRIDGE_PID=$!
  else
    BRIDGE_PID=""
  fi

  STF_ROS_ARGS=(--ros-args -p use_sim_time:=true --)
  ros2 run tf2_ros static_transform_publisher "${STF_ROS_ARGS[@]}" \
    --frame-id world --child-frame-id course_robot_odom \
    --x "$COURSE_ROBOT_SPAWN_X" --y "$COURSE_ROBOT_SPAWN_Y" --z 0 \
    --qx 0 --qy 0 --qz 0 --qw 1 &
  STF_PID=$!

  ros2 run tf2_ros static_transform_publisher "${STF_ROS_ARGS[@]}" \
    --frame-id world --child-frame-id course_world \
    --x 0 --y 0 --z 0 --qx 0 --qy 0 --qz 0 --qw 1 &
  STF_WORLD_PID=$!

  ros2 run tf2_ros static_transform_publisher "${STF_ROS_ARGS[@]}" \
    --frame-id course_robot_base_link --child-frame-id course_robot/course_robot_base_link \
    --x 0 --y 0 --z 0 --qx 0 --qy 0 --qz 0 --qw 1 &
  STF_MERGE_PID=$!

  # robot_state_publisher now publishes course_robot_base_link -> course_robot/lidar_link from URDF
  # STF_LIDAR_BASE is no longer needed.

  ros2 run tf2_ros static_transform_publisher "${STF_ROS_ARGS[@]}" \
    --frame-id course_robot/lidar_link --child-frame-id course_robot/lidar_link/lidar_3d \
    --x 0 --y 0 --z 0 --qx 0 --qy 0 --qz 0 --qw 1 &
  STF_LIDAR_PID=$!

  python3 "$CR_DIR/emit_robot_description.py" --ros-args -p use_sim_time:=true -p urdf_path:="$URDF" &
  DESC_PID=$!

  python3 "$CR_DIR/odom_to_tf.py" --ros-args -p use_sim_time:=true &
  ODOM_TF_PID=$!

  if ros2 pkg prefix robot_state_publisher &>/dev/null; then
    ros2 run robot_state_publisher robot_state_publisher "$URDF" --ros-args \
      -p use_sim_time:=true >/tmp/robot_state_publisher_course_robot_${USER:-user}_$$.log 2>&1 &
    RSP_PID=$!
  else
    RSP_PID=""
    echo "ros_tf_stack: пакет robot_state_publisher не найден — TF только из static + odom_to_tf." >&2
  fi
}

course_robot_stop_ros_tf_stack() {
  [[ -n "${BRIDGE_PID:-}" ]] && kill "$BRIDGE_PID" 2>/dev/null || true
  [[ -n "${STF_PID:-}" ]] && kill "$STF_PID" 2>/dev/null || true
  [[ -n "${STF_WORLD_PID:-}" ]] && kill "$STF_WORLD_PID" 2>/dev/null || true
  [[ -n "${STF_MERGE_PID:-}" ]] && kill "$STF_MERGE_PID" 2>/dev/null || true
  [[ -n "${STF_LIDAR_BASE_PID:-}" ]] && kill "$STF_LIDAR_BASE_PID" 2>/dev/null || true
  [[ -n "${STF_LIDAR_PID:-}" ]] && kill "$STF_LIDAR_PID" 2>/dev/null || true
  [[ -n "${DESC_PID:-}" ]] && kill "$DESC_PID" 2>/dev/null || true
  [[ -n "${ODOM_TF_PID:-}" ]] && kill "$ODOM_TF_PID" 2>/dev/null || true
  [[ -n "${RSP_PID:-}" ]] && kill "$RSP_PID" 2>/dev/null || true
}
