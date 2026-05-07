#!/usr/bin/env bash
# Spawn RL training obstacles + arena walls into an already running gz sim world.
# Usage:
#   bash spawn_rl_obstacles.bash [world_name]
#
# Defaults to world "diff_drive" (same as ros_gz_sim_demos diff_drive.launch.py).
set -euo pipefail

WORLD="${1:-diff_drive}"

spawn_model() {
  local model_name="$1"
  local pose="$2" # "x y z roll pitch yaw"
  local sdf_content="$3"

  local tmp_sdf="/tmp/${model_name}_$$.sdf"
  printf '%s\n' "$sdf_content" > "$tmp_sdf"

  local x y z rr pp yy
  read -r x y z rr pp yy <<<"$pose"

  local req
  req="sdf_filename: \"${tmp_sdf}\", name: \"${model_name}\", allow_renaming: true, pose: {position: {x: ${x}, y: ${y}, z: ${z}}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}"

  if ! gz service -s "/world/${WORLD}/create" \
    --reqtype gz.msgs.EntityFactory \
    --reptype gz.msgs.Boolean \
    --timeout 10000 \
    --req "$req"; then
    echo "Failed to spawn model '${model_name}' into world '${WORLD}'." >&2
    echo "Tip: check available worlds/services via: gz service --list | grep /world" >&2
    return 1
  fi
}

echo "== Spawning RL arena + obstacles into world: ${WORLD} =="

# One model containing the 4 walls (as separate collisions/visuals).
spawn_model "rl_arena_walls" "0 0 0 0 0 0" "$(cat <<'EOF'
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="rl_arena_walls">
    <static>true</static>
    <link name="walls">
      <pose>0 0 0.25 0 0 0</pose>

      <collision name="north_collision">
        <pose>0 5.05 0 0 0 0</pose>
        <geometry><box><size>10.2 0.1 0.5</size></box></geometry>
      </collision>
      <visual name="north_visual">
        <pose>0 5.05 0 0 0 0</pose>
        <geometry><box><size>10.2 0.1 0.5</size></box></geometry>
        <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material>
      </visual>

      <collision name="south_collision">
        <pose>0 -5.05 0 0 0 0</pose>
        <geometry><box><size>10.2 0.1 0.5</size></box></geometry>
      </collision>
      <visual name="south_visual">
        <pose>0 -5.05 0 0 0 0</pose>
        <geometry><box><size>10.2 0.1 0.5</size></box></geometry>
        <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material>
      </visual>

      <collision name="east_collision">
        <pose>5.05 0 0 0 0 0</pose>
        <geometry><box><size>0.1 10.2 0.5</size></box></geometry>
      </collision>
      <visual name="east_visual">
        <pose>5.05 0 0 0 0 0</pose>
        <geometry><box><size>0.1 10.2 0.5</size></box></geometry>
        <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material>
      </visual>

      <collision name="west_collision">
        <pose>-5.05 0 0 0 0 0</pose>
        <geometry><box><size>0.1 10.2 0.5</size></box></geometry>
      </collision>
      <visual name="west_visual">
        <pose>-5.05 0 0 0 0 0</pose>
        <geometry><box><size>0.1 10.2 0.5</size></box></geometry>
        <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
EOF
)"

# Individual obstacles (simple primitives). Z pose is the center of the object.
spawn_model "rl_obstacle_box_1" "1.5 1.0 0.25 0 0 0" "$(cat <<'EOF'
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="rl_obstacle_box_1">
    <static>true</static>
    <link name="link">
      <collision name="collision"><geometry><box><size>0.6 0.6 0.5</size></box></geometry></collision>
      <visual name="visual">
        <geometry><box><size>0.6 0.6 0.5</size></box></geometry>
        <material><ambient>0.8 0.3 0.3 1</ambient><diffuse>0.8 0.3 0.3 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
EOF
)"

spawn_model "rl_obstacle_box_2" "-2.0 0.5 0.20 0 0 0" "$(cat <<'EOF'
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="rl_obstacle_box_2">
    <static>true</static>
    <link name="link">
      <collision name="collision"><geometry><box><size>1.0 0.4 0.4</size></box></geometry></collision>
      <visual name="visual">
        <geometry><box><size>1.0 0.4 0.4</size></box></geometry>
        <material><ambient>0.3 0.6 0.9 1</ambient><diffuse>0.3 0.6 0.9 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
EOF
)"

spawn_model "rl_obstacle_box_3" "0.0 -2.0 0.15 0 0 0" "$(cat <<'EOF'
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="rl_obstacle_box_3">
    <static>true</static>
    <link name="link">
      <collision name="collision"><geometry><box><size>0.4 1.4 0.3</size></box></geometry></collision>
      <visual name="visual">
        <geometry><box><size>0.4 1.4 0.3</size></box></geometry>
        <material><ambient>0.4 0.9 0.4 1</ambient><diffuse>0.4 0.9 0.4 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
EOF
)"

spawn_model "rl_obstacle_cyl_1" "2.2 -1.5 0.25 0 0 0" "$(cat <<'EOF'
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="rl_obstacle_cyl_1">
    <static>true</static>
    <link name="link">
      <collision name="collision"><geometry><cylinder><radius>0.25</radius><length>0.5</length></cylinder></geometry></collision>
      <visual name="visual">
        <geometry><cylinder><radius>0.25</radius><length>0.5</length></cylinder></geometry>
        <material><ambient>0.9 0.8 0.2 1</ambient><diffuse>0.9 0.8 0.2 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
EOF
)"

spawn_model "rl_obstacle_cyl_2" "-1.0 2.5 0.25 0 0 0" "$(cat <<'EOF'
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="rl_obstacle_cyl_2">
    <static>true</static>
    <link name="link">
      <collision name="collision"><geometry><cylinder><radius>0.30</radius><length>0.5</length></cylinder></geometry></collision>
      <visual name="visual">
        <geometry><cylinder><radius>0.30</radius><length>0.5</length></cylinder></geometry>
        <material><ambient>0.7 0.4 0.9 1</ambient><diffuse>0.7 0.4 0.9 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
EOF
)"

echo "== Done. =="

