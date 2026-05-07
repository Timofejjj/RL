#!/usr/bin/env python3
"""
Gymnasium environment + PPO training for course_robot in ROS 2 + Gazebo Sim.

Key features:
- Observations: N lidar rays + normalized distance/angle to goal
- Actions: continuous [linear, angular] command
- Reward: distance-to-goal progress (asymmetric backtrack), time penalty, collision/goal terminal rewards
- ROS/Gym sync via rclpy.spin_once(...) in reset() and step()
- Gazebo reset through gz services (world reset + model set_pose)
- Goal and obstacles move on every reset() for domain randomization (obstacle count follows curriculum)
- Stuck-against-wall ends the episode when lidar is close, forward odometry speed stays ~0, and patience elapses
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn

# --- Optional ROS 2 imports ---
# This file is designed to run in two modes:
# - ROS backend (rclpy + message types available)
# - standalone backend (no ROS installed)
#
# For static type-checkers, we keep ROS symbols as `Any` to avoid conflicts between
# real ROS types and lightweight runtime stubs defined below.
rclpy: Any
RclpyError: type[Exception] | tuple[type[Exception], ...] = Exception

PoseStamped: Any
Twist: Any
Odometry: Any
Node: Any
HistoryPolicy: Any
QoSProfile: Any
ReliabilityPolicy: Any
PointCloud2: Any
point_cloud2: Any
tf2_ros: Any
Time: Any
TransformException: type[Exception] | tuple[type[Exception], ...] = Exception
ROS_IMPORT_ERROR: Exception | None

try:
    import rclpy as _rclpy
    from rclpy.time import Time as _Time
    from geometry_msgs.msg import PoseStamped as _PoseStamped, Twist as _Twist
    from nav_msgs.msg import Odometry as _Odometry
    from rclpy.node import Node as _Node
    from rclpy.qos import HistoryPolicy as _HistoryPolicy, QoSProfile as _QoSProfile, ReliabilityPolicy as _ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2 as _PointCloud2
    from sensor_msgs_py import point_cloud2 as _point_cloud2
    import tf2_ros as _tf2_ros

    rclpy = _rclpy
    Time = _Time
    PoseStamped = _PoseStamped
    Twist = _Twist
    Odometry = _Odometry
    Node = _Node
    HistoryPolicy = _HistoryPolicy
    QoSProfile = _QoSProfile
    ReliabilityPolicy = _ReliabilityPolicy
    PointCloud2 = _PointCloud2
    point_cloud2 = _point_cloud2
    tf2_ros = _tf2_ros
    TransformException = cast(type[Exception], getattr(_tf2_ros, "TransformException", Exception))

    ROS_IMPORT_ERROR = None
except ImportError as import_error:
    class _RclpyStub:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("ROS backend is unavailable in this environment.")

    class _LoggerStub:
        def info(self, message: str) -> None:
            print(message)

        def warn(self, message: str) -> None:
            print(message)

        def error(self, message: str) -> None:
            print(message)

    class _SimpleVec:
        def __init__(self) -> None:
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.w = 1.0

    class _SimpleHeader:
        def __init__(self) -> None:
            self.frame_id = ""

    class _SimplePose:
        def __init__(self) -> None:
            self.position = _SimpleVec()
            self.orientation = _SimpleVec()

    class _TwistStub:
        def __init__(self) -> None:
            self.linear = _SimpleVec()
            self.angular = _SimpleVec()

    class _PoseStampedStub:
        def __init__(self) -> None:
            self.header = _SimpleHeader()
            self.pose = _SimplePose()

    class _OdometryStub:
        pass

    class _NodeStub:
        pass

    class _HistoryPolicyStub:
        KEEP_LAST = 1

    class _ReliabilityPolicyStub:
        BEST_EFFORT = 1
        RELIABLE = 2

    class _QoSProfileStub:
        def __init__(self, **_: Any) -> None:
            pass

    class _PointCloud2Stub:
        pass

    class _PointCloudReader:
        @staticmethod
        def read_points(*_: Any, **__: Any) -> list[tuple[float, float, float]]:
            return []

    class _Tf2RosStub:
        class Buffer:
            def __init__(self, *args: Any, **kwargs: Any) -> None: pass
            def lookup_transform(self, *args: Any, **kwargs: Any) -> Any: raise NotImplementedError()
        class TransformListener:
            def __init__(self, *args: Any, **kwargs: Any) -> None: pass

    class _TransformExceptionStub(Exception):
        pass

    class _TimeStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

    rclpy = _RclpyStub()
    Time = _TimeStub
    point_cloud2 = _PointCloudReader()
    PoseStamped = _PoseStampedStub
    Twist = _TwistStub
    Odometry = _OdometryStub
    Node = _NodeStub
    HistoryPolicy = _HistoryPolicyStub
    QoSProfile = _QoSProfileStub
    ReliabilityPolicy = _ReliabilityPolicyStub
    PointCloud2 = _PointCloud2Stub
    tf2_ros = _Tf2RosStub()
    TransformException = _TransformExceptionStub
    ROS_IMPORT_ERROR = import_error


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Convert quaternion to yaw angle (radians)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _dense_navigation_reward(
    prev_goal_distance_m: float,
    distance_to_goal_m: float,
    *,
    reward_progress_scale: float,
    reward_backtrack_scale: float,
    reward_progress_clip: float,
    reward_step_penalty: float,
    reward_angular_penalty: float,
    angular_magnitude: float,
    angle_to_goal_rad: float,
    reward_heading_scale: float,
    heading_speed_scale: float,
) -> float:
    """
    Dense shaping: reward only for reduced goal distance; stronger penalty when distance grows.
    Heading term is small and gated by forward motion (heading_speed_scale).
    """
    progress = float(prev_goal_distance_m - distance_to_goal_m)
    if reward_progress_clip > 0.0:
        progress = float(np.clip(progress, -reward_progress_clip, reward_progress_clip))
    reward = 0.0
    if progress > 0.0:
        reward += reward_progress_scale * progress
    elif progress < 0.0:
        reward -= reward_backtrack_scale * abs(progress)
    reward -= reward_step_penalty
    reward -= reward_angular_penalty * abs(angular_magnitude)
    if reward_heading_scale > 0.0 and heading_speed_scale > 0.0:
        cos_align = max(0.0, math.cos(angle_to_goal_rad))
        reward += reward_heading_scale * cos_align * heading_speed_scale
    return reward


def yaw_to_quat_z_w(yaw_rad: float) -> tuple[float, float]:
    return math.sin(float(yaw_rad) * 0.5), math.cos(float(yaw_rad) * 0.5)


ROS_AVAILABLE = ROS_IMPORT_ERROR is None

MANDATORY_INFO_KEYS: tuple[str, ...] = (
    "distance_to_goal_m",
    "angle_to_goal_rad",
    "min_lidar_m",
    "collision",
    "success",
    "stuck",
    "terminated_reason",
    "truncated_reason",
    "episode_step",
)

MONITOR_INFO_KEYWORDS: tuple[str, ...] = (
    "event",
    "distance_to_goal_m",
    "angle_to_goal_rad",
    "min_lidar_m",
    "collision",
    "success",
    "stuck",
    "terminated_reason",
    "truncated_reason",
    "episode_step",
)

FATAL_TRAINING_EVENTS: frozenset[str] = frozenset({"shutdown", "sensor_failure", "exception"})

INFO_SCHEMA_DEFAULTS: dict[str, Any] = {
    "distance_to_goal_m": 0.0,
    "angle_to_goal_rad": 0.0,
    "min_lidar_m": 0.0,
    "collision": False,
    "success": False,
    "stuck": False,
    "terminated_reason": "none",
    "truncated_reason": "none",
    "episode_step": 0,
}

# Lidar XY noise floor (robot frame): ignore closer returns from body / self-reflections.
DEFAULT_LIDAR_POINT_MIN_XY_M: float = 0.10

# Curriculum: no obstacles until enough episodes AND success rate, then enable obstacles.
CURRICULUM_FREE_EPISODES: int = 500
CURRICULUM_SUCCESS_THRESHOLD: float = 0.8
CURRICULUM_OBSTACLE_COUNT_HARD: int = 4

# Playable interior in course_robot_world.sdf (inside rl_arena_walls): 2 m × 10 m.
DEFAULT_ARENA_WIDTH_M: float = 2.0
DEFAULT_ARENA_LENGTH_M: float = 10.0
# Keep randomized spawn/goal away from inner wall faces (finish cylinder radius ≈ 0.12 m in SDF).
ARENA_BOUNDARY_INSET_M: float = 0.15


def _arena_sampling_bounds(
    arena_half_width_m: float,
    arena_half_length_m: float,
    inset_m: float,
) -> tuple[float, float, float, float]:
    """Axis-aligned rectangle strictly inside the arena, for sampling and clamping goals/spawns."""
    inset = max(0.05, float(inset_m))
    inset = min(inset, arena_half_width_m * 0.48, arena_half_length_m * 0.48)
    x_min = -float(arena_half_width_m) + inset
    x_max = float(arena_half_width_m) - inset
    y_min = -float(arena_half_length_m) + inset
    y_max = float(arena_half_length_m) - inset
    return x_min, x_max, y_min, y_max


def _sanitize_name(raw_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "model"


def _resolve_writable_root(preferred: Path, fallback: Path, label: str) -> Path:
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return preferred
    except OSError:
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"WARN: cannot write {label} root at {preferred}, fallback to {fallback}", file=sys.stderr)
        return fallback


def _create_run_dirs(log_root: Path, save_root: Path, model_name: str, requested_run_id: str = "") -> dict[str, Path | str]:
    base_run_id = requested_run_id.strip()
    if not base_run_id:
        timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
        base_run_id = f"run_{timestamp}_{_sanitize_name(model_name)}"

    for suffix_idx in range(1000):
        suffix = "" if suffix_idx == 0 else f"_{suffix_idx:02d}"
        run_id = f"{base_run_id}{suffix}"
        log_run_dir = log_root / run_id
        save_run_dir = save_root / run_id
        if log_run_dir.exists() or save_run_dir.exists():
            continue
        log_run_dir.mkdir(parents=True, exist_ok=False)
        save_run_dir.mkdir(parents=True, exist_ok=False)
        tb_dir = log_run_dir / "tensorboard"
        stdout_dir = log_run_dir / "stdout_stderr"
        checkpoint_dir = save_run_dir / "checkpoints"
        final_model_dir = save_run_dir / "final_model"
        tb_dir.mkdir(parents=True, exist_ok=True)
        stdout_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_model_dir.mkdir(parents=True, exist_ok=True)
        return {
            "run_id": run_id,
            "run_dir": log_run_dir,
            "log_dir": log_run_dir,
            "save_dir": save_run_dir,
            "tensorboard_dir": tb_dir,
            "stdout_dir": stdout_dir,
            "checkpoint_dir": checkpoint_dir,
            "final_model_dir": final_model_dir,
        }
    raise RuntimeError(f"Could not create unique run directory for base id: {base_run_id}")


def _git_hash_or_unknown(repo_dir: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or "unknown"
    except OSError:
        pass
    return "unknown"


class _TeeStream:
    def __init__(self, original: Any, log_file: Any) -> None:
        self._original = original
        self._log_file = log_file

    def write(self, message: str) -> int:
        self._original.write(message)
        self._log_file.write(message)
        return len(message)

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._original, "isatty", lambda: False)())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=True)
        f.write("\n")


class RobotEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    Custom Gymnasium environment for differential-drive obstacle avoidance + goal reaching.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        points_topic: str = "/lidar/points",
        odom_topic: str = "/model/course_robot/odometry",
        cmd_vel_topic: str = "/model/course_robot/cmd_vel",
        world_name: str = "course_world",
        model_name: str = "course_robot",
        num_lidar_beams: int = 72,
        max_lidar_range: float = 10.0,
        lidar_min_z: float = -0.10,
        lidar_max_z: float = 0.60,
        lidar_point_min_xy_m: float = DEFAULT_LIDAR_POINT_MIN_XY_M,
        collision_distance_m: float = 0.10,
        goal_threshold_m: float = 0.5,
        arena_width_m: float = DEFAULT_ARENA_WIDTH_M,
        arena_length_m: float = DEFAULT_ARENA_LENGTH_M,
        min_goal_distance_m: float = 1.0,
        max_goal_distance_norm_m: float = 13.0,
        max_linear_speed_mps: float = 0.50,
        min_linear_speed_mps: float = 0.0,
        max_angular_speed_radps: float = 1.5,
        control_dt_sec: float = 0.05,
        max_episode_steps: int = 600,
        max_linear_accel_mps2: float = 1.0,
        max_angular_accel_radps2: float = 5.0,
        angular_deadband: float = 0.08,
        reset_world_on_episode: bool = False,
        obstacle_randomize_every_episodes: int = 20,
        stuck_lidar_below_m: float = 0.22,
        stuck_linvel_below_mps: float = 0.028,
        stuck_patience_steps: int = 10,
        randomization_max_attempts: int = 120,
        goal_min_distance_m: float = 1.1,
        obstacle_min_distance_m: float = 0.95,
        gz_set_pose_timeout_ms: int = 2000,
        gz_world_control_timeout_ms: int = 2500,
        gz_service_retries: int = 2,
        auto_disable_world_reset_failures: int = 3,

        reward_progress_scale: float = 65.0,
        reward_progress_clip: float = 0.12,
        reward_backtrack_scale: float = 90.0,
        reward_step_penalty: float = 1.0,
        reward_angular_penalty: float = 0.01,
        reward_heading_scale: float = 0.06,
        reward_collision_penalty: float = 200.0,
        reward_stuck_penalty: float = 50.0,
        reward_goal_bonus: float = 2200.0,
        reward_clip_abs: float = 4000.0,
        spawn_x: float = 0.0,
        spawn_y: float = -4.6,
        spawn_z: float = 0.36,
        spawn_yaw_rad: float = 1.5708,
    ) -> None:
        super().__init__()
        if not ROS_AVAILABLE:
            raise RuntimeError(
                "ROS 2 python modules are not available. "
                "Use --backend standalone or install ROS 2/rclpy."
            ) from ROS_IMPORT_ERROR

        # rclpy.create_node() is dynamically typed (and can be stubbed when ROS is unavailable),
        # so we keep this attribute as Any for static type-checkers.
        self.node: Any = rclpy.create_node("course_robot_ppo_env")
        self._logger = self.node.get_logger()

        self.points_topic = points_topic
        self.odom_topic = odom_topic
        self.cmd_vel_topic = cmd_vel_topic
        self.world_name = world_name
        self.model_name = model_name

        self.num_lidar_beams = int(num_lidar_beams)
        self.max_lidar_range = float(max_lidar_range)
        self.lidar_min_z = float(lidar_min_z)
        self.lidar_max_z = float(lidar_max_z)
        self.lidar_point_min_xy_m = max(0.0, float(lidar_point_min_xy_m))
        self.collision_distance_m = float(collision_distance_m)
        self.goal_threshold_m = float(goal_threshold_m)
        self.arena_half_width_m = float(arena_width_m) * 0.5
        self.arena_half_length_m = float(arena_length_m) * 0.5
        self.min_goal_distance_m = float(min_goal_distance_m)
        self.max_goal_distance_norm_m = float(max_goal_distance_norm_m)
        self.max_linear_speed_mps = float(max_linear_speed_mps)
        self.min_linear_speed_mps = float(min_linear_speed_mps)
        self.max_angular_speed_radps = float(max_angular_speed_radps)
        self.control_dt_sec = float(control_dt_sec)
        self.max_episode_steps = int(max_episode_steps)
        self.max_linear_accel_mps2 = float(max_linear_accel_mps2)
        self.max_angular_accel_radps2 = float(max_angular_accel_radps2)
        self.angular_deadband = float(angular_deadband)
        self.reset_world_on_episode = bool(reset_world_on_episode)
        self.obstacle_randomize_every_episodes = max(0, int(obstacle_randomize_every_episodes))
        self.stuck_lidar_below_m = float(stuck_lidar_below_m)
        self.stuck_linvel_below_mps = float(stuck_linvel_below_mps)
        self.stuck_patience_steps = max(1, int(stuck_patience_steps))
        self.randomization_max_attempts = max(30, int(randomization_max_attempts))
        self.goal_min_distance_m = max(0.5, float(goal_min_distance_m))
        self.obstacle_min_distance_m = max(0.35, float(obstacle_min_distance_m))
        self.gz_set_pose_timeout_ms = max(300, int(gz_set_pose_timeout_ms))
        self.gz_world_control_timeout_ms = max(500, int(gz_world_control_timeout_ms))
        self.gz_service_retries = max(1, int(gz_service_retries))
        self.auto_disable_world_reset_failures = max(1, int(auto_disable_world_reset_failures))
        self.reward_progress_scale = float(reward_progress_scale)
        self.reward_progress_clip = max(0.0, float(reward_progress_clip))
        self.reward_backtrack_scale = max(0.0, float(reward_backtrack_scale))
        self.reward_step_penalty = max(0.0, float(reward_step_penalty))
        self.reward_angular_penalty = max(0.0, float(reward_angular_penalty))
        self.reward_heading_scale = float(reward_heading_scale)
        self.reward_collision_penalty = max(0.0, float(reward_collision_penalty))
        self.reward_stuck_penalty = max(0.0, float(reward_stuck_penalty))
        self.reward_goal_bonus = max(0.0, float(reward_goal_bonus))
        self.reward_clip_abs = max(0.0, float(reward_clip_abs))
        self.spawn_x = float(spawn_x)
        self.spawn_y = float(spawn_y)
        self.spawn_z = float(spawn_z)
        self.spawn_yaw_rad = float(spawn_yaw_rad)
        self.spawn_safe_x_min, self.spawn_safe_x_max, self.spawn_safe_y_min, self.spawn_safe_y_max = (
            _arena_sampling_bounds(self.arena_half_width_m, self.arena_half_length_m, ARENA_BOUNDARY_INSET_M)
        )
        self.goal_entity_name = "rl_finish_marker"
        self.obstacle_entity_names = [
            "obs_3_mid_small",
            "obs_4_mid_left",
            "obs_5_mid_right",
            "obs_6_lower_left_small",
            "obs_8_bottom_left",
        ]
        self.goal_spawn_z = 0.001
        self.obstacle_spawn_z = 0.15
        self.target_x = 0.0
        self.target_y = 4.7
        tx, ty = self._clamp_xy_to_arena_interior(self.target_x, self.target_y)
        self.target_x, self.target_y = tx, ty

        # PointCloud2 from Gazebo/gz_ros2_transport is almost always BEST_EFFORT. A RELIABLE
        # subscriber would not match such publishers and would receive no data. Use a deeper
        # KEEP_LAST queue to reduce callback loss vs depth=1 without changing reliability.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.node.create_subscription(PointCloud2, self.points_topic, self._on_pointcloud, sensor_qos)
        self.node.create_subscription(Odometry, self.odom_topic, self._on_odometry, 10)
        self.cmd_pub = self.node.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.reset_pose_pub = self.node.create_publisher(PoseStamped, "/course_robot/reset_pose", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

        self.latest_scan_norm = np.ones(self.num_lidar_beams, dtype=np.float32)
        self.latest_scan_m = np.full(self.num_lidar_beams, self.max_lidar_range, dtype=np.float32)
        self.odom_msg: Any | None = None
        self._fresh_lidar = False
        self.goal_xy = np.array([self.target_x, self.target_y], dtype=np.float32)
        self.prev_goal_distance = 0.0
        self.episode_step = 0
        self.current_linear_cmd = 0.0
        self.current_angular_cmd = 0.0
        self._episode_index = 0
        self._stuck_steps = 0
        self._blocked_steps = 0
        self._obstacle_xy_cache: list[tuple[float, float]] = []
        self._goal_xy_cache: tuple[float, float] | None = None
        self._world_reset_failures = 0
        self._last_terminal_event = "initial"
        self._fast_collision_count = 0
        self._curriculum_episodes_total: int = 0
        self._curriculum_success_rate: float = 0.0
        self._last_layout_obstacle_k_applied: int | None = None

        obs_dim = self.num_lidar_beams + 2
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self._logger.info(
            f"RobotEnv started. points={self.points_topic}, odom={self.odom_topic}, cmd={self.cmd_vel_topic}"
        )

    # ---------- ROS callbacks ----------
    def _on_odometry(self, msg: Any) -> None:
        self.odom_msg = msg

    def _on_pointcloud(self, msg: Any) -> None:
        self.latest_scan_m = self._pointcloud_to_planar_rays(msg)
        self.latest_scan_norm = np.clip(1.0 - np.log1p(self.latest_scan_m) / np.log1p(self.max_lidar_range), 0.0, 1.0).astype(np.float32)
        self._fresh_lidar = True

    # ---------- PointCloud2 processing ----------
    def _pointcloud_to_planar_rays(self, msg: Any) -> np.ndarray:
        """
        Project PointCloud2 to horizontal lidar rays.
        Each bin stores minimum XY distance for the corresponding azimuth sector.
        """
        rays = np.full(self.num_lidar_beams, self.max_lidar_range, dtype=np.float32)
        bin_width = 2.0 * math.pi / float(self.num_lidar_beams)

        for x, y, z in point_cloud2.read_points(cast(Any, msg), field_names=["x", "y", "z"], skip_nans=True):
            if z < self.lidar_min_z or z > self.lidar_max_z:
                continue

            distance_xy = math.hypot(x, y)
            if distance_xy < self.lidar_point_min_xy_m:
                continue
            if distance_xy > self.max_lidar_range:
                continue

            angle = math.atan2(y, x)  # [-pi, pi]
            idx = int((angle + math.pi) / bin_width)
            if idx < 0:
                idx = 0
            elif idx >= self.num_lidar_beams:
                idx = self.num_lidar_beams - 1

            if distance_xy < rays[idx]:
                rays[idx] = distance_xy

        return rays

    # ---------- State helpers ----------
    def _current_pose(self) -> tuple[float, float, float]:
        try:
            # Get transform from odom to base_link
            t = self.tf_buffer.lookup_transform(
                "course_robot_odom",
                "course_robot_base_link",
                Time()
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
            return float(x), float(y), float(yaw)
        except TransformException:
            # Fallback to raw odometry if TF is unavailable
            if self.odom_msg is None:
                return self.spawn_x, self.spawn_y, self.spawn_yaw_rad

            odom = cast(Any, self.odom_msg)
            p = odom.pose.pose.position
            q = odom.pose.pose.orientation
            yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
            return float(p.x), float(p.y), float(yaw)

    def _goal_features(self) -> tuple[float, float]:
        robot_x, robot_y, robot_yaw = self._current_pose()
        dx = float(self.target_x - robot_x)
        dy = float(self.target_y - robot_y)
        distance = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        angle_to_goal = wrap_to_pi(target_heading - robot_yaw)
        return distance, angle_to_goal

    def _safe_goal_features(self) -> tuple[float, float]:
        try:
            return self._goal_features()
        except Exception:
            return 0.0, 0.0

    def _safe_min_lidar(self) -> float:
        try:
            if self.latest_scan_m.size == 0:
                return self.max_lidar_range
            return float(np.min(self.latest_scan_m))
        except Exception:
            return self.max_lidar_range

    def _safe_observation(self) -> np.ndarray:
        try:
            return self._build_observation()
        except Exception:
            shape = cast(tuple[int, ...], self.observation_space.shape)
            return np.zeros(shape, dtype=np.float32)

    def _build_info(
        self,
        *,
        event: str,
        collision: bool = False,
        success: bool = False,
        stuck: bool = False,
        terminated_reason: str = "none",
        truncated_reason: str = "none",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        distance_to_goal, angle_to_goal = self._safe_goal_features()
        min_lidar = self._safe_min_lidar()
        info: dict[str, Any] = {
            "event": event,
            "goal_xy": self.goal_xy.copy(),
            "goal": self.goal_xy.copy(),
            "goal_x": float(self.goal_xy[0]),
            "goal_y": float(self.goal_xy[1]),
            "distance_to_goal_m": float(distance_to_goal),
            "angle_to_goal_rad": float(angle_to_goal),
            "min_lidar_m": float(min_lidar),
            "collision": bool(collision),
            "success": bool(success),
            "stuck": bool(stuck),
            "terminated_reason": terminated_reason,
            "truncated_reason": truncated_reason,
            "episode_step": int(self.episode_step),
        }
        if extra:
            info.update(extra)
        for key in MANDATORY_INFO_KEYS:
            info.setdefault(key, INFO_SCHEMA_DEFAULTS[key])
        return info

    def _build_observation(self) -> np.ndarray:
        distance_to_goal, angle_to_goal = self._goal_features()
        dist_norm = float(np.clip(distance_to_goal / self.max_goal_distance_norm_m, 0.0, 1.0))
        ang_norm = float(np.clip(angle_to_goal / math.pi, -1.0, 1.0))
        obs = np.concatenate(
            [
                self.latest_scan_norm.astype(np.float32),
                np.array([dist_norm, ang_norm], dtype=np.float32),
            ]
        )
        return obs.astype(np.float32)

    # ---------- Gazebo reset ----------
    def _call_gz_service(
        self,
        service: str,
        reqtype: str,
        reptype: str,
        request: str,
        timeout_ms: int = 700,
        retries: int = 1,
        retry_sleep_sec: float = 0.05,
    ) -> bool:
        cmd = [
            "gz",
            "service",
            "-s",
            service,
            "--reqtype",
            reqtype,
            "--reptype",
            reptype,
            "--timeout",
            str(timeout_ms),
            "--req",
            request,
        ]
        attempts = max(1, int(retries))
        last_error = "unknown"
        for attempt in range(attempts):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            except FileNotFoundError:
                self._logger.error("Command 'gz' not found. Make sure Gazebo Sim CLI is available in PATH.")
                return False

            out = (proc.stdout or "").replace(" ", "").lower()
            if proc.returncode == 0 and ("data:true" in out or "success:true" in out):
                return True

            if proc.returncode != 0:
                last_error = proc.stderr.strip() or f"returncode={proc.returncode}"
            else:
                last_error = (proc.stdout or "").strip() or "service returned non-success payload"

            if attempt + 1 < attempts:
                time.sleep(max(0.0, float(retry_sleep_sec)))

        self._logger.warn(
            f"Service call failed: {service}; attempts={attempts}; timeout_ms={timeout_ms}; last_error={last_error}"
        )
        return False

    def _set_world_paused(self, paused: bool) -> bool:
        req = f"pause: {'true' if paused else 'false'}"
        return self._call_gz_service(
            service=f"/world/{self.world_name}/control",
            reqtype="gz.msgs.WorldControl",
            reptype="gz.msgs.Boolean",
            request=req,
            timeout_ms=self.gz_world_control_timeout_ms,
            retries=self.gz_service_retries,
            retry_sleep_sec=0.08,
        )

    def _reset_world(self) -> bool:
        # model_only resets poses/velocities without rewinding simulation time used by RViz/TF.
        req = "reset { model_only: true } pause: true"
        return self._call_gz_service(
            service=f"/world/{self.world_name}/control",
            reqtype="gz.msgs.WorldControl",
            reptype="gz.msgs.Boolean",
            request=req,
            timeout_ms=self.gz_world_control_timeout_ms,
            retries=self.gz_service_retries,
            retry_sleep_sec=0.08,
        )

    def _set_entity_pose(self, entity_name: str, x: float, y: float, z: float, yaw_rad: float) -> bool:
        qz, qw = yaw_to_quat_z_w(yaw_rad)
        req = (
            f'name: "{entity_name}" '
            f'position {{ x: {x} y: {y} z: {z} }} '
            f'orientation {{ x: 0 y: 0 z: {qz} w: {qw} }}'
        )
        return self._call_gz_service(
            service=f"/world/{self.world_name}/set_pose",
            reqtype="gz.msgs.Pose",
            reptype="gz.msgs.Boolean",
            request=req,
            timeout_ms=self.gz_set_pose_timeout_ms,
            retries=self.gz_service_retries,
            retry_sleep_sec=0.05,
        )

    def _set_entity_pose_with_retry(
        self,
        entity_name: str,
        x: float,
        y: float,
        z: float,
        yaw_rad: float,
        retries: int = 3,
        retry_sleep_sec: float = 0.04,
    ) -> bool:
        for attempt in range(max(1, retries)):
            ok = self._set_entity_pose(entity_name=entity_name, x=x, y=y, z=z, yaw_rad=yaw_rad)
            if ok:
                return True
            if attempt + 1 < max(1, retries):
                time.sleep(retry_sleep_sec)
        return False

    def _set_robot_pose(self, retries: int = 6, retry_sleep_sec: float = 0.08) -> bool:
        return self._set_entity_pose_with_retry(
            entity_name=self.model_name,
            x=self.spawn_x,
            y=self.spawn_y,
            z=self.spawn_z,
            yaw_rad=self.spawn_yaw_rad,
            retries=retries,
            retry_sleep_sec=retry_sleep_sec,
        )

    def _safe_spin_once(self, timeout_sec: float) -> bool:
        if not rclpy.ok():
            return False
        try:
            rclpy.spin_once(self.node, timeout_sec=timeout_sec)
            return True
        except RclpyError:
            return False

    def _wait_for_odom(self, timeout_sec: float = 0.8) -> None:
        """Spin until at least one odometry sample is available after reset."""
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < timeout_sec:
            if not self._safe_spin_once(timeout_sec=0.05):
                break
            if self.odom_msg is not None:
                return

    def _publish_reset_pose(self) -> None:
        if not rclpy.ok():
            return
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.pose.position.x = self.spawn_x
        msg.pose.position.y = self.spawn_y
        msg.pose.position.z = self.spawn_z
        qz, qw = yaw_to_quat_z_w(self.spawn_yaw_rad)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        try:
            self.reset_pose_pub.publish(msg)
        except RclpyError:
            pass

    def _stop_robot(self, repeats: int = 3, spin_timeout_sec: float = 0.03) -> None:
        self.current_linear_cmd = 0.0
        self.current_angular_cmd = 0.0
        if not rclpy.ok():
            return
        zero_cmd = Twist()
        for _ in range(max(1, repeats)):
            try:
                self.cmd_pub.publish(zero_cmd)
            except RclpyError:
                break
            if not self._safe_spin_once(timeout_sec=spin_timeout_sec):
                break

    def _forward_speed_mps(self) -> float:
        if self.odom_msg is None:
            return 0.0
        odom = cast(Any, self.odom_msg)
        return abs(float(odom.twist.twist.linear.x))

    def _sample_random_xy(self) -> tuple[float, float]:
        x = float(self.np_random.uniform(self.spawn_safe_x_min, self.spawn_safe_x_max))
        y = float(self.np_random.uniform(self.spawn_safe_y_min, self.spawn_safe_y_max))
        return x, y

    def _clamp_xy_to_arena_interior(self, x: float, y: float) -> tuple[float, float]:
        """Force a point into the same bounds used for randomization (inside red walls)."""
        return (
            float(np.clip(x, self.spawn_safe_x_min, self.spawn_safe_x_max)),
            float(np.clip(y, self.spawn_safe_y_min, self.spawn_safe_y_max)),
        )

    def _is_position_valid(
        self,
        x: float,
        y: float,
        occupied_positions: list[tuple[float, float]],
        min_distance_m: float,
    ) -> bool:
        # Reject samples that are too close to any occupied point
        # so that goal/obstacles do not overlap and form blocked walls.
        for occ_x, occ_y in occupied_positions:
            if math.hypot(x - occ_x, y - occ_y) < min_distance_m:
                return False
        return True

    def _sample_safe_position(
        self,
        occupied_positions: list[tuple[float, float]],
        min_distance_m: float,
        max_attempts: int = 50,
        min_distance_to_robot_m: float = 0.0,
    ) -> tuple[float, float] | None:
        attempts = max(1, int(max_attempts))
        for attempt in range(attempts):
            progress = (attempt + 1) / attempts
            if progress < 0.6:
                relax_factor = 1.0
            elif progress < 0.85:
                relax_factor = 0.85
            else:
                relax_factor = 0.7

            effective_min_distance = max(0.25, float(min_distance_m) * relax_factor)
            if min_distance_to_robot_m <= 0.0:
                effective_robot_distance = 0.0
            else:
                effective_robot_distance = max(0.35, float(min_distance_to_robot_m) * relax_factor)

            x, y = self._sample_random_xy()
            if math.hypot(x - self.spawn_x, y - self.spawn_y) < effective_robot_distance:
                continue
            if self._is_position_valid(x, y, occupied_positions, effective_min_distance):
                return x, y
        return None

    def set_training_curriculum_state(self, episodes_total: int, success_rate: float) -> None:
        """Updated from training callback so reset() can match obstacle count to curriculum."""
        self._curriculum_episodes_total = max(0, int(episodes_total))
        self._curriculum_success_rate = float(success_rate)

    def _curriculum_obstacle_count(self) -> int:
        if self._curriculum_episodes_total < CURRICULUM_FREE_EPISODES or self._curriculum_success_rate < CURRICULUM_SUCCESS_THRESHOLD:
            return 0
        return CURRICULUM_OBSTACLE_COUNT_HARD

    def _obstacle_stash_pose(self, stash_index: int) -> tuple[float, float, float]:
        """Move unused Gazebo obstacles far from the arena so lidar does not see them."""
        step = 2.5
        x = self.arena_half_width_m + 12.0 + float(stash_index) * step
        y = self.arena_half_length_m + 12.0
        return x, y, -10.0

    def _randomize_episode_layout(self, randomize_obstacles: bool) -> bool:
        occupied_positions: list[tuple[float, float]] = [(self.spawn_x, self.spawn_y)]
        if not randomize_obstacles:
            if len(self._obstacle_xy_cache) != len(self.obstacle_entity_names):
                randomize_obstacles = True
            else:
                occupied_positions.extend(self._obstacle_xy_cache)

        goal_xy = self._sample_safe_position(
            occupied_positions=occupied_positions,
            min_distance_m=max(self.goal_min_distance_m, self.min_goal_distance_m),
            min_distance_to_robot_m=max(self.goal_min_distance_m, self.min_goal_distance_m),
            max_attempts=self.randomization_max_attempts,
        )
        if goal_xy is None:
            if self._goal_xy_cache is not None:
                goal_xy = self._goal_xy_cache
                self._logger.warn(
                    "Could not sample safe goal position; reusing last valid goal position from cache."
                )
            else:
                fallback_goal = (float(self.goal_xy[0]), float(self.goal_xy[1]))
                goal_xy = fallback_goal
                self._logger.warn(
                    "Could not sample safe goal position; keeping previous goal position for this episode."
                )

        goal_x, goal_y = self._clamp_xy_to_arena_interior(float(goal_xy[0]), float(goal_xy[1]))
        occupied_positions.append((goal_x, goal_y))
        goal_ok = self._set_entity_pose_with_retry(
            entity_name=self.goal_entity_name,
            x=goal_x,
            y=goal_y,
            z=self.goal_spawn_z,
            yaw_rad=float(self.np_random.uniform(-math.pi, math.pi)),
            retries=6,
            retry_sleep_sec=0.08,
        )
        if not goal_ok:
            if self._goal_xy_cache is not None:
                goal_x, goal_y = self._clamp_xy_to_arena_interior(
                    float(self._goal_xy_cache[0]),
                    float(self._goal_xy_cache[1]),
                )
                occupied_positions[-1] = (goal_x, goal_y)
                self._logger.warn(
                    f"Failed to move goal marker '{self.goal_entity_name}'; using cached goal coordinates in observations."
                )
            else:
                self._logger.warn(f"Failed to move goal marker '{self.goal_entity_name}'.")
        else:
            self._goal_xy_cache = (goal_x, goal_y)

        goal_x, goal_y = self._clamp_xy_to_arena_interior(goal_x, goal_y)
        self.target_x = goal_x
        self.target_y = goal_y
        self.goal_xy = np.array([self.target_x, self.target_y], dtype=np.float32)

        if not randomize_obstacles:
            return goal_ok

        obstacles_ok = True
        moved_obstacles = 0
        new_cache: list[tuple[float, float]] = []
        k_active = min(self._curriculum_obstacle_count(), len(self.obstacle_entity_names))
        for idx, obstacle_name in enumerate(self.obstacle_entity_names):
            if idx >= k_active:
                sx, sy, sz = self._obstacle_stash_pose(idx)
                obs_ok = self._set_entity_pose_with_retry(
                    entity_name=obstacle_name,
                    x=sx,
                    y=sy,
                    z=sz,
                    yaw_rad=float(self.np_random.uniform(-math.pi, math.pi)),
                    retries=6,
                    retry_sleep_sec=0.08,
                )
                new_cache.append((sx, sy))
                obstacles_ok = obstacles_ok and obs_ok
                if obs_ok:
                    moved_obstacles += 1
                continue

            cached_xy = self._obstacle_xy_cache[idx] if idx < len(self._obstacle_xy_cache) else None
            obstacle_xy = self._sample_safe_position(
                occupied_positions=occupied_positions,
                min_distance_m=self.obstacle_min_distance_m,
                max_attempts=self.randomization_max_attempts,
            )
            if obstacle_xy is None:
                if cached_xy is not None:
                    occupied_positions.append(cached_xy)
                    new_cache.append(cached_xy)
                    self._logger.warn(
                        f"Could not sample safe position for {obstacle_name}; reusing cached coordinates."
                    )
                else:
                    self._logger.warn(
                        f"Could not sample safe position for {obstacle_name} after {self.randomization_max_attempts} attempts."
                    )
                    obstacles_ok = False
                continue

            obs_x, obs_y = obstacle_xy
            occupied_positions.append((obs_x, obs_y))
            obs_ok = self._set_entity_pose_with_retry(
                entity_name=obstacle_name,
                x=obs_x,
                y=obs_y,
                z=self.obstacle_spawn_z,
                yaw_rad=float(self.np_random.uniform(-math.pi, math.pi)),
                retries=6,
                retry_sleep_sec=0.08,
            )
            if obs_ok:
                moved_obstacles += 1
                new_cache.append((obs_x, obs_y))
            else:
                if cached_xy is not None:
                    occupied_positions[-1] = cached_xy
                    new_cache.append(cached_xy)
                    self._logger.warn(
                        f"Failed to move obstacle '{obstacle_name}'; keeping cached obstacle coordinates."
                    )
                else:
                    self._logger.warn(f"Failed to move obstacle '{obstacle_name}'.")
            obstacles_ok = obstacles_ok and obs_ok

        if len(new_cache) == len(self.obstacle_entity_names):
            self._obstacle_xy_cache = new_cache
        self._last_layout_obstacle_k_applied = k_active
        self._logger.info(
            f"Randomized layout: goal=({goal_x:.2f}, {goal_y:.2f}), moved_obstacles={moved_obstacles}/{len(self.obstacle_entity_names)}, "
            f"curriculum_active_obstacles={k_active}"
        )
        return goal_ok and obstacles_ok

    # ---------- Gym API ----------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.episode_step = 0
        self.current_linear_cmd = 0.0
        self.current_angular_cmd = 0.0
        self._stuck_steps = 0
        self._blocked_steps = 0

        if not hasattr(self, '_fast_collision_count'):
            self._fast_collision_count = 0

        every = self.obstacle_randomize_every_episodes
        layout_cache_ready = self._goal_xy_cache is not None and len(self._obstacle_xy_cache) == len(
            self.obstacle_entity_names
        )
        if every <= 0:
            randomize_obstacles = not layout_cache_ready
        elif every == 1:
            randomize_obstacles = True
        else:
            randomize_obstacles = (self._episode_index % every) == 0

        k_now = self._curriculum_obstacle_count()
        if self._last_layout_obstacle_k_applied != k_now:
            randomize_obstacles = True
            self._obstacle_xy_cache = []

        fast_collision_reset = (
            self._last_terminal_event in {"collision", "stuck"}
            and layout_cache_ready
            and not randomize_obstacles
        )

        if fast_collision_reset:
            self._fast_collision_count += 1
            if self._fast_collision_count >= 5:
                fast_collision_reset = False
                randomize_obstacles = True
                
                # Force cache drop to ensure actual randomization happens
                self._goal_xy_cache = None
                self._obstacle_xy_cache = []
                
                self._fast_collision_count = 0
                self._logger.info("Fast collision count reached 5, forcing full layout randomization.")
        else:
            self._fast_collision_count = 0

        # Reset model poses/velocities without rewinding simulation time, then apply randomized layout.
        world_ok = True
        pause_ok = True
        unpause_ok = True
        randomization_ok = True
        self._stop_robot(repeats=2 if fast_collision_reset else 4)
        if not fast_collision_reset:
            pause_ok = self._set_world_paused(True)
            if not pause_ok:
                self._logger.warn("Could not pause world before reset; proceeding with best effort.")
        if self.reset_world_on_episode and not fast_collision_reset:
            world_ok = self._reset_world()
            if world_ok:
                self._world_reset_failures = 0
            else:
                self._world_reset_failures += 1
                if self._world_reset_failures >= self.auto_disable_world_reset_failures:
                    self._logger.warn(
                        "World reset keeps failing; auto-disabling reset_world_on_episode for this run."
                    )
                    self.reset_world_on_episode = False
        pose_ok = self._set_robot_pose(
            retries=2 if fast_collision_reset else 6,
            retry_sleep_sec=0.03 if fast_collision_reset else 0.08,
        )
        if not world_ok:
            self._logger.warn("World reset failed or unavailable; continuing with pose reset.")
        if not pose_ok:
            self._logger.warn("Pose reset failed; episode will continue from current robot pose.")

        if not fast_collision_reset:
            randomization_ok = self._randomize_episode_layout(randomize_obstacles=randomize_obstacles)
            if not randomization_ok:
                self._logger.warn("Domain randomization finished with warnings. Continuing episode.")

        self.odom_msg = None
        self._fresh_lidar = False
        if not fast_collision_reset:
            unpause_ok = self._set_world_paused(False)
            if not unpause_ok:
                self._logger.warn("Could not unpause world after reset; sensor updates may stall.")
        if pose_ok:
            self._wait_for_odom(timeout_sec=0.35 if fast_collision_reset else 0.8)
        self._publish_reset_pose()
        self._stop_robot(repeats=2 if fast_collision_reset else 4)

        # Synchronize and wait for fresh sensor data.
        sensor_timeout_sec = 0.8 if fast_collision_reset else 2.0
        t_start = time.time()
        while rclpy.ok() and time.time() - t_start < sensor_timeout_sec:
            if not self._safe_spin_once(timeout_sec=0.1):
                break
            if self.odom_msg is not None and self._fresh_lidar:
                break

        self._episode_index += 1
        self._last_terminal_event = "reset"
        self.prev_goal_distance, _ = self._goal_features()
        obs = self._build_observation()
        info = self._build_info(
            event="reset",
            extra={
                "fast_collision_reset": bool(fast_collision_reset),
                "world_reset_ok": bool(world_ok),
                "pose_reset_ok": bool(pose_ok),
                "randomization_ok": bool(randomization_ok),
                "layout_randomized": bool(not fast_collision_reset),
                "obstacles_randomized": bool(randomize_obstacles and not fast_collision_reset),
                "pause_ok": bool(pause_ok),
                "unpause_ok": bool(unpause_ok),
                "sensor_ready": bool(self.odom_msg is not None and self._fresh_lidar),
            },
        )
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        try:
            self.episode_step += 1
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

            # action[0] in [-1, 1] -> linear speed in [min_linear_speed, max_linear_speed]
            target_linear = self.min_linear_speed_mps + ((float(action[0]) + 1.0) * 0.5) * (
                self.max_linear_speed_mps - self.min_linear_speed_mps
            )
            # action[1] in [-1, 1] -> angular speed in [-max_angular_speed, max_angular_speed]
            angular_action = float(action[1])
            if abs(angular_action) < self.angular_deadband:
                angular_action = 0.0
            target_angular = angular_action * self.max_angular_speed_radps

            # Smooth commands to avoid oscillatory jerk and reduce wobble.
            max_linear_delta = self.max_linear_accel_mps2 * self.control_dt_sec
            max_angular_delta = self.max_angular_accel_radps2 * self.control_dt_sec
            self.current_linear_cmd += float(
                np.clip(target_linear - self.current_linear_cmd, -max_linear_delta, max_linear_delta)
            )
            self.current_angular_cmd += float(
                np.clip(target_angular - self.current_angular_cmd, -max_angular_delta, max_angular_delta)
            )
            linear = float(np.clip(self.current_linear_cmd, self.min_linear_speed_mps, self.max_linear_speed_mps))
            angular = float(np.clip(self.current_angular_cmd, -self.max_angular_speed_radps, self.max_angular_speed_radps))

            cmd = Twist()
            cmd.linear.x = linear
            cmd.angular.z = angular
            if not rclpy.ok():
                info = self._build_info(
                    event="shutdown",
                    terminated_reason="shutdown",
                    extra={"forward_speed_mps": 0.0},
                )
                return self._safe_observation(), 0.0, True, False, info
            try:
                self.cmd_pub.publish(cmd)
            except RclpyError:
                info = self._build_info(
                    event="shutdown",
                    terminated_reason="shutdown",
                    extra={"forward_speed_mps": 0.0},
                )
                return self._safe_observation(), 0.0, True, False, info

            # Let ROS callbacks update odom/lidar before computing next state.
            t_end = time.time() + self.control_dt_sec
            while rclpy.ok() and time.time() < t_end:
                if not self._safe_spin_once(timeout_sec=0.1):
                    fail_reason = "sensor_failure" if rclpy.ok() else "shutdown"
                    info = self._build_info(
                        event=fail_reason,
                        terminated_reason=fail_reason,
                        extra={"forward_speed_mps": 0.0},
                    )
                    return self._safe_observation(), 0.0, True, False, info

            distance_to_goal, angle_to_goal = self._safe_goal_features()
            min_lidar = self._safe_min_lidar()
            sensor_snapshot_min_lidar = float(min_lidar)
            sensor_snapshot_distance = float(distance_to_goal)
            sensor_snapshot_angle = float(angle_to_goal)
            fwd_speed = self._forward_speed_mps()
            raw_progress_delta = self.prev_goal_distance - distance_to_goal
            spinning = abs(self.current_angular_cmd) > 0.22
            if (
                min_lidar < self.stuck_lidar_below_m
                and fwd_speed < self.stuck_linvel_below_mps
                and not spinning
            ):
                self._stuck_steps += 1
            else:
                self._stuck_steps = 0
            # Blocked: commanded forward motion with almost no turn, but odometry shows almost no progress.
            if (
                linear > 0.02
                and abs(self.current_angular_cmd) < 0.1
                and fwd_speed < self.stuck_linvel_below_mps
                and raw_progress_delta < 0.002
            ):
                self._blocked_steps += 1
            else:
                self._blocked_steps = 0

            reward = _dense_navigation_reward(
                self.prev_goal_distance,
                distance_to_goal,
                reward_progress_scale=self.reward_progress_scale,
                reward_backtrack_scale=self.reward_backtrack_scale,
                reward_progress_clip=self.reward_progress_clip,
                reward_step_penalty=self.reward_step_penalty,
                reward_angular_penalty=self.reward_angular_penalty,
                angular_magnitude=angular,
                angle_to_goal_rad=angle_to_goal,
                reward_heading_scale=self.reward_heading_scale,
                heading_speed_scale=fwd_speed,
            )

            # Штрафуем за препятствия только если они РЕАЛЬНО близко (например, ближе 30 см)
            if min_lidar < 0.30:
                safety_penalty = 10.0 * (0.30 - min_lidar) ** 2
                reward -= safety_penalty

            terminated = False
            truncated = False
            event = "running"
            collision = False
            success = False
            stuck = False
            terminated_reason = "none"
            truncated_reason = "none"

            if min_lidar < self.collision_distance_m:
                reward -= self.reward_collision_penalty
                terminated = True
                event = "collision"
                collision = True
                terminated_reason = "collision"
                self._logger.info(f"Collision detected! min_lidar={min_lidar:.3f} < {self.collision_distance_m}")
            elif self._stuck_steps >= self.stuck_patience_steps:
                reward -= self.reward_stuck_penalty
                terminated = True
                event = "stuck"
                stuck = True
                terminated_reason = "stuck"
            elif self._blocked_steps >= max(12, self.stuck_patience_steps * 4):
                reward -= self.reward_stuck_penalty
                terminated = True
                event = "stuck"
                stuck = True
                terminated_reason = "stuck"
            elif distance_to_goal < self.goal_threshold_m:
                reward += self.reward_goal_bonus
                terminated = True
                event = "goal_reached"
                success = True
                terminated_reason = "goal_reached"
            elif self.episode_step >= self.max_episode_steps:
                truncated = True
                event = "max_steps"
                truncated_reason = "max_steps"
                reward -= 10.0 * distance_to_goal

            if self.reward_clip_abs > 0.0:
                reward = float(np.clip(reward, -self.reward_clip_abs, self.reward_clip_abs))

            self.prev_goal_distance = distance_to_goal
            obs = self._safe_observation()
            info = self._build_info(
                event=event,
                collision=collision,
                success=success,
                stuck=stuck,
                terminated_reason=terminated_reason,
                truncated_reason=truncated_reason,
                extra={
                    "distance_to_goal_m": sensor_snapshot_distance,
                    "angle_to_goal_rad": sensor_snapshot_angle,
                    "min_lidar_m": sensor_snapshot_min_lidar,
                    "forward_speed_mps": float(fwd_speed),
                    "blocked_steps": int(self._blocked_steps),
                },
            )
            if terminated or truncated:
                self._last_terminal_event = event
                self._stop_robot(repeats=4)
            return obs, float(reward), terminated, truncated, info
        except Exception as exc:
            try:
                self._logger.error(f"step exception: {exc}")
            except Exception:
                pass
            info = self._build_info(
                event="exception",
                terminated_reason="exception",
                extra={"exception": repr(exc)},
            )
            self._last_terminal_event = "exception"
            return self._safe_observation(), 0.0, True, False, info

    def close(self) -> None:
        try:
            self._stop_robot(repeats=2)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass


class StandaloneRobotEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    Lightweight fallback env that mimics course_robot task without ROS/Gazebo.
    Useful when ROS 2 python modules are unavailable on the host machine.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_lidar_beams: int = 72,
        max_lidar_range: float = 10.0,
        lidar_point_min_xy_m: float = DEFAULT_LIDAR_POINT_MIN_XY_M,
        collision_distance_m: float = 0.10,
        goal_threshold_m: float = 0.5,
        arena_width_m: float = DEFAULT_ARENA_WIDTH_M,
        arena_length_m: float = DEFAULT_ARENA_LENGTH_M,
        min_goal_distance_m: float = 1.0,
        max_goal_distance_norm_m: float = 13.0,
        max_linear_speed_mps: float = 0.50,
        min_linear_speed_mps: float = 0.0,
        max_angular_speed_radps: float = 1.5,
        control_dt_sec: float = 0.05,
        max_episode_steps: int = 600,
        randomization_max_attempts: int = 120,
        goal_min_distance_m: float = 1.1,
        obstacle_min_distance_m: float = 0.95,
        stuck_lidar_below_m: float = 0.22,
        stuck_linvel_below_mps: float = 0.028,
        stuck_patience_steps: int = 10,
        angular_deadband: float = 0.08,
        reward_progress_scale: float = 65.0,
        reward_progress_clip: float = 0.12,
        reward_backtrack_scale: float = 90.0,
        reward_step_penalty: float = 1.0,
        reward_angular_penalty: float = 0.01,
        reward_heading_scale: float = 0.06,
        reward_collision_penalty: float = 200.0,
        reward_stuck_penalty: float = 50.0,
        reward_goal_bonus: float = 2200.0,
        reward_clip_abs: float = 4000.0,
        spawn_x: float = 0.0,
        spawn_y: float = -4.6,
        spawn_yaw_rad: float = 1.5708,
    ) -> None:
        super().__init__()
        self.num_lidar_beams = int(num_lidar_beams)
        self.max_lidar_range = float(max_lidar_range)
        self.lidar_point_min_xy_m = max(0.0, float(lidar_point_min_xy_m))
        self.collision_distance_m = float(collision_distance_m)
        self.goal_threshold_m = float(goal_threshold_m)
        self.arena_half_width_m = float(arena_width_m) * 0.5
        self.arena_half_length_m = float(arena_length_m) * 0.5
        self.min_goal_distance_m = float(min_goal_distance_m)
        self.max_goal_distance_norm_m = float(max_goal_distance_norm_m)
        self.max_linear_speed_mps = float(max_linear_speed_mps)
        self.min_linear_speed_mps = float(min_linear_speed_mps)
        self.max_angular_speed_radps = float(max_angular_speed_radps)
        self.control_dt_sec = float(control_dt_sec)
        self.max_episode_steps = int(max_episode_steps)
        self.randomization_max_attempts = max(30, int(randomization_max_attempts))
        self.goal_min_distance_m = max(0.5, float(goal_min_distance_m))
        self.obstacle_min_distance_m = max(0.35, float(obstacle_min_distance_m))
        self.stuck_lidar_below_m = float(stuck_lidar_below_m)
        self.stuck_linvel_below_mps = float(stuck_linvel_below_mps)
        self.stuck_patience_steps = max(1, int(stuck_patience_steps))
        self.angular_deadband = float(angular_deadband)
        self.reward_progress_scale = float(reward_progress_scale)
        self.reward_progress_clip = max(0.0, float(reward_progress_clip))
        self.reward_backtrack_scale = max(0.0, float(reward_backtrack_scale))
        self.reward_step_penalty = max(0.0, float(reward_step_penalty))
        self.reward_angular_penalty = max(0.0, float(reward_angular_penalty))
        self.reward_heading_scale = float(reward_heading_scale)
        self.reward_collision_penalty = max(0.0, float(reward_collision_penalty))
        self.reward_stuck_penalty = max(0.0, float(reward_stuck_penalty))
        self.reward_goal_bonus = max(0.0, float(reward_goal_bonus))
        self.reward_clip_abs = max(0.0, float(reward_clip_abs))
        self.spawn_x = float(spawn_x)
        self.spawn_y = float(spawn_y)
        self.spawn_yaw_rad = float(spawn_yaw_rad)

        self.spawn_safe_x_min, self.spawn_safe_x_max, self.spawn_safe_y_min, self.spawn_safe_y_max = (
            _arena_sampling_bounds(self.arena_half_width_m, self.arena_half_length_m, ARENA_BOUNDARY_INSET_M)
        )
        self._curriculum_episodes_total: int = 0
        self._curriculum_success_rate: float = 0.0
        self.obstacle_radius_m = 0.35

        self.robot_x = self.spawn_x
        self.robot_y = self.spawn_y
        self.robot_yaw = self.spawn_yaw_rad
        self.target_x = 0.0
        self.target_y = 4.7
        tx, ty = self._clamp_xy_to_arena_interior(self.target_x, self.target_y)
        self.target_x, self.target_y = tx, ty
        self.goal_xy = np.array([self.target_x, self.target_y], dtype=np.float32)
        self.obstacles_xy: list[tuple[float, float]] = []
        self.episode_step = 0
        self.prev_goal_distance = 0.0
        self._stuck_steps = 0
        self._blocked_steps = 0
        self.latest_scan_m = np.full(self.num_lidar_beams, self.max_lidar_range, dtype=np.float32)
        self.latest_scan_norm = np.ones(self.num_lidar_beams, dtype=np.float32)

        obs_dim = self.num_lidar_beams + 2
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _sample_random_xy(self) -> tuple[float, float]:
        x = float(self.np_random.uniform(self.spawn_safe_x_min, self.spawn_safe_x_max))
        y = float(self.np_random.uniform(self.spawn_safe_y_min, self.spawn_safe_y_max))
        return x, y

    def _clamp_xy_to_arena_interior(self, x: float, y: float) -> tuple[float, float]:
        """Force a point into the same bounds used for randomization (inside red walls)."""
        return (
            float(np.clip(x, self.spawn_safe_x_min, self.spawn_safe_x_max)),
            float(np.clip(y, self.spawn_safe_y_min, self.spawn_safe_y_max)),
        )

    def _is_position_valid(
        self,
        x: float,
        y: float,
        occupied_positions: list[tuple[float, float]],
        min_distance_m: float,
    ) -> bool:
        for occ_x, occ_y in occupied_positions:
            if math.hypot(x - occ_x, y - occ_y) < min_distance_m:
                return False
        return True

    def _sample_safe_position(
        self,
        occupied_positions: list[tuple[float, float]],
        min_distance_m: float,
        max_attempts: int = 50,
        min_distance_to_robot_m: float = 0.0,
    ) -> tuple[float, float] | None:
        attempts = max(1, int(max_attempts))
        for attempt in range(attempts):
            progress = (attempt + 1) / attempts
            if progress < 0.6:
                relax_factor = 1.0
            elif progress < 0.85:
                relax_factor = 0.85
            else:
                relax_factor = 0.7

            effective_min_distance = max(0.25, float(min_distance_m) * relax_factor)
            if min_distance_to_robot_m <= 0.0:
                effective_robot_distance = 0.0
            else:
                effective_robot_distance = max(0.35, float(min_distance_to_robot_m) * relax_factor)

            x, y = self._sample_random_xy()
            if math.hypot(x - self.spawn_x, y - self.spawn_y) < effective_robot_distance:
                continue
            if self._is_position_valid(x, y, occupied_positions, effective_min_distance):
                return x, y
        return None

    def set_training_curriculum_state(self, episodes_total: int, success_rate: float) -> None:
        self._curriculum_episodes_total = max(0, int(episodes_total))
        self._curriculum_success_rate = float(success_rate)

    def _curriculum_obstacle_count(self) -> int:
        if self._curriculum_episodes_total < CURRICULUM_FREE_EPISODES or self._curriculum_success_rate < CURRICULUM_SUCCESS_THRESHOLD:
            return 0
        return CURRICULUM_OBSTACLE_COUNT_HARD

    def _randomize_episode_layout(self) -> None:
        occupied_positions: list[tuple[float, float]] = [(self.spawn_x, self.spawn_y)]
        goal_xy = self._sample_safe_position(
            occupied_positions=occupied_positions,
            min_distance_m=max(self.goal_min_distance_m, self.min_goal_distance_m),
            min_distance_to_robot_m=max(self.goal_min_distance_m, self.min_goal_distance_m),
            max_attempts=self.randomization_max_attempts,
        )
        if goal_xy is None:
            goal_xy = (0.0, 4.6)

        gx, gy = self._clamp_xy_to_arena_interior(float(goal_xy[0]), float(goal_xy[1]))
        self.target_x, self.target_y = gx, gy
        self.goal_xy = np.array([self.target_x, self.target_y], dtype=np.float32)
        occupied_positions.append((self.target_x, self.target_y))
        self.obstacles_xy = []

        for _ in range(self._curriculum_obstacle_count()):
            obstacle_xy = self._sample_safe_position(
                occupied_positions=occupied_positions,
                min_distance_m=self.obstacle_min_distance_m,
                max_attempts=self.randomization_max_attempts,
            )
            if obstacle_xy is None:
                continue
            occupied_positions.append(obstacle_xy)
            self.obstacles_xy.append(obstacle_xy)

    def _goal_features(self) -> tuple[float, float]:
        dx = float(self.target_x - self.robot_x)
        dy = float(self.target_y - self.robot_y)
        distance = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        angle_to_goal = wrap_to_pi(target_heading - self.robot_yaw)
        return distance, angle_to_goal

    def _safe_goal_features(self) -> tuple[float, float]:
        try:
            return self._goal_features()
        except Exception:
            return 0.0, 0.0

    def _safe_min_lidar(self) -> float:
        try:
            if self.latest_scan_m.size == 0:
                return self.max_lidar_range
            return float(np.min(self.latest_scan_m))
        except Exception:
            return self.max_lidar_range

    def _safe_observation(self) -> np.ndarray:
        try:
            return self._build_observation()
        except Exception:
            shape = cast(tuple[int, ...], self.observation_space.shape)
            return np.zeros(shape, dtype=np.float32)

    def _build_info(
        self,
        *,
        event: str,
        collision: bool = False,
        success: bool = False,
        stuck: bool = False,
        terminated_reason: str = "none",
        truncated_reason: str = "none",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        distance_to_goal, angle_to_goal = self._safe_goal_features()
        min_lidar = self._safe_min_lidar()
        info: dict[str, Any] = {
            "event": event,
            "goal_xy": self.goal_xy.copy(),
            "goal": self.goal_xy.copy(),
            "goal_x": float(self.goal_xy[0]),
            "goal_y": float(self.goal_xy[1]),
            "distance_to_goal_m": float(distance_to_goal),
            "angle_to_goal_rad": float(angle_to_goal),
            "min_lidar_m": float(min_lidar),
            "collision": bool(collision),
            "success": bool(success),
            "stuck": bool(stuck),
            "terminated_reason": terminated_reason,
            "truncated_reason": truncated_reason,
            "episode_step": int(self.episode_step),
        }
        if extra:
            info.update(extra)
        for key in MANDATORY_INFO_KEYS:
            info.setdefault(key, INFO_SCHEMA_DEFAULTS[key])
        return info

    def _ray_distance(self, world_angle: float) -> float:
        step_m = 0.05
        dist = step_m
        while dist <= self.max_lidar_range:
            px = self.robot_x + math.cos(world_angle) * dist
            py = self.robot_y + math.sin(world_angle) * dist

            if (
                px <= -self.arena_half_width_m
                or px >= self.arena_half_width_m
                or py <= -self.arena_half_length_m
                or py >= self.arena_half_length_m
            ):
                return dist

            for obs_x, obs_y in self.obstacles_xy:
                if math.hypot(px - obs_x, py - obs_y) <= self.obstacle_radius_m:
                    return dist
            dist += step_m
        return self.max_lidar_range

    def _update_lidar(self) -> None:
        rays = np.full(self.num_lidar_beams, self.max_lidar_range, dtype=np.float32)
        rel_angles = np.linspace(-math.pi, math.pi, self.num_lidar_beams, endpoint=False)
        for i, rel_angle in enumerate(rel_angles):
            rays[i] = self._ray_distance(self.robot_yaw + float(rel_angle))
        # Match ROS point cloud floor: suppress nearer-than-noise readings (body / numerical chatter).
        rays = np.maximum(rays, self.lidar_point_min_xy_m).astype(np.float32)
        self.latest_scan_m = rays
        self.latest_scan_norm = np.clip(1.0 - np.log1p(rays) / np.log1p(self.max_lidar_range), 0.0, 1.0).astype(np.float32)

    def _build_observation(self) -> np.ndarray:
        distance_to_goal, angle_to_goal = self._goal_features()
        dist_norm = float(np.clip(distance_to_goal / self.max_goal_distance_norm_m, 0.0, 1.0))
        ang_norm = float(np.clip(angle_to_goal / math.pi, -1.0, 1.0))
        obs = np.concatenate(
            [
                self.latest_scan_norm.astype(np.float32),
                np.array([dist_norm, ang_norm], dtype=np.float32),
            ]
        )
        return obs.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.episode_step = 0
        self._stuck_steps = 0
        self._blocked_steps = 0
        self.robot_x = self.spawn_x
        self.robot_y = self.spawn_y
        self.robot_yaw = self.spawn_yaw_rad
        self._randomize_episode_layout()
        self._update_lidar()
        self.prev_goal_distance, _ = self._goal_features()
        obs = self._build_observation()
        info = self._build_info(
            event="reset",
            extra={
                "world_reset_ok": True,
                "pose_reset_ok": True,
                "randomization_ok": True,
                "sensor_ready": True,
            },
        )
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.episode_step += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        linear = self.min_linear_speed_mps + ((float(action[0]) + 1.0) * 0.5) * (
            self.max_linear_speed_mps - self.min_linear_speed_mps
        )
        angular_action = float(action[1])
        if abs(angular_action) < self.angular_deadband:
            angular_action = 0.0
        angular = angular_action * self.max_angular_speed_radps

        self.robot_yaw = wrap_to_pi(self.robot_yaw + angular * self.control_dt_sec)
        self.robot_x += math.cos(self.robot_yaw) * linear * self.control_dt_sec
        self.robot_y += math.sin(self.robot_yaw) * linear * self.control_dt_sec

        self._update_lidar()
        distance_to_goal, angle_to_goal = self._goal_features()
        min_lidar = float(np.min(self.latest_scan_m))
        raw_progress_delta = self.prev_goal_distance - distance_to_goal
        # Kinematic sim has no wheel odometry; approximate forward progress toward goal like |v| cos(heading error).
        fwd_speed = max(0.0, linear * math.cos(angle_to_goal))
        spinning = abs(angular) > 0.22
        if (
            min_lidar < self.stuck_lidar_below_m
            and fwd_speed < self.stuck_linvel_below_mps
            and not spinning
        ):
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        if (
            linear > 0.02
            and abs(angular) < 0.1
            and fwd_speed < self.stuck_linvel_below_mps
            and raw_progress_delta < 0.002
        ):
            self._blocked_steps += 1
        else:
            self._blocked_steps = 0

        reward = _dense_navigation_reward(
            self.prev_goal_distance,
            distance_to_goal,
            reward_progress_scale=self.reward_progress_scale,
            reward_backtrack_scale=self.reward_backtrack_scale,
            reward_progress_clip=self.reward_progress_clip,
            reward_step_penalty=self.reward_step_penalty,
            reward_angular_penalty=self.reward_angular_penalty,
            angular_magnitude=angular,
            angle_to_goal_rad=angle_to_goal,
            reward_heading_scale=self.reward_heading_scale,
            heading_speed_scale=fwd_speed,
        )

        if min_lidar < 0.30:
            safety_penalty = 10.0 * (0.30 - min_lidar) ** 2
            reward -= safety_penalty

        terminated = False
        truncated = False
        event = "running"
        collision = False
        success = False
        stuck = False
        terminated_reason = "none"
        truncated_reason = "none"

        if min_lidar < self.collision_distance_m:
            reward -= self.reward_collision_penalty
            terminated = True
            event = "collision"
            collision = True
            terminated_reason = "collision"
        elif self._stuck_steps >= self.stuck_patience_steps:
            reward -= self.reward_stuck_penalty
            terminated = True
            event = "stuck"
            stuck = True
            terminated_reason = "stuck"
        elif self._blocked_steps >= max(12, self.stuck_patience_steps * 4):
            reward -= self.reward_stuck_penalty
            terminated = True
            event = "stuck"
            stuck = True
            terminated_reason = "stuck"
        elif distance_to_goal < self.goal_threshold_m:
            reward += self.reward_goal_bonus
            terminated = True
            event = "goal_reached"
            success = True
            terminated_reason = "goal_reached"
        elif self.episode_step >= self.max_episode_steps:
            truncated = True
            event = "max_steps"
            truncated_reason = "max_steps"
            reward -= 10.0 * distance_to_goal

        if self.reward_clip_abs > 0.0:
            reward = float(np.clip(reward, -self.reward_clip_abs, self.reward_clip_abs))

        self.prev_goal_distance = distance_to_goal
        obs = self._safe_observation()
        info = self._build_info(
            event=event,
            collision=collision,
            success=success,
            stuck=stuck,
            terminated_reason=terminated_reason,
            truncated_reason=truncated_reason,
            extra={
                "forward_speed_mps": float(linear),
                "blocked_steps": int(self._blocked_steps),
            },
        )
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        return None


class TrainingMetricsCallback(BaseCallback):
    """Store per-episode metrics and counters for training analysis."""

    def __init__(self, metrics_file: Path, verbose: int = 0, curriculum_env: Any | None = None) -> None:
        super().__init__(verbose)
        self.metrics_file = metrics_file
        self.curriculum_env = curriculum_env
        self.total_episodes = 0
        self.total_collisions = 0
        self.total_stuck = 0
        self.total_goals = 0
        self.total_max_steps = 0
        self.system_events_skipped = 0
        self.stop_reason: str | None = None

    def _on_training_start(self) -> None:
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.metrics_file.exists() or self.metrics_file.stat().st_size == 0:
            with self.metrics_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timesteps",
                        "episode",
                        "event",
                        "episode_reward",
                        "episode_length",
                        "distance_to_goal_m",
                        "angle_to_goal_rad",
                        "min_lidar_m",
                        "collisions_total",
                        "goals_total",
                        "max_steps_total",
                        "success_rate",
                        "collision_rate",
                    ]
                )

    def _append_row(self, info: dict[str, Any], event: str) -> None:
        episode_info = info.get("episode", {})
        reward = float(episode_info.get("r", 0.0))
        length = int(episode_info.get("l", 0))
        success_rate = self.total_goals / max(self.total_episodes, 1)
        collision_rate = self.total_collisions / max(self.total_episodes, 1)
        try:
            with self.metrics_file.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        int(self.num_timesteps),
                        self.total_episodes,
                        event,
                        reward,
                        length,
                        float(info.get("distance_to_goal_m", 0.0)),
                        float(info.get("angle_to_goal_rad", 0.0)),
                        float(info.get("min_lidar_m", 0.0)),
                        self.total_collisions,
                        self.total_goals,
                        self.total_max_steps,
                        success_rate,
                        collision_rate,
                    ]
                )
        except OSError as exc:
            if self.verbose > 0:
                print(f"WARN: failed to append metrics CSV row: {exc}", file=sys.stderr)

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = list(self.locals.get("infos", []))
        dones = list(self.locals.get("dones", []))
        if not infos or not dones:
            return True

        for done, info in zip(dones, infos):
            if not done:
                continue

            event = str(
                info.get("event")
                or info.get("terminated_reason")
                or info.get("truncated_reason")
                or "unknown"
            )

            if event in FATAL_TRAINING_EVENTS:
                self.system_events_skipped += 1
                self.stop_reason = event
                if self.verbose > 0:
                    print(
                        f"WARN: terminal system event '{event}' received. "
                        "Stopping training to avoid polluted episode metrics.",
                        file=sys.stderr,
                    )
                try:
                    self.logger.record("custom/system_events_skipped", self.system_events_skipped)
                except Exception:
                    pass
                return False

            self.total_episodes += 1

            if event == "collision":
                self.total_collisions += 1
            elif event == "stuck":
                self.total_stuck += 1
            elif event == "goal_reached":
                self.total_goals += 1
            elif event == "max_steps":
                self.total_max_steps += 1

            if self.curriculum_env is not None:
                sr = self.total_goals / max(self.total_episodes, 1)
                try:
                    self.curriculum_env.set_training_curriculum_state(self.total_episodes, sr)
                except Exception:
                    pass

            self._append_row(info=info, event=event)
            try:
                self.logger.record("custom/episodes_total", self.total_episodes)
                self.logger.record("custom/collisions_total", self.total_collisions)
                self.logger.record("custom/stuck_total", self.total_stuck)
                self.logger.record("custom/goals_total", self.total_goals)
                self.logger.record("custom/max_steps_total", self.total_max_steps)
                self.logger.record("custom/system_events_skipped", self.system_events_skipped)
                self.logger.record("custom/success_rate", self.total_goals / max(self.total_episodes, 1))
                self.logger.record("custom/collision_rate", self.total_collisions / max(self.total_episodes, 1))
            except Exception:
                if self.verbose > 0:
                    print("WARN: custom logger.record failed; continuing training", file=sys.stderr)

        return True


def _log_message(env: gym.Env[Any, Any], message: str) -> None:
    node = getattr(env, "node", None)
    if node is None:
        print(message)
        return

    cast(Any, node).get_logger().info(message)


def create_env(args: argparse.Namespace) -> gym.Env[np.ndarray, np.ndarray]:
    backend = args.backend
    use_ros = False

    if backend == "ros":
        if not ROS_AVAILABLE:
            raise RuntimeError(
                "Selected --backend ros, but ROS 2 python modules are not available."
            ) from ROS_IMPORT_ERROR
        use_ros = True
    elif backend == "auto":
        use_ros = ROS_AVAILABLE

    if use_ros:
        return RobotEnv(
            points_topic=args.points_topic,
            odom_topic=args.odom_topic,
            cmd_vel_topic=args.cmd_vel_topic,
            world_name=args.world_name,
            num_lidar_beams=args.num_lidar_beams,
            max_linear_speed_mps=args.max_linear_speed_mps,
            max_angular_speed_radps=args.max_angular_speed_radps,
            control_dt_sec=args.control_dt_sec,
            max_episode_steps=args.max_episode_steps,
            max_linear_accel_mps2=args.max_linear_accel_mps2,
            max_angular_accel_radps2=args.max_angular_accel_radps2,
            angular_deadband=args.angular_deadband,
            reset_world_on_episode=args.reset_world_on_episode,
            obstacle_randomize_every_episodes=args.obstacle_randomize_every,
            stuck_lidar_below_m=args.stuck_lidar_below_m,
            stuck_linvel_below_mps=args.stuck_linvel_below_mps,
            stuck_patience_steps=args.stuck_patience_steps,
            randomization_max_attempts=args.randomization_max_attempts,
            goal_min_distance_m=args.goal_min_distance_m,
            obstacle_min_distance_m=args.obstacle_min_distance_m,
            gz_set_pose_timeout_ms=args.gz_set_pose_timeout_ms,
            gz_world_control_timeout_ms=args.gz_world_control_timeout_ms,
            gz_service_retries=args.gz_service_retries,
            auto_disable_world_reset_failures=args.auto_disable_world_reset_failures,
            reward_progress_scale=args.reward_progress_scale,
            reward_progress_clip=args.reward_progress_clip,
            reward_backtrack_scale=args.reward_backtrack_scale,
            reward_step_penalty=args.reward_step_penalty,
            reward_angular_penalty=args.reward_angular_penalty,
            reward_heading_scale=args.reward_heading_scale,
            reward_collision_penalty=args.reward_collision_penalty,
            reward_stuck_penalty=args.reward_stuck_penalty,
            reward_goal_bonus=args.reward_goal_bonus,
            reward_clip_abs=args.reward_clip_abs,
            arena_width_m=args.arena_width_m,
            arena_length_m=args.arena_length_m,
            spawn_x=args.spawn_x,
            spawn_y=args.spawn_y,
            spawn_z=args.spawn_z,
            spawn_yaw_rad=args.spawn_yaw,
        )

    print(
        "ROS backend is unavailable in this environment. "
        "Starting standalone backend for local training."
    )
    return StandaloneRobotEnv(
        num_lidar_beams=args.num_lidar_beams,
        max_linear_speed_mps=args.max_linear_speed_mps,
        max_angular_speed_radps=args.max_angular_speed_radps,
        control_dt_sec=args.control_dt_sec,
        max_episode_steps=args.max_episode_steps,
        randomization_max_attempts=args.randomization_max_attempts,
        goal_min_distance_m=args.goal_min_distance_m,
        obstacle_min_distance_m=args.obstacle_min_distance_m,
        stuck_lidar_below_m=args.stuck_lidar_below_m,
        stuck_linvel_below_mps=args.stuck_linvel_below_mps,
        stuck_patience_steps=args.stuck_patience_steps,
        angular_deadband=args.angular_deadband,
        reward_progress_scale=args.reward_progress_scale,
        reward_progress_clip=args.reward_progress_clip,
        reward_backtrack_scale=args.reward_backtrack_scale,
        reward_step_penalty=args.reward_step_penalty,
        reward_angular_penalty=args.reward_angular_penalty,
        reward_heading_scale=args.reward_heading_scale,
        reward_collision_penalty=args.reward_collision_penalty,
        reward_stuck_penalty=args.reward_stuck_penalty,
        reward_goal_bonus=args.reward_goal_bonus,
        reward_clip_abs=args.reward_clip_abs,
        arena_width_m=args.arena_width_m,
        arena_length_m=args.arena_length_m,
        spawn_x=args.spawn_x,
        spawn_y=args.spawn_y,
        spawn_yaw_rad=args.spawn_yaw,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for course_robot navigation.")
    parser.add_argument("--total-timesteps", type=int, default=200000, help="Total PPO timesteps.")
    parser.add_argument("--save-dir", type=str, default="models", help="Root directory where run-specific model folders are created.")
    parser.add_argument("--log-dir", type=str, default="training_logs", help="Root directory where run-specific logs are created.")
    parser.add_argument("--run-id", type=str, default="", help="Optional explicit run id. If omitted, generated from timestamp and model name.")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=20_000,
        help="Save model checkpoint every N environment steps.",
    )
    parser.add_argument("--model-name", type=str, default="ppo_course_robot", help="Saved model file name.")
    parser.add_argument("--points-topic", type=str, default="/lidar/points")
    parser.add_argument("--odom-topic", type=str, default="/model/course_robot/odometry")
    parser.add_argument(
        "--cmd-vel-topic",
        type=str,
        default="/model/course_robot/cmd_vel",
        help="Velocity command topic.",
    )
    parser.add_argument("--world-name", type=str, default="course_world")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "ros", "standalone"],
        help="Training backend: ROS/Gazebo env or standalone fallback.",
    )
    parser.add_argument("--num-lidar-beams", type=int, default=72, choices=[24, 36, 72])
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--max-linear-speed-mps", type=float, default=0.5)
    parser.add_argument("--max-angular-speed-radps", type=float, default=1.5)
    parser.add_argument("--control-dt-sec", type=float, default=0.05)
    parser.add_argument(
        "--max-linear-accel-mps2",
        type=float,
        default=1.0,
        help="Limit linear command acceleration for smoother motion.",
    )
    parser.add_argument(
        "--max-angular-accel-radps2",
        type=float,
        default=5.0,
        help="Limit angular command acceleration for smoother turns.",
    )
    parser.add_argument(
        "--angular-deadband",
        type=float,
        default=0.08,
        help="Small angular action deadband to reduce wobble near zero.",
    )
    parser.add_argument(
        "--reset-world-on-episode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use Gazebo world reset each episode. Disabled by default for faster resets; "
            "robot pose reset still runs every episode."
        ),
    )
    parser.add_argument(
        "--obstacle-randomize-every",
        type=int,
        default=20,
        help=(
            "Reposition RL obstacles via gz every N episodes. "
            "20 = default. 0 = once after first successful layout, then fixed obstacles. "
            "Use 20-50 for faster training resets with mostly static obstacles."
        ),
    )
    parser.add_argument(
        "--stuck-lidar-below-m",
        type=float,
        default=0.22,
        help="With near-zero forward speed, lidar below this for stuck_patience_steps ends episode.",
    )
    parser.add_argument(
        "--stuck-linvel-below-mps",
        type=float,
        default=0.028,
        help="Odometry |vx| below this with close lidar counts toward stuck termination.",
    )
    parser.add_argument(
        "--stuck-patience-steps",
        type=int,
        default=10,
        help="Consecutive stuck-condition steps before the episode ends as a collision.",
    )
    parser.add_argument(
        "--randomization-max-attempts",
        type=int,
        default=120,
        help="Max placement sampling attempts per entity during domain randomization.",
    )
    parser.add_argument(
        "--goal-min-distance-m",
        type=float,
        default=1.1,
        help="Minimum distance from goal to occupied positions during randomization.",
    )
    parser.add_argument(
        "--obstacle-min-distance-m",
        type=float,
        default=0.95,
        help="Minimum distance between sampled obstacle positions.",
    )
    parser.add_argument(
        "--gz-set-pose-timeout-ms",
        type=int,
        default=2000,
        help="Timeout for Gazebo /set_pose service calls, in milliseconds.",
    )
    parser.add_argument(
        "--gz-world-control-timeout-ms",
        type=int,
        default=900,
        help="Timeout for Gazebo /world/*/control service calls, in milliseconds.",
    )
    parser.add_argument(
        "--gz-service-retries",
        type=int,
        default=4,
        help="Retry count for Gazebo service calls.",
    )
    parser.add_argument(
        "--auto-disable-world-reset-failures",
        type=int,
        default=3,
        help="Disable reset_world_on_episode after this many consecutive reset failures.",
    )
    parser.add_argument(
        "--reward-progress-scale",
        type=float,
        default=65.0,
        help="Scale k_progress for reduced distance-to-goal per step (after clip).",
    )
    parser.add_argument(
        "--reward-progress-clip",
        type=float,
        default=0.12,
        help="Clip per-step distance progress before scaling; 0 disables clipping.",
    )
    parser.add_argument(
        "--reward-backtrack-scale",
        type=float,
        default=90.0,
        help="Scale k_backtrack when distance to goal increases (after same clip).",
    )
    parser.add_argument(
        "--reward-step-penalty",
        type=float,
        default=1.0,
        help="Per-step time penalty; wandering without progress should be negative overall.",
    )
    parser.add_argument(
        "--reward-angular-penalty",
        type=float,
        default=0.01,
        help="Penalty multiplier for angular velocity magnitude.",
    )
    parser.add_argument(
        "--reward-heading-scale",
        type=float,
        default=0.06,
        help="Tiny heading alignment bonus; gated by real forward speed (ROS) or toward-goal speed (standalone).",
    )
    parser.add_argument(
        "--reward-collision-penalty",
        type=float,
        default=200.0,
        help="Terminal collision penalty magnitude.",
    )
    parser.add_argument(
        "--reward-stuck-penalty",
        type=float,
        default=80.0,
        help="Terminal stuck penalty magnitude.",
    )
    parser.add_argument(
        "--reward-goal-bonus",
        type=float,
        default=2200.0,
        help="Terminal goal reached bonus (should dominate per-step shaping).",
    )
    parser.add_argument(
        "--reward-clip-abs",
        type=float,
        default=4000.0,
        help="Absolute reward clip threshold; 0 disables clipping.",
    )
    parser.add_argument("--spawn-x", type=float, default=0.0)
    parser.add_argument("--spawn-y", type=float, default=-4.6)
    parser.add_argument("--spawn-z", type=float, default=0.36)
    parser.add_argument("--spawn-yaw", type=float, default=1.5708)
    parser.add_argument(
        "--arena-width-m",
        type=float,
        default=DEFAULT_ARENA_WIDTH_M,
        help="Playable arena width (X), meters; default matches inner size in course_robot_world.sdf.",
    )
    parser.add_argument(
        "--arena-length-m",
        type=float,
        default=DEFAULT_ARENA_LENGTH_M,
        help="Playable arena length (Y), meters; default matches inner size in course_robot_world.sdf.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend in ("auto", "ros") and ROS_AVAILABLE:
        rclpy.init()

    env: gym.Env[np.ndarray, np.ndarray] | None = None
    stdout_file: Any | None = None
    stderr_file: Any | None = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        script_dir = Path(__file__).resolve().parent
        default_log_fallback = Path("/tmp/course_robot_training_logs")
        default_save_fallback = script_dir / "models"
        log_root = _resolve_writable_root(Path(args.log_dir).expanduser(), default_log_fallback, "log")
        save_root = _resolve_writable_root(Path(args.save_dir).expanduser(), default_save_fallback, "save")
        run_paths = _create_run_dirs(
            log_root=log_root,
            save_root=save_root,
            model_name=args.model_name,
            requested_run_id=args.run_id,
        )
        run_id = str(run_paths["run_id"])
        run_dir = cast(Path, run_paths["run_dir"])
        log_dir = cast(Path, run_paths["log_dir"])
        save_dir = cast(Path, run_paths["save_dir"])
        tensorboard_dir = cast(Path, run_paths["tensorboard_dir"])
        checkpoint_dir = cast(Path, run_paths["checkpoint_dir"])
        final_model_dir = cast(Path, run_paths["final_model_dir"])
        stdout_dir = cast(Path, run_paths["stdout_dir"])

        stdout_path = stdout_dir / "stdout.log"
        stderr_path = stdout_dir / "stderr.log"
        stdout_file = stdout_path.open("a", encoding="utf-8")
        stderr_file = stderr_path.open("a", encoding="utf-8")
        sys.stdout = cast(Any, _TeeStream(original_stdout, stdout_file))
        sys.stderr = cast(Any, _TeeStream(original_stderr, stderr_file))

        print(f"Training run_id: {run_id}")
        print(f"Training run_dir: {run_dir}")
        print(f"Training log_dir: {log_dir}")
        print(f"Training save_dir: {save_dir}")
        print(f"Training checkpoint_dir: {checkpoint_dir}")
        print(f"Training tensorboard_dir: {tensorboard_dir}")
        print(f"Training stdout_log: {stdout_path}")
        print(f"Training stderr_log: {stderr_path}")

        env = create_env(args)
        monitor_path = log_dir / "monitor.csv"
        metrics_path = log_dir / "episode_metrics.csv"
        args_snapshot_path = run_dir / "config_snapshot.json"
        metadata_path = save_dir / "run_metadata.json"
        git_hash = _git_hash_or_unknown(script_dir)

        _write_json(
            args_snapshot_path,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "args": vars(args),
                "mandatory_info_keys": list(MANDATORY_INFO_KEYS),
                "monitor_info_keywords": list(MONITOR_INFO_KEYWORDS),
                "log_root": str(log_root),
                "save_root": str(save_root),
                "log_dir": str(log_dir),
                "save_dir": str(save_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "tensorboard_dir": str(tensorboard_dir),
                "git_hash": git_hash,
            },
        )
        _write_json(
            metadata_path,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "model_name": args.model_name,
                "total_timesteps": int(args.total_timesteps),
                "checkpoint_freq": int(args.checkpoint_freq),
                "backend": args.backend,
                "env_params": vars(args),
                "git_hash": git_hash,
                "log_dir": str(log_dir),
                "save_dir": str(save_dir),
            },
        )

        monitored_env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=MONITOR_INFO_KEYWORDS,
        )
        metrics_callback = TrainingMetricsCallback(
            metrics_file=metrics_path,
            curriculum_env=monitored_env.env,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, int(args.checkpoint_freq)),
            save_path=str(checkpoint_dir),
            name_prefix=f"{_sanitize_name(args.model_name)}_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callback_list = CallbackList([metrics_callback, checkpoint_callback])

        learning_rate_schedule = get_linear_fn(
            start=3e-4,
            end=5e-5,
            end_fraction=1.0,
        )
        clip_range_schedule = get_linear_fn(
            start=0.15,
            end=0.08,
            end_fraction=1.0,
        )

        model = PPO(
            policy="MlpPolicy",
            env=monitored_env,
            verbose=1,
            learning_rate=learning_rate_schedule,
            n_steps=1024,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=clip_range_schedule,
            target_kl=0.015,
            tensorboard_log=str(tensorboard_dir),
        )
        model.learn(total_timesteps=args.total_timesteps, callback=callback_list)
        if metrics_callback.stop_reason is not None:
            _log_message(
                env,
                f"Training stopped early due to environment terminal event: {metrics_callback.stop_reason}.",
            )

        model_path = final_model_dir / args.model_name
        model.save(str(model_path))
        _write_json(
            metadata_path,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "model_name": args.model_name,
                "requested_timesteps": int(args.total_timesteps),
                "trained_timesteps": int(model.num_timesteps),
                "checkpoint_freq": int(args.checkpoint_freq),
                "backend": args.backend,
                "env_params": vars(args),
                "git_hash": git_hash,
                "log_dir": str(log_dir),
                "save_dir": str(save_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "final_model": f"{model_path}.zip",
                "monitor_path": str(monitor_path),
                "metrics_path": str(metrics_path),
                "tensorboard_dir": str(tensorboard_dir),
            },
        )
        _log_message(env, f"Model saved to: {model_path}.zip")
        _log_message(
            env,
            f"Training logs saved: monitor={monitor_path}, metrics={metrics_path}, "
            f"tensorboard={tensorboard_dir}, checkpoints={checkpoint_dir}, metadata={metadata_path}",
        )
    except KeyboardInterrupt:
        if env is not None:
            _log_message(env, "Training interrupted by user.")
    finally:
        if sys.stdout is not original_stdout:
            sys.stdout.flush()
            sys.stdout = original_stdout
        if sys.stderr is not original_stderr:
            sys.stderr.flush()
            sys.stderr = original_stderr
        if stdout_file is not None:
            stdout_file.close()
        if stderr_file is not None:
            stderr_file.close()
        if env is not None:
            env.close()
        if ROS_AVAILABLE and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

