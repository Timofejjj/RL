#!/usr/bin/env python3
"""
ROS 2 узел: навигация реального робота политикой PPO (Stable-Baselines3), обученной в RobotEnv.

Наблюдение совпадает с rl_train.RobotEnv._build_observation:
  - num_lidar_beams секторов лидара из PointCloud2 (робот: x вперёд, y влево);
  - нормализация дальностей: clip(1 - log1p(d)/log1p(max_lidar_range), 0, 1);
  - dist_norm = distance_to_goal / max_goal_distance_norm_m;
  - ang_norm = angle_to_goal / pi;
  - lin_norm, ang_vel_norm — по текущим командам cmd (как в среде обучения).

Цель по умолчанию: одна точка в плоскости одометрии на расстоянии goal_distance_m вперёд
от позы робота в момент «взведения» (первые валидные odom + lidar). Так вы получаете
«цель в 3 м» от стартовой конфигурации. Можно задать goal_x, goal_y в одометрии вручную.

Зависимости на Raspberry Pi: ROS 2 (rclpy), numpy, torch, stable-baselines3, sensor_msgs_py.

Пример (Linux / Pi), путь к весам как у вас в репозитории:
  ros2 run ...  # или напрямую:
  python3 ppo_real_nav.py \\
    --model \"$(ws)/course_robot/models/run_20260512_011044_ppo_course_robot_100k/final_model/ppo_course_robot_100k.zip\" \\
    --metadata \"$(ws)/course_robot/models/run_20260512_011044_ppo_course_robot_100k/run_metadata.json\" \\
    --goal-distance 3.0

Дальше подключите ESP32 / diff_drive так, чтобы подписка была на тот же cmd_vel_topic,
что в метаданных (или переопределите параметры ROS).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import tf2_ros
from rclpy.time import Time
from stable_baselines3 import PPO


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pointcloud_to_planar_rays(
    msg: PointCloud2,
    *,
    num_beams: int,
    max_range_m: float,
    lidar_min_z: float,
    lidar_max_z: float,
    lidar_point_min_xy_m: float,
) -> np.ndarray:
    rays = np.full(num_beams, max_range_m, dtype=np.float32)
    bin_width = 2.0 * math.pi / float(num_beams)
    for x, y, z in point_cloud2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True):
        if z < lidar_min_z or z > lidar_max_z:
            continue
        dxy = math.hypot(x, y)
        if dxy < lidar_point_min_xy_m or dxy > max_range_m:
            continue
        angle = math.atan2(y, x)
        idx = int((angle + math.pi) / bin_width)
        idx = max(0, min(num_beams - 1, idx))
        if dxy < rays[idx]:
            rays[idx] = dxy
    return rays


def load_run_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


class PpoRealNavNode(Node):
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path | None,
        *,
        goal_distance_m: float,
        goal_xy_odom: tuple[float, float] | None,
    ) -> None:
        super().__init__("ppo_real_nav")

        env_from_meta: dict[str, Any] = {}
        if metadata_path is not None and metadata_path.is_file():
            meta = load_run_metadata(metadata_path)
            env_from_meta = cast(dict[str, Any], meta.get("env_params", {}))
            self.get_logger().info(f"Loaded env_params from {metadata_path}")

        def meta_float(key: str, default: float) -> float:
            if key in env_from_meta:
                return float(env_from_meta[key])
            return default

        def meta_int(key: str, default: int) -> int:
            if key in env_from_meta:
                return int(env_from_meta[key])
            return default

        def meta_str(key: str, default: str) -> str:
            if key in env_from_meta:
                return str(env_from_meta[key])
            return default

        self.declare_parameter("points_topic", meta_str("points_topic", "/lidar/points"))
        self.declare_parameter("odom_topic", meta_str("odom_topic", "/model/course_robot/odometry"))
        self.declare_parameter("cmd_vel_topic", meta_str("cmd_vel_topic", "/cmd_vel"))
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("use_tf", True)

        self.declare_parameter("num_lidar_beams", meta_int("num_lidar_beams", 72))
        self.declare_parameter("max_lidar_range_m", 10.0)
        self.declare_parameter("lidar_min_z", -0.10)
        self.declare_parameter("lidar_max_z", 0.60)
        self.declare_parameter("lidar_point_min_xy_m", 0.05)

        self.declare_parameter("max_goal_distance_norm_m", 13.0)
        self.declare_parameter("goal_distance_m", float(goal_distance_m))
        self.declare_parameter("goal_success_m", 0.45)
        self.declare_parameter("emergency_stop_lidar_m", 0.22)

        self.declare_parameter("max_linear_speed_mps", meta_float("max_linear_speed_mps", 0.35))
        self.declare_parameter("min_linear_speed_mps", meta_float("min_linear_speed_mps", 0.0))
        self.declare_parameter("max_angular_speed_radps", meta_float("max_angular_speed_radps", 0.8))
        self.declare_parameter("control_dt_sec", meta_float("control_dt_sec", 0.12))
        self.declare_parameter("max_linear_accel_mps2", meta_float("max_linear_accel_mps2", 0.8))
        self.declare_parameter("max_angular_accel_radps2", meta_float("max_angular_accel_radps2", 2.5))
        self.declare_parameter("angular_deadband", meta_float("angular_deadband", 0.08))

        if goal_xy_odom is not None:
            self.declare_parameter("goal_x_odom", goal_xy_odom[0])
            self.declare_parameter("goal_y_odom", goal_xy_odom[1])
            self._fixed_goal_xy = True
        else:
            self.declare_parameter("goal_x_odom", float("nan"))
            self.declare_parameter("goal_y_odom", float("nan"))
            self._fixed_goal_xy = False

        self._points_topic = str(self.get_parameter("points_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._cmd_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._odom_frame = str(self.get_parameter("odom_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._use_tf = bool(self.get_parameter("use_tf").value)

        self._num_beams = int(self.get_parameter("num_lidar_beams").value)
        self._max_range = float(self.get_parameter("max_lidar_range_m").value)
        self._lidar_min_z = float(self.get_parameter("lidar_min_z").value)
        self._lidar_max_z = float(self.get_parameter("lidar_max_z").value)
        self._lidar_xy_min = float(self.get_parameter("lidar_point_min_xy_m").value)
        self._max_goal_norm_m = float(self.get_parameter("max_goal_distance_norm_m").value)
        self._goal_distance_m = float(self.get_parameter("goal_distance_m").value)
        self._goal_success_m = float(self.get_parameter("goal_success_m").value)
        self._emergency_m = float(self.get_parameter("emergency_stop_lidar_m").value)

        self._v_max = float(self.get_parameter("max_linear_speed_mps").value)
        self._v_min = float(self.get_parameter("min_linear_speed_mps").value)
        self._w_max = float(self.get_parameter("max_angular_speed_radps").value)
        self._dt = float(self.get_parameter("control_dt_sec").value)
        self._a_lin_max = float(self.get_parameter("max_linear_accel_mps2").value)
        self._a_ang_max = float(self.get_parameter("max_angular_accel_radps2").value)
        self._angular_deadband = float(self.get_parameter("angular_deadband").value)

        self._goal_x: float | None = None
        self._goal_y: float | None = None
        if self._fixed_goal_xy:
            gx = float(self.get_parameter("goal_x_odom").value)
            gy = float(self.get_parameter("goal_y_odom").value)
            if not (math.isnan(gx) or math.isnan(gy)):
                self._goal_x, self._goal_y = gx, gy

        self._odom_msg: Odometry | None = None
        self._latest_cloud: PointCloud2 | None = None
        self._fresh_cloud = False

        self._current_linear = 0.0
        self._current_angular = 0.0

        self._v_eps = max(1e-6, abs(self._v_max))
        self._w_eps = max(1e-6, abs(self._w_max))

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(PointCloud2, self._points_topic, self._on_cloud, sensor_qos)
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, 10)
        self._pub = self.create_publisher(Twist, self._cmd_topic, 10)

        self._tf_buffer: tf2_ros.Buffer | None = None
        self._tf_listener: tf2_ros.TransformListener | None = None
        if self._use_tf:
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.get_logger().info(f"Loading PPO model from {model_path}")
        self._model = PPO.load(str(model_path))

        period = max(self._dt, 0.02)
        self._timer = self.create_timer(period, self._control_tick)

        self.get_logger().info(
            f"ppo_real_nav: lidar={self._points_topic}, odom={self._odom_topic}, cmd={self._cmd_topic}, "
            f"dt={self._dt:.3f}s, v∈[{self._v_min},{self._v_max}] m/s, w_max={self._w_max} rad/s, beams={self._num_beams}"
        )

    def _on_cloud(self, msg: PointCloud2) -> None:
        self._latest_cloud = msg
        self._fresh_cloud = True

    def _on_odom(self, msg: Odometry) -> None:
        self._odom_msg = msg

    def _pose_xy_yaw(self) -> tuple[float, float, float] | None:
        if self._use_tf and self._tf_buffer is not None:
            try:
                t = self._tf_buffer.lookup_transform(
                    self._odom_frame,
                    self._base_frame,
                    Time(nanoseconds=0),
                )
                x = float(t.transform.translation.x)
                y = float(t.transform.translation.y)
                q = t.transform.rotation
                yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
                return x, y, yaw
            except Exception:
                pass
        if self._odom_msg is None:
            return None
        p = self._odom_msg.pose.pose.position
        q = self._odom_msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        return float(p.x), float(p.y), yaw

    def _arm_goal(self, rx: float, ry: float, yaw: float) -> None:
        if self._goal_x is not None and self._goal_y is not None:
            return
        gd = self._goal_distance_m
        self._goal_x = rx + gd * math.cos(yaw)
        self._goal_y = ry + gd * math.sin(yaw)
        self.get_logger().info(f"Goal set in odom frame: ({self._goal_x:.3f}, {self._goal_y:.3f}), distance={gd:.2f} m")

    def _goal_distance_angle(self, rx: float, ry: float, yaw: float) -> tuple[float, float]:
        assert self._goal_x is not None and self._goal_y is not None
        dx = self._goal_x - rx
        dy = self._goal_y - ry
        dist = math.hypot(dx, dy)
        target_h = math.atan2(dy, dx)
        ang = wrap_to_pi(target_h - yaw)
        return dist, ang

    def _build_obs(self, scan_m: np.ndarray, dist_m: float, ang_rad: float) -> np.ndarray:
        scan_norm = np.clip(
            1.0 - np.log1p(scan_m) / np.log1p(self._max_range),
            0.0,
            1.0,
        ).astype(np.float32)
        dist_n = float(np.clip(dist_m / self._max_goal_norm_m, 0.0, 1.0))
        ang_n = float(np.clip(ang_rad / math.pi, -1.0, 1.0))
        lin_n = float(np.clip(self._current_linear / self._v_eps, -1.0, 1.0))
        w_n = float(np.clip(self._current_angular / self._w_eps, -1.0, 1.0))
        return np.concatenate(
            [scan_norm, np.array([dist_n, ang_n, lin_n, w_n], dtype=np.float32)]
        )

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        target_linear = self._v_min + ((float(action[0]) + 1.0) * 0.5) * (self._v_max - self._v_min)
        angular_action = float(action[1])
        if abs(angular_action) < self._angular_deadband:
            angular_action = 0.0
        target_angular = angular_action * self._w_max

        max_lin_d = self._a_lin_max * self._dt
        max_ang_d = self._a_ang_max * self._dt
        self._current_linear += float(
            np.clip(target_linear - self._current_linear, -max_lin_d, max_lin_d)
        )
        self._current_angular += float(
            np.clip(target_angular - self._current_angular, -max_ang_d, max_ang_d)
        )
        self._current_linear = float(
            np.clip(self._current_linear, min(self._v_min, self._v_max), max(self._v_min, self._v_max))
        )
        self._current_angular = float(np.clip(self._current_angular, -self._w_max, self._w_max))

    def _publish_zero(self) -> None:
        t = Twist()
        self._pub.publish(t)
        self._current_linear = 0.0
        self._current_angular = 0.0

    def _control_tick(self) -> None:
        if self._latest_cloud is None or not self._fresh_cloud:
            return
        pose = self._pose_xy_yaw()
        if pose is None:
            return
        rx, ry, yaw = pose

        if self._goal_x is None or self._goal_y is None:
            self._arm_goal(rx, ry, yaw)

        if self._goal_x is None or self._goal_y is None:
            return

        cloud = self._latest_cloud
        assert cloud is not None
        scan_m = pointcloud_to_planar_rays(
            cloud,
            num_beams=self._num_beams,
            max_range_m=self._max_range,
            lidar_min_z=self._lidar_min_z,
            lidar_max_z=self._lidar_max_z,
            lidar_point_min_xy_m=self._lidar_xy_min,
        )
        min_ray = float(np.min(scan_m))

        dist_m, ang_rad = self._goal_distance_angle(rx, ry, yaw)
        if dist_m <= self._goal_success_m:
            self.get_logger().info("Goal reached — stopping.")
            self._publish_zero()
            self.destroy_timer(self._timer)
            return

        if min_ray <= self._emergency_m:
            self.get_logger().warn(f"Emergency stop: min lidar {min_ray:.3f} m <= {self._emergency_m:.3f} m")
            self._publish_zero()
            return

        obs = self._build_obs(scan_m, dist_m, ang_rad)
        action, _ = self._model.predict(obs, deterministic=True)
        self._apply_action(cast(np.ndarray, action))

        cmd = Twist()
        cmd.linear.x = float(self._current_linear)
        cmd.angular.z = float(self._current_angular)
        self._pub.publish(cmd)
        self._fresh_cloud = False


def parse_cli(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO real-robot navigation (ROS 2).")
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to SB3 .zip (e.g. .../final_model/ppo_course_robot_100k.zip)",
    )
    p.add_argument(
        "--metadata",
        type=str,
        default="",
        help="Optional run_metadata.json (same run folder as model) to match training hyperparameters.",
    )
    p.add_argument("--goal-distance", type=float, default=3.0, help="Goal ahead of start pose in odom (meters).")
    p.add_argument(
        "--goal-x",
        type=float,
        default=float("nan"),
        help="If set with --goal-y, fixed goal in odom (meters); ignores --goal-distance.",
    )
    p.add_argument("--goal-y", type=float, default=float("nan"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_cli(argv)
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    meta_path: Path | None = None
    if str(args.metadata).strip():
        meta_path = Path(args.metadata).expanduser().resolve()
        if not meta_path.is_file():
            print(f"Metadata file not found: {meta_path}", file=sys.stderr)
            sys.exit(1)
    else:
        cand = model_path.parent.parent / "run_metadata.json"
        if cand.is_file():
            meta_path = cand

    goal_xy: tuple[float, float] | None = None
    if not math.isnan(args.goal_x) and not math.isnan(args.goal_y):
        goal_xy = (float(args.goal_x), float(args.goal_y))

    rclpy.init()
    node = PpoRealNavNode(
        model_path,
        meta_path,
        goal_distance_m=float(args.goal_distance),
        goal_xy_odom=goal_xy,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            z = Twist()
            node._pub.publish(z)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
