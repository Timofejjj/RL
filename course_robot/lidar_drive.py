#!/usr/bin/env python3
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class LidarDrive(Node):
    def __init__(self) -> None:
        super().__init__("lidar_drive")

        self.declare_parameter("points_topic", "/lidar/points")
        self.declare_parameter("cmd_vel_topic", "/model/course_robot/cmd_vel")
        self.declare_parameter("clear_distance_m", 5.0)
        self.declare_parameter("min_z_m", -0.05)
        self.declare_parameter("linear_speed_mps", 0.6)
        self.declare_parameter("avoid_linear_speed_mps", 0.08)
        self.declare_parameter("turn_speed_radps", 0.8)
        self.declare_parameter("front_half_angle_deg", 35.0)
        self.declare_parameter("publish_hz", 10.0)

        self._points_topic = (
            self.get_parameter("points_topic").get_parameter_value().string_value
        )
        self._cmd_vel_topic = (
            self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        )

        self._clear_distance_m = (
            self.get_parameter("clear_distance_m").get_parameter_value().double_value
        )
        self._min_z_m = self.get_parameter("min_z_m").get_parameter_value().double_value
        self._linear_speed_mps = (
            self.get_parameter("linear_speed_mps").get_parameter_value().double_value
        )
        self._avoid_linear_speed_mps = (
            self.get_parameter("avoid_linear_speed_mps")
            .get_parameter_value()
            .double_value
        )
        self._turn_speed_radps = (
            self.get_parameter("turn_speed_radps").get_parameter_value().double_value
        )
        self._front_half_angle_rad = math.radians(
            self.get_parameter("front_half_angle_deg")
            .get_parameter_value()
            .double_value
        )
        self._publish_hz = self.get_parameter("publish_hz").get_parameter_value().double_value

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._sub = self.create_subscription(
            PointCloud2, self._points_topic, self._on_points, sensor_qos
        )
        self._pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)

        self._is_clear = False
        self._last_min_dist = float("inf")
        self._front_min_dist = float("inf")
        self._left_min_dist = float("inf")
        self._right_min_dist = float("inf")
        self._turn_sign = 1.0

        period_s = 1.0 / max(self._publish_hz, 1e-3)
        self._timer = self.create_timer(period_s, self._publish_cmd)

        self.get_logger().info(
            f"Subscribing: {self._points_topic}; publishing: {self._cmd_vel_topic}; "
            f"clear_distance={self._clear_distance_m:.2f}m; min_z={self._min_z_m:.2f}m; "
            f"avoid_v={self._avoid_linear_speed_mps:.2f}mps; turn_w={self._turn_speed_radps:.2f}rad/s"
        )

    def _on_points(self, msg: PointCloud2) -> None:
        min_dist = float("inf")
        front_min = float("inf")
        left_min = float("inf")
        right_min = float("inf")

        for x, y, z in point_cloud2.read_points(
            msg, field_names=["x", "y", "z"], skip_nans=True
        ):
            if z < self._min_z_m:
                continue

            d = math.hypot(x, y)
            if d < min_dist:
                min_dist = d

            # React only to points in front hemisphere for navigation decisions.
            if x <= 0.0:
                continue

            if y >= 0.0:
                if d < left_min:
                    left_min = d
            else:
                if d < right_min:
                    right_min = d

            if abs(math.atan2(y, x)) <= self._front_half_angle_rad and d < front_min:
                front_min = d

        self._last_min_dist = min_dist
        self._front_min_dist = front_min
        self._left_min_dist = left_min
        self._right_min_dist = right_min
        self._is_clear = front_min > self._clear_distance_m

        if not self._is_clear:
            if left_min > right_min:
                self._turn_sign = 1.0
            elif right_min > left_min:
                self._turn_sign = -1.0

    def _publish_cmd(self) -> None:
        cmd = Twist()
        if self._is_clear:
            cmd.linear.x = float(self._linear_speed_mps)
            cmd.angular.z = 0.0
        else:
            cmd.linear.x = float(self._avoid_linear_speed_mps)
            cmd.angular.z = float(self._turn_sign * self._turn_speed_radps)

        self._pub.publish(cmd)


def main() -> None:
    rclpy.init(args=sys.argv)
    node = LidarDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

