#!/usr/bin/env python3
"""
Publish TF from nav_msgs/Odometry.

Needed because Gazebo->ROS TF bridging can differ in frame naming, and RViz needs a complete chain
to the fixed frame (world). We use:
  parent: course_robot_odom
  child:  course_robot_base_link
and take pose from /model/course_robot/odometry.

Do not bridge gz /model/.../tf into ROS /tf while publishing this transform: tf2 allows only one
parent per frame; duplicate trees cause intermittent RViz transform errors.
"""

import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from tf2_ros import TransformBroadcaster


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat_z_w(yaw_rad: float) -> tuple[float, float]:
    return math.sin(yaw_rad * 0.5), math.cos(yaw_rad * 0.5)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class OdomToTf(Node):
    def __init__(self) -> None:
        super().__init__(
            "course_robot_odom_to_tf",
            automatically_declare_parameters_from_overrides=True,
        )
        defaults = (
            ("odom_topic", "/model/course_robot/odometry"),
            ("reset_pose_topic", "/course_robot/reset_pose"),
            ("parent_frame", "course_robot_odom"),
            ("child_frame", "course_robot_base_link"),
            ("use_sim_time", True),
        )
        for name, value in defaults:
            if not self.has_parameter(name):
                self.declare_parameter(name, value)

        self._parent = str(self.get_parameter("parent_frame").value)
        self._child = str(self.get_parameter("child_frame").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        reset_pose_topic = str(self.get_parameter("reset_pose_topic").value)
        self._reset_target: tuple[float, float, float, float] | None = None
        self._odom_reference: tuple[float, float, float] | None = None

        self._br = TransformBroadcaster(self)
        self._sub = self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self._reset_sub = self.create_subscription(PoseStamped, reset_pose_topic, self._on_reset_pose, 10)
        self.get_logger().info(
            f"Publishing TF {self._parent} -> {self._child} from {odom_topic}; reset topic={reset_pose_topic}"
        )

    def _on_reset_pose(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        self._reset_target = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            quat_to_yaw(q.x, q.y, q.z, q.w),
        )
        self._odom_reference = None
        self.get_logger().info(
            f"Received TF reset target: x={self._reset_target[0]:.3f}, y={self._reset_target[1]:.3f}"
        )

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        odom_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        if self._reset_target is not None and self._odom_reference is None:
            self._odom_reference = (float(p.x), float(p.y), odom_yaw)

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self._parent
        t.child_frame_id = self._child

        if self._reset_target is None or self._odom_reference is None:
            t.transform.translation.x = p.x
            t.transform.translation.y = p.y
            t.transform.translation.z = p.z
            t.transform.rotation = q
        else:
            target_x, target_y, target_z, target_yaw = self._reset_target
            ref_x, ref_y, ref_yaw = self._odom_reference
            corrected_yaw = wrap_to_pi(target_yaw + odom_yaw - ref_yaw)
            qz, qw = yaw_to_quat_z_w(corrected_yaw)
            t.transform.translation.x = target_x + (float(p.x) - ref_x)
            t.transform.translation.y = target_y + (float(p.y) - ref_y)
            t.transform.translation.z = target_z
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
        self._br.sendTransform(t)


def main(argv: list[str] | None = None) -> None:
    rclpy.init(args=argv if argv is not None else sys.argv)
    node = OdomToTf()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
