#!/usr/bin/env python3
"""Публикует URDF в /robot_description (std_msgs/String), чтобы RViz взял модель без выбора файла."""
import pathlib
import re
import sys
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def main() -> None:
    rclpy.init(args=sys.argv)
    node = Node(
        "course_robot_description_pub",
        automatically_declare_parameters_from_overrides=True,
    )
    if not node.has_parameter("urdf_path"):
        node.declare_parameter("urdf_path", "")
    path = str(node.get_parameter("urdf_path").value)
    if not path:
        node.get_logger().fatal("Задайте параметр urdf_path (абсолютный путь к .urdf)")
        raise SystemExit(1)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # RViz не всегда находит meshes/ относительно cwd — подставляем file:// от каталога URDF
    urdf_dir = pathlib.Path(path).resolve().parent
    mesh_root = urdf_dir / "meshes"

    def _mesh_uri(m):
        rel = m.group(1)
        full = (mesh_root / rel).resolve()
        return f'filename="{full.as_uri()}"'

    text = re.sub(r'filename="meshes/([^"]+)"', _mesh_uri, text)
    qos = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    pub = node.create_publisher(String, "robot_description", qos)
    msg = String(data=text)
    # Даём подписчикам время подключиться (RViz + TRANSIENT_LOCAL)
    for _ in range(40):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
    node.get_logger().info("URDF опубликован в /robot_description")

    def _republish() -> None:
        pub.publish(msg)

    node.create_timer(1.0, _republish)
    rclpy.spin(node)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
