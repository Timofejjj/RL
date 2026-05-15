#!/usr/bin/env python3
"""Публикует URDF в /robot_description (std_msgs/String), чтобы RViz взял модель без выбора файла."""
from __future__ import annotations

import pathlib
import re
import sys
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def resolve_mesh_uris(text: str, urdf_path: str, logger: Any) -> str:
    """
    Replace mesh filename placeholders with absolute file:// URIs so RViz resolves STL reliably.

    - Relative ``meshes/...`` (same directory layout as this repo): resolved next to the URDF file.
    - ``package://pkg/sub/path``: resolved via ament_index_python when ROS workspace is sourced.
    """
    urdf_dir = pathlib.Path(urdf_path).resolve().parent
    mesh_root = urdf_dir / "meshes"

    def meshes_relative_sub(m: re.Match[str]) -> str:
        rel = m.group(1)
        full = (mesh_root / rel).resolve()
        return f'filename="{full.as_uri()}"'

    text = re.sub(r'filename="meshes/([^"]+)"', meshes_relative_sub, text)

    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        get_package_share_directory = None

    def package_uri_sub(m: re.Match[str]) -> str:
        pkg = m.group(1)
        sub = m.group(2).lstrip("/")
        if get_package_share_directory is None:
            logger.warning(
                f'Keeping unresolved mesh URI package://{pkg}/{sub} (install python3-ament-index-python '
                "or use meshes/ paths relative to the URDF)."
            )
            return m.group(0)
        try:
            share = pathlib.Path(get_package_share_directory(pkg))
            full = (share / sub).resolve()
            return f'filename="{full.as_uri()}"'
        except LookupError as exc:
            logger.warning(f'Could not resolve package://{pkg}/{sub}: {exc}')
            return m.group(0)

    text = re.sub(r'filename="package://([^/]+)/([^"]+)"', package_uri_sub, text)
    return text


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
    text = resolve_mesh_uris(text, path, node.get_logger())
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
