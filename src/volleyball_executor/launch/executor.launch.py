#!/usr/bin/env python3
"""Launch the vision-to-execution bridge (no camera / YOLO)."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("volleyball_executor").find("volleyball_executor")
    default_params = os.path.join(pkg_share, "config", "executor_params.yaml")

    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Executor bridge yaml",
            ),
            Node(
                package="volleyball_executor",
                executable="intercept_bridge_node",
                name="intercept_bridge_node",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
