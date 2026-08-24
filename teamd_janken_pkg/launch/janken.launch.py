#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""認識サービスと片付けステートマシンをまとめて起動する."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """片付けタスクに必要なノードの LaunchDescription を返す."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'use_rviz',
                default_value='false',
                description='YOLOv8 の RViz を起動するか',
            ),
            DeclareLaunchArgument(
                'use_yasmin_viewer',
                default_value='true',
                description='YASMIN Viewer を起動するか',
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare('yolov8_detection'),
                            'launch',
                            'yolov8_detection_launch.py',
                        ]
                    )
                ),
                launch_arguments={
                    'use_rviz': LaunchConfiguration('use_rviz'),
                }.items(),
            ),
            Node(
                package='grasp_point_detection',
                executable='grasp_point_service',
                name='grasp_point_detection',
                output='screen',
            ),
            Node(
                package='yasmin_viewer',
                executable='yasmin_viewer_node',
                name='yasmin_viewer',
                output='screen',
                condition=IfCondition(LaunchConfiguration('use_yasmin_viewer')),
            ),
            Node(
                package='teamd_janken_pkg',
                executable='janken_sm',
                name='teamd_janken',
                output='screen',
            ),
        ]
    )
