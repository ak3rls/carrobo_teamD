#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""objectlist.yaml の YOLOE と tidy_sm2 を起動する片付けランチャー."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """既存の片付け起動処理を YOLOE 構成で再利用する."""
    package_share = FindPackageShare('teamd_tidyup_pkg')
    declarations = [
        DeclareLaunchArgument(
            'reset_world_on_start',
            default_value='true',
            description='開始時に Isaac と localization を同期するか',
        ),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        DeclareLaunchArgument('use_yasmin_viewer', default_value='true'),
        DeclareLaunchArgument('use_rex_omni', default_value='false'),
        DeclareLaunchArgument('rex_omni_weights_dir', default_value=''),
        DeclareLaunchArgument(
            'model_path',
            default_value=PathJoinSubstitution([
                package_share,
                'models',
                'yoloe-11l-seg.pt',
            ]),
            description='YOLOE の重みファイル',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/head_rgbd_sensor/rgb/image_rect_color',
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/head_rgbd_sensor/rgb/camera_info',
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/head_rgbd_sensor/depth_registered/image_rect_raw',
        ),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/yoloe_detection/output_image',
        ),
    ]
    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                package_share,
                'launch',
                'tidyup.launch.py',
            ])
        ),
        launch_arguments={
            'detection_node_name': 'yoloe_detection',
            'detection_service_name': '/yoloe_detection/service',
            'state_machine_executable': 'tidy_sm2',
            'model_path': LaunchConfiguration('model_path'),
            'output_topic': LaunchConfiguration('output_topic'),
            'use_rex_omni': LaunchConfiguration('use_rex_omni'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'use_yasmin_viewer': LaunchConfiguration('use_yasmin_viewer'),
            'reset_world_on_start': LaunchConfiguration(
                'reset_world_on_start'
            ),
            'image_topic': LaunchConfiguration('image_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'rex_omni_weights_dir': LaunchConfiguration(
                'rex_omni_weights_dir'
            ),
        }.items(),
    )
    return LaunchDescription(declarations + [include])
