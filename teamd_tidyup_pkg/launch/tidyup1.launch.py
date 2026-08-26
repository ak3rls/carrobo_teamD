#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""認識サービスと片付けステートマシンをまとめて起動する."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import OpaqueFunction
from launch.actions import RegisterEventHandler
from launch.actions import TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

_WORKSPACE = os.environ.get(
    'COLCON_WORKSPACE', os.path.expanduser('~/hma2_ws')
)
DEFAULT_MODEL_PATH = os.path.join(
    _WORKSPACE,
    'src',
    '5_perception',
    'hma_object_detection2',
    'hma_object_detection2',
    'models',
    'yoloe',
    'yoloe-11s-seg.pt',
)


def expand_model_path(context):
    """launch引数のモデルパスに含まれるチルダを展開する."""
    model_path = LaunchConfiguration('model_path').perform(context)
    return os.path.expanduser(model_path)


def yoloe_detection_actions(context, *args, **kwargs):
    """RC26 prompt付きYOLOE検出ノードと任意のRVizを返す."""
    return [
        Node(
            package='teamd_tidyup_pkg',
            executable='yoloe_detection_service',
            name='yolov8_detection',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'camera_info_topic': LaunchConfiguration(
                    'camera_info_topic'
                ),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'output_topic': LaunchConfiguration('output_topic'),
                'model_path': expand_model_path(context),
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=[
                '-d',
                PathJoinSubstitution([
                    FindPackageShare('yolov8_detection'),
                    'rviz',
                    'yolov8_detection.rviz',
                ]),
            ],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
    ]


def create_tidyup_sm_node():
    """Return the state-machine node with the calibrated drawer driver."""
    return Node(
        package='teamd_tidyup_pkg',
        executable='tidyup_sm1',
        name='teamd_tidyup',
        output='screen',
        # The calibrated drawer approach uses odom-frame waypoints. Keep it on
        # the deterministic odom driver even though the PUMAS bridge exposes
        # /move_base/move. Room-to-room states use NavModule and are
        # unaffected.
        additional_env={'CARROBO_BASE_DRIVER': 'odom'},
    )


def generate_launch_description():
    """片付けタスクに必要なノードの LaunchDescription を返す."""
    reset_world = ExecuteProcess(
        cmd=[
            'ros2',
            'service',
            'call',
            '/isaac/reset_world',
            'std_srvs/srv/Empty',
            '{}',
        ],
        output='screen',
        condition=IfCondition(
            LaunchConfiguration('reset_world_on_start')
        ),
    )

    start_after_reset = RegisterEventHandler(
        OnProcessExit(
            target_action=reset_world,
            on_exit=[
                TimerAction(
                    period=3.0,
                    actions=[create_tidyup_sm_node()],
                )
            ],
        ),
        condition=IfCondition(
            LaunchConfiguration('reset_world_on_start')
        ),
    )

    start_without_reset = TimerAction(
        period=3.0,
        actions=[create_tidyup_sm_node()],
        condition=UnlessCondition(
            LaunchConfiguration('reset_world_on_start')
        ),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'reset_world_on_start',
                default_value='true',
                description=(
                    'Isaacとlocalizationを同期するため、開始時にworldをresetするか'
                ),
            ),
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
            DeclareLaunchArgument(
                'model_path',
                default_value=DEFAULT_MODEL_PATH,
                description=(
                    '物体検出の重みファイル (既定: YOLOE 11s segmentation)'
                ),
            ),
            DeclareLaunchArgument(
                'image_topic',
                default_value='/head_rgbd_sensor/rgb/image_rect_color',
                description='購読するRGB画像トピック',
            ),
            DeclareLaunchArgument(
                'camera_info_topic',
                default_value='/head_rgbd_sensor/rgb/camera_info',
                description='購読するCameraInfoトピック',
            ),
            DeclareLaunchArgument(
                'depth_topic',
                default_value=(
                    '/head_rgbd_sensor/depth_registered/image_rect_raw'
                ),
                description='購読する深度画像トピック',
            ),
            DeclareLaunchArgument(
                'output_topic',
                default_value='/yolov8/output_image',
                description='検出結果画像のpublish先',
            ),
            OpaqueFunction(function=yoloe_detection_actions),
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
                condition=IfCondition(
                    LaunchConfiguration('use_yasmin_viewer')
                ),
            ),
            Node(
                package='carrobo_manipulation_pkg',
                executable='pumas_navigation_bridge',
                name='pumas_navigation_bridge',
                output='screen',
            ),
            # Register before launching the short-lived service process so its
            # exit event cannot be missed.
            start_after_reset,
            reset_world,
            start_without_reset,
        ]
    )
