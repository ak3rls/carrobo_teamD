#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""認識サービスと片付けステートマシンをまとめて起動する."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.actions import LogInfo
from launch.actions import OpaqueFunction
from launch.actions import RegisterEventHandler
from launch.actions import TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

DEFAULT_MODEL_PATH = PathJoinSubstitution([
    FindPackageShare('teamd_tidyup_pkg'),
    'models',
    '2_yolov11s_10.pt',
])
DRAWER_KNOB_MODEL_PATH = PathJoinSubstitution([
    FindPackageShare('teamd_tidyup_pkg'),
    'models',
    'yoloe-11l-seg.pt',
])

INITIAL_POSE = (
    '{header: {stamp: now, frame_id: map}, pose: {pose: {'
    'position: {x: 0.0, y: 0.0, z: 0.0}, '
    'orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}'
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
        executable='tidyup_sm',
        name='teamd_tidyup',
        output='screen',
        # The calibrated drawer approach uses odom-frame waypoints. Keep it on
        # the deterministic odom driver even though the PUMAS bridge exposes
        # /move_base/move. Room-to-room states use NavModule and are
        # unaffected.
        additional_env={
            'CARROBO_BASE_DRIVER': 'odom',
            # This launch file completes localization synchronization before
            # starting tidyup_sm. Avoid doing the same reset a second time in
            # the executable's direct-run safety path.
            'CARROBO_SKIP_LOCALIZATION_SYNC': '1',
        },
    )


def generate_launch_description():
    """片付けタスクに必要なノードの LaunchDescription を返す."""
    wait_for_localization = ExecuteProcess(
        cmd=[
            'ros2',
            'topic',
            'echo',
            '--once',
            '--qos-reliability',
            'reliable',
            '--qos-durability',
            'volatile',
            '--field',
            'header.stamp',
            '/rtabmap/info',
            'rtabmap_msgs/msg/Info',
        ],
        output='screen',
        condition=IfCondition(
            LaunchConfiguration('reset_world_on_start')
        ),
    )

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

    republish_initial_pose = ExecuteProcess(
        cmd=[
            'ros2',
            'topic',
            'pub',
            '--times',
            '3',
            '--rate',
            '5',
            '--wait-matching-subscriptions',
            '1',
            '--qos-reliability',
            'reliable',
            '--qos-durability',
            'volatile',
            '--use-sim-time',
            '--print',
            '0',
            '/initialpose',
            'geometry_msgs/msg/PoseWithCovarianceStamped',
            INITIAL_POSE,
        ],
        output='screen',
        condition=IfCondition(
            LaunchConfiguration('reset_world_on_start')
        ),
    )

    reset_after_localization_ready = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_localization,
            on_exit=[
                LogInfo(
                    msg=(
                        'RTAB-Map is ready. Resetting Isaac and localization.'
                    )
                ),
                reset_world,
            ],
        ),
        condition=IfCondition(
            LaunchConfiguration('reset_world_on_start')
        ),
    )

    republish_pose_after_reset = RegisterEventHandler(
        OnProcessExit(
            target_action=reset_world,
            on_exit=[
                TimerAction(
                    period=3.0,
                    actions=[
                        LogInfo(
                            msg=(
                                'Fresh odometry is ready. Republishing the '
                                'initial pose.'
                            )
                        ),
                        republish_initial_pose,
                    ],
                )
            ],
        ),
        condition=IfCondition(
            LaunchConfiguration('reset_world_on_start')
        ),
    )

    start_after_pose_sync = RegisterEventHandler(
        OnProcessExit(
            target_action=republish_initial_pose,
            on_exit=[
                TimerAction(
                    period=2.0,
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
                'use_rex_omni',
                default_value='false',
                description=(
                    'YOLO未検出時のRex-Omni最終確認を有効にするか。'
                    'recog.py の USE_REX_OMNI も True にする必要があります。'
                ),
            ),
            DeclareLaunchArgument(
                'rex_omni_weights_dir',
                default_value='',
                description=(
                    'Rex-Omni weightsディレクトリ。空なら既定位置を使用する'
                ),
            ),
            DeclareLaunchArgument(
                'model_path',
                default_value=DEFAULT_MODEL_PATH,
                description=(
                    '物体検出の重みファイル (既定: 2_yolov11s_10.pt)'
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
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('hma_object_detection2'),
                        'launch',
                        'hsr_head_rgbd_rex_omni_sam2_service.launch.py',
                    ])
                ),
                launch_arguments={
                    'rex_omni_action_name': '/rex_omni/infer',
                    'image_topic': LaunchConfiguration('image_topic'),
                    'depth_topic': LaunchConfiguration('depth_topic'),
                    'camera_info_topic': LaunchConfiguration(
                        'camera_info_topic'
                    ),
                    'image_is_compressed': 'false',
                    'depth_is_compressed': 'false',
                    'default_prompt': 'object',
                    'rex_omni_weights_dir': LaunchConfiguration(
                        'rex_omni_weights_dir'
                    ),
                }.items(),
                condition=IfCondition(LaunchConfiguration('use_rex_omni')),
            ),
            Node(
                package='teamd_tidyup_pkg',
                executable='drawer_knob_detection_service',
                name='drawer_knob_detection',
                output='screen',
                parameters=[{
                    'image_topic': LaunchConfiguration('image_topic'),
                    'camera_info_topic': LaunchConfiguration(
                        'camera_info_topic'
                    ),
                    'depth_topic': LaunchConfiguration('depth_topic'),
                    'output_topic': '/drawer_knob/output_image',
                    'model_path': DRAWER_KNOB_MODEL_PATH,
                }],
                # The base class always creates yolov8_detection/service, so
                # the knob detector has to be remapped off the object
                # detector's name.
                remappings=[(
                    'yolov8_detection/service',
                    '/drawer_knob_detection/service',
                )],
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
            # Register all process-exit handlers before starting the
            # short-lived readiness check so no exit event can be missed.
            start_after_pose_sync,
            republish_pose_after_reset,
            reset_after_localization_ready,
            LogInfo(
                msg='Waiting for RTAB-Map localization before world reset...',
                condition=IfCondition(
                    LaunchConfiguration('reset_world_on_start')
                ),
            ),
            wait_for_localization,
            start_without_reset,
        ]
    )
