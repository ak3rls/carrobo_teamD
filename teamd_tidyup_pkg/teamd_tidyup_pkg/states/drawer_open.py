#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-route drawer-opening task for the Isaac Carrobo competition scene.

The robot starts at the arena entrance, drives forward, turns right, then
drives to the drawer.  The handle pose is fixed relative to the stopped robot;
this task deliberately does not use YOLO or another detector.
"""

import math
import time

import rclpy
import tf2_ros
import tf_transformations as tft
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State

from carrobo_manipulation_pkg.hsrif import HSRInterfaces


DRAWER_PARAMETER_DEFAULTS = {
    'first_forward_distance': 1.10,
    'right_turn_degrees': 90.0,
    'second_forward_distance': 0.90,
    'navigation_timeout': 60.0,
    'require_entrance_start': True,
    'entrance_tolerance': 0.25,
    'handle_x': 0.88,
    'handle_y': 0.0,
    'handle_z': 0.54,
    'pregrasp_standoff': 0.10,
    'engage_distance': 0.02,
    'engage_step': 0.01,
    'pull_distance': 0.20,
    'upper_pull_distance': 0.22,
    'base_pull_speed': 0.08,
    'left_shift_speed': 0.12,
    'upper_forward_distance': 0.17,
    'pull_step': 0.04,
    'retreat_distance': 0.10,
    'preparation_arm_lift': 0.0,
    'preparation_arm_flex': -1.5708,
    'preparation_wrist_flex': 0.0,
    'preparation_wrist_roll': 0.10,
    'drawer_level': 'bottom',
    'upper_arm_lift': 0.28,
    'preparation_forward_distance': 0.08,
    'gripper_open_angle': 0.90,
    'gripper_open_duration': 0.70,
    'gripper_close_force': 0.40,
    'gripper_close_duration': 2.0,
    'gripper_hold_force': 0.40,
    'base_motion_weight': 100.0,
    'route_only': False,
    'prepare_only': False,
    'contact_only': False,
    'verify_grasp': False,
    'pull_after_close': True,
    'open_upper_after_lower': True,
    'open_left_drawer_after_upper': True,
    'left_drawer_shift_distance': 0.37,
    'left_drawer_forward_after_arm_down_distance': 0.21,
    'left_drawer_close_angle': -0.10,
    'left_drawer_grasp_settle_seconds': 0.50,
    'calibration_hold_seconds': 5.0,
    # After drawer 3, clear the cabinet before navigation takes control.
    'post_task_retreat_distance': 0.10,
    'post_task_retreat_speed': 0.08,
}


def launch_default(value) -> str:
    """Convert a typed default into ROS launch argument text."""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class DrawerOpenTask:
    """Reusable drawer task that runs on an existing ROS node and HSR interface."""

    MIN_FINGERTIP_DISTANCE = 0.004
    MAX_HELD_FINGERTIP_DISTANCE = 0.075

    def __init__(
        self,
        node: Node,
        hsrif: HSRInterfaces = None,
        tf_buffer=None,
    ) -> None:
        """Attach the task to shared ROS and robot interfaces."""
        self.node = node
        self._declare_parameters()
        if tf_buffer is None:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(
                self.tf_buffer, self.node
            )
        else:
            self.tf_buffer = tf_buffer
            self.tf_listener = None
        # Direct wheel command used only for the straight backward drawer pull.
        self.base_cmd_pub = self.node.create_publisher(
            Twist, '/omni_base_controller/cmd_vel', 10
        )
        self.hsrif = hsrif

    def _declare_parameters(self) -> None:
        """Declare the same calibrated parameters for standalone and SM use."""
        for name, default in DRAWER_PARAMETER_DEFAULTS.items():
            self.node.declare_parameter(name, default)

    def get_logger(self):
        """Return the logger owned by the shared ROS node."""
        return self.node.get_logger()

    def stop_base(self) -> None:
        """Publish an explicit zero velocity during cleanup or handoff."""
        self.base_cmd_pub.publish(Twist())

    def _value(self, name: str):
        return self.node.get_parameter(name).value

    def _base_pose(self):
        """Return the current odom pose of base_footprint."""
        last_error = None
        for _ in range(30):
            try:
                transform = self.tf_buffer.lookup_transform(
                    'odom', 'base_footprint', rclpy.time.Time()
                )
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                yaw = math.atan2(
                    2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
                    1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
                )
                return translation.x, translation.y, yaw
            except Exception as error:
                last_error = error
                rclpy.spin_once(self.node, timeout_sec=0.1)
        raise RuntimeError(f'odom/base_footprint TF を取得できません: {last_error}')

    def _go_abs(self, x: float, y: float, yaw: float, label: str) -> None:
        """Drive to one route waypoint and fail if navigation reports failure."""
        self.get_logger().info(
            f'{label}: odom goal=(x={x:.3f}, y={y:.3f}, '
            f'yaw={math.degrees(yaw):.1f} deg)'
        )
        result = self.hsrif.omni_base.go_abs(
            x, y, yaw, timeout=float(self._value('navigation_timeout')), sync=True
        )
        if result is False:
            raise RuntimeError(f'{label} へ到達できませんでした。')

    def _drive_from_entrance(self) -> None:
        """Forward -> right turn -> forward from the initial odom pose."""
        start_x, start_y, start_yaw = self._base_pose()
        self.get_logger().info(
            f'開始位置: odom=(x={start_x:.3f}, y={start_y:.3f}, '
            f'yaw={math.degrees(start_yaw):.1f} deg)'
        )
        if bool(self._value('require_entrance_start')):
            tolerance = float(self._value('entrance_tolerance'))
            if math.hypot(start_x, start_y) > tolerance or abs(start_yaw) > 0.20:
                raise RuntimeError(
                    'ロボットが入口の初期位置にいません。Isaac Sim をリセットしてから再実行してください。'
                )
        first = float(self._value('first_forward_distance'))
        first_x = start_x + first * math.cos(start_yaw)
        first_y = start_y + first * math.sin(start_yaw)
        self._go_abs(first_x, first_y, start_yaw, '1/3 前進')
        turn_yaw = wrap_angle(
            start_yaw - math.radians(float(self._value('right_turn_degrees')))
        )
        self._go_abs(first_x, first_y, turn_yaw, '2/3 右回転')
        second = float(self._value('second_forward_distance'))
        second_x = first_x + second * math.cos(turn_yaw)
        second_y = first_y + second * math.sin(turn_yaw)
        self._go_abs(second_x, second_y, turn_yaw, '3/3 前進')
        self.get_logger().info(
            f'経路ゴール送信済み: odom=(x={second_x:.3f}, y={second_y:.3f}, '
            f'yaw={math.degrees(turn_yaw):.1f} deg)'
        )

    def _pregrasp_pose(self) -> Pose:
        """Return the fixed safe standoff pose in base_link."""
        pose = Pose()
        pose.position = Point(
            x=float(self._value('handle_x')) - float(self._value('pregrasp_standoff')),
            y=float(self._value('handle_y')),
            z=float(self._value('handle_z')),
        )
        # Hand local +Z points toward drawer; local -Z pulls it back.
        qx, qy, qz, qw = tft.quaternion_from_euler(math.pi, math.pi / 2.0, 0.0)
        pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        return pose

    def _prepare_straight_arm(
        self,
        drawer_level: str,
        close_hand: bool = True,
        base_forward_after_posture: float = 0.0,
        base_forward_is_final_approach: bool = False,
        arm_already_neutral: bool = False,
        close_command_override=None,
        grasp_settle_seconds: float = 0.0,
    ) -> None:
        """Point arm forward, advance its open hand, then optionally close it."""
        if drawer_level == 'upper':
            arm_lift = float(self._value('upper_arm_lift'))
            wrist_roll = float(self._value('preparation_wrist_roll'))
            self.get_logger().info(
                'upper drawer mode: 手首を回転せず、開いたハンドで腕を上段高さまで上げます。'
            )
        elif drawer_level == 'bottom':
            arm_lift = float(self._value('preparation_arm_lift'))
            wrist_roll = float(self._value('preparation_wrist_roll'))
        else:
            raise RuntimeError("drawer_level は 'bottom' または 'upper' にしてください。")
        self.get_logger().info(
            'arm_flex_joint を -90 度の前方姿勢にし、ハンドを開きます。'
        )
        # After the lower drawer, keep the current straight posture.  The
        # upper-drawer command below only raises/repositions the arm; sending
        # the robot back to neutral here would add an unwanted arm motion.
        if drawer_level == 'bottom' and not arm_already_neutral:
            self.hsrif.whole_body.move_to_neutral(sync=True)
        elif drawer_level == 'bottom':
            self.get_logger().info(
                'left drawer: 腕はすでに neutral のため重複動作を省略します。'
            )
        self.hsrif.gripper.command(
            float(self._value('gripper_open_angle')),
            float(self._value('gripper_open_duration')),
        )
        self.hsrif.whole_body.move_to_joint_positions(
            {
                'arm_lift_joint': arm_lift,
                'arm_flex_joint': float(self._value('preparation_arm_flex')),
                'arm_roll_joint': 0.0,
                'wrist_flex_joint': float(self._value('preparation_wrist_flex')),
                'wrist_roll_joint': wrist_roll,
            },
            sync=True,
        )
        if base_forward_after_posture > 0.0:
            self.get_logger().info(
                'left drawer: ハンドを開いて腕を下段姿勢にした後、台車を前進します。'
            )
            self._drive_base_straight(
                base_forward_after_posture,
                float(self._value('base_pull_speed')),
                'left drawer: 把持前の台車前進',
            )
            if base_forward_is_final_approach:
                self.get_logger().info(
                    'left drawer: 台車前進で取っ手に到達したため、'
                    '追加のハンド前進は行いません。'
                )
        forward = float(self._value('preparation_forward_distance'))
        # The upper drawer uses one motion containing the former base
        # correction, normal approach, and final handle insertion.
        upper_single_approach = drawer_level == 'upper' and close_hand
        if upper_single_approach:
            forward = max(0.0, float(self._value('upper_forward_distance')))
        elif base_forward_is_final_approach:
            forward = 0.0
        if forward > 0.0:
            self.hsrif.whole_body.move_end_effector_by_line(
                (0, 0, 1), forward, sync=True
            )
            if upper_single_approach:
                self.get_logger().info(
                    f'upper drawer: 腕を上げた後、開いたハンドを一度で '
                    f'{forward:.3f} m 前方へ動かしました。'
                )
            else:
                self.get_logger().info(
                    f'開いたハンドを前方へ {forward:.3f} m 動かしました。'
                )
        if close_hand:
            if not upper_single_approach and not base_forward_is_final_approach:
                try:
                    self._approach_open_handle()
                except Exception as error:
                    # Do not discard the calibrated original approach if the extra
                    # 2 cm would collide in a particular simulator reset.
                    self.get_logger().warn(
                        f'取っ手への追加接近を行えません。現在位置で閉じます: {error}'
                    )
            self.get_logger().info('取っ手を挟むため、調整済みのコマンドでハンドを閉じます。')
            close_command = (
                float(self._value('gripper_close_force'))
                if close_command_override is None
                else float(close_command_override)
            )
            self.hsrif.gripper.command(
                close_command,
                float(self._value('gripper_close_duration')),
            )
            try:
                self.hsrif.gripper.apply_force(
                    float(self._value('gripper_hold_force')),
                    delicate=True,
                    sync=False,
                )
            except Exception as error:
                # The position-close still holds in simulators that do not
                # provide the optional force-control action.
                self.get_logger().warn(f'把持力制御を使えません。位置閉じで継続します: {error}')
            settle = max(0.0, float(grasp_settle_seconds))
            if settle > 0.0:
                self.get_logger().info(
                    f'ハンドを閉じたまま {settle:.2f} 秒待ってから台車を後退します。'
                )
                time.sleep(settle)
            self.get_logger().info('ハンドを閉じました。把持姿勢を保持します。')

    def _approach_open_handle(self) -> None:
        """Advance to the handle in small visible increments with fingers open."""
        remaining = float(self._value('engage_distance'))
        step = max(0.01, float(self._value('engage_step')))
        while remaining > 1e-6:
            distance = min(step, remaining)
            self.hsrif.whole_body.move_end_effector_by_line(
                (0, 0, 1), distance, sync=True
            )
            remaining -= distance
            self.get_logger().info(f'開いた指を取っ手へ {distance:.3f} m 近づけました。')

    def _gripper_holds_handle(self) -> bool:
        """Confirm that the fingers did not close on empty space."""
        time.sleep(0.25)
        distance = self.hsrif.gripper.get_distance()
        self.get_logger().info(f'取っ手把持確認: 指先距離={distance:.3f} m')
        return self.MIN_FINGERTIP_DISTANCE <= distance <= self.MAX_HELD_FINGERTIP_DISTANCE

    def _drive_base_straight(
        self, distance: float, speed: float, label: str, axis: str = 'x'
    ) -> None:
        """Drive along one current base axis for a calibrated duration."""
        if distance <= 0.0:
            return
        if axis not in ('x', 'y'):
            raise ValueError("axis は 'x' または 'y' にしてください。")
        speed = math.copysign(max(0.01, abs(speed)), speed)
        duration = distance / abs(speed)
        command = Twist()
        if axis == 'x':
            command.linear.x = speed
        else:
            command.linear.y = speed
        self.get_logger().info(
            f'{label}: 回転せず、台車を {distance:.3f} m '
            f'({abs(speed):.3f} m/s, {duration:.2f} 秒) 動かします。'
        )
        started = time.monotonic()
        try:
            # Isaac's direct cmd_vel path can leave odometry stale for this
            # node.  Use calibrated speed × time so the stop is guaranteed.
            while rclpy.ok() and time.monotonic() - started < duration:
                # Linear velocity is in base_link. angular.z remains exactly
                # zero, so this command cannot intentionally rotate the robot.
                self.base_cmd_pub.publish(command)
                rclpy.spin_once(self.node, timeout_sec=0.05)
        finally:
            self.base_cmd_pub.publish(Twist())

    def _pull_drawer(self, drawer_level: str) -> None:
        """Pull straight backward without issuing a turn or arm command."""
        distance_parameter = (
            'upper_pull_distance' if drawer_level == 'upper' else 'pull_distance'
        )
        distance = max(0.0, float(self._value(distance_parameter)))
        speed = float(self._value('base_pull_speed'))
        self._drive_base_straight(
            distance, -speed, f'{drawer_level} drawer: 引き出しを引くため台車後退'
        )
        self.get_logger().info('台車後退による引き出し操作が完了しました。')

    def _move_to_left_drawer(self) -> None:
        """Safely align with the single lower-height drawer on the left."""
        speed = float(self._value('left_shift_speed'))
        left = max(0.0, float(self._value('left_drawer_shift_distance')))
        self.get_logger().info(
            'left drawer: 上段の後、腕を収納して左側の引き出しへ移動します。'
        )
        self.hsrif.whole_body.move_to_neutral(sync=True)
        # ROS base_link uses +Y for robot-left.  angular.z remains zero.
        self._drive_base_straight(
            left, speed, 'left drawer: 台車を左へ移動', axis='y'
        )
        self.get_logger().info(
            'left drawer: 1番目の下段と全く同じ手順で把持を開始します。'
        )

    def _release_and_retreat(self, return_to_go: bool = True) -> None:
        """Leave the drawer front clear after a pull or unsuccessful grasp."""
        self.hsrif.gripper.command(
            1.0, float(self._value('gripper_open_duration'))
        )
        self.hsrif.whole_body.move_end_effector_by_line(
            (0, 0, -1), float(self._value('retreat_distance')), sync=True
        )
        if return_to_go:
            self.hsrif.whole_body.move_to_go(sync=True)

    def _release_gripper_only(self) -> None:
        """Release the handle without moving the base, arm, or wrist."""
        self.hsrif.gripper.command(
            1.0, float(self._value('gripper_open_duration'))
        )
        self.get_logger().info('台車と腕の姿勢を保ったまま、ハンドを開いて取っ手を離しました。')

    def _prepare_for_navigation(self) -> None:
        """Clear the final drawer and fold the robot into its travel posture."""
        distance = max(
            0.0, float(self._value('post_task_retreat_distance'))
        )
        speed = abs(float(self._value('post_task_retreat_speed')))
        if distance > 0.0:
            self._drive_base_straight(
                distance,
                -speed,
                '全引き出し完了後のナビゲーション用後退',
            )
        self.hsrif.whole_body.move_to_go(sync=True)
        self.base_cmd_pub.publish(Twist())
        self.get_logger().info(
            '全引き出しが完了しました。台車を離し、'
            '腕を go 姿勢に収納してナビゲーションへ引き渡します。'
        )

    def _open_one_drawer(
        self,
        drawer_level: str,
        base_forward_after_posture: float = 0.0,
        base_forward_is_final_approach: bool = False,
        arm_already_neutral: bool = False,
        close_command_override=None,
        grasp_settle_seconds: float = 0.0,
    ) -> bool:
        """Close, pull, release, and retreat from one selected drawer."""
        self.get_logger().info(f'{drawer_level} drawer: 把持と引き出し操作を開始します。')
        self._prepare_straight_arm(
            drawer_level,
            base_forward_after_posture=base_forward_after_posture,
            base_forward_is_final_approach=base_forward_is_final_approach,
            arm_already_neutral=arm_already_neutral,
            close_command_override=close_command_override,
            grasp_settle_seconds=grasp_settle_seconds,
        )
        if bool(self._value('verify_grasp')) and not self._gripper_holds_handle():
            self.get_logger().error('取っ手を掴めませんでした。引かずに退避します。')
            self._release_and_retreat()
            return False
        if not bool(self._value('verify_grasp')):
            self.get_logger().info('把持確認を省略します。')
        if not bool(self._value('pull_after_close')):
            self.get_logger().info('pull_after_close=false: ハンドを閉じた位置で停止します。')
            return False
        self._pull_drawer(drawer_level)
        # Keep the arm exactly where it is after the wheel pull.  If the
        # upper drawer follows, its preparation step raises this same arm.
        self._release_gripper_only()
        self.get_logger().info(f'{drawer_level} drawer: 完了しました。')
        return True

    def _validate_full_sequence(self) -> None:
        """Reject calibration modes when invoked as the first YASMIN task."""
        if bool(self._value('route_only')):
            raise RuntimeError('state machine で route_only は使用できません。')
        if bool(self._value('prepare_only')):
            raise RuntimeError('state machine で prepare_only は使用できません。')
        if bool(self._value('contact_only')):
            raise RuntimeError('state machine で contact_only は使用できません。')
        if str(self._value('drawer_level')).lower() != 'bottom':
            raise RuntimeError('state machine は bottom drawer から開始します。')
        required_true = (
            'pull_after_close',
            'open_upper_after_lower',
            'open_left_drawer_after_upper',
        )
        disabled = [name for name in required_true if not bool(self._value(name))]
        if disabled:
            raise RuntimeError(
                'state machine で全引き出しを開ける設定が必要です: '
                + ', '.join(disabled)
            )
        if float(self._value('post_task_retreat_distance')) <= 0.0:
            raise RuntimeError(
                'state machine では最後の後退距離を正の値にしてください。'
            )
        if float(self._value('post_task_retreat_speed')) <= 0.0:
            raise RuntimeError(
                'state machine では最後の後退速度を正の値にしてください。'
            )

    def run_task(self, require_full_sequence: bool = False) -> bool:
        """Run the fixed-route drawer-opening procedure once."""
        whole_body = None
        original_linear_weight = None
        original_angular_weight = None
        manipulation_started = False
        try:
            if require_full_sequence:
                self._validate_full_sequence()
            if self.hsrif is None:
                self.hsrif = HSRInterfaces()
            self._drive_from_entrance()
            if bool(self._value('route_only')):
                self.get_logger().info('route_only=true: 移動のみ完了しました。最終位置を確認してください。')
                return True
            # HSRInterfaces wraps the standard HSR interface. The high weights
            # prevent its planner from adding mobile-base motion while arm moves.
            whole_body = self.hsrif.whole_body._wb
            try:
                original_linear_weight = whole_body.linear_weight
                original_angular_weight = whole_body.angular_weight
            except Exception as error:
                raise RuntimeError(
                    f'元の全身プランナ重みを読み取れません: {error}'
                ) from error
            base_weight = float(self._value('base_motion_weight'))
            whole_body.linear_weight = base_weight
            whole_body.angular_weight = base_weight
            self.get_logger().info(
                f'固定取っ手位置: base_link=(x={float(self._value("handle_x")):.3f}, '
                f'y={float(self._value("handle_y")):.3f}, '
                f'z={float(self._value("handle_z")):.3f})'
            )
            drawer_level = str(self._value('drawer_level')).lower()
            if drawer_level not in ('bottom', 'upper'):
                raise RuntimeError("drawer_level は 'bottom' または 'upper' にしてください。")
            contact_only = bool(self._value('contact_only'))
            if bool(self._value('prepare_only')) or contact_only:
                manipulation_started = True
                self._prepare_straight_arm(drawer_level, close_hand=not contact_only)
            if bool(self._value('prepare_only')):
                self.get_logger().info(
                    'prepare_only=true: 前方姿勢でハンドを閉じた位置に停止します。'
                )
                return True
            if contact_only:
                self.get_logger().info('contact_only=true: 指を開いたまま取っ手位置で停止します。')
                time.sleep(float(self._value('calibration_hold_seconds')))
                self._release_and_retreat(return_to_go=False)
                return True
            if drawer_level == 'upper':
                manipulation_started = True
                return self._open_one_drawer('upper')
            manipulation_started = True
            if not self._open_one_drawer('bottom'):
                return False
            if bool(self._value('open_upper_after_lower')):
                self.get_logger().info('lower drawer の後、upper drawer を開始します。')
                if not self._open_one_drawer('upper'):
                    return False
                if bool(self._value('open_left_drawer_after_upper')):
                    self._move_to_left_drawer()
                    left_forward = max(
                        0.0,
                        float(
                            self._value('left_drawer_forward_after_arm_down_distance')
                        ),
                    )
                    if not self._open_one_drawer(
                        'bottom',
                        base_forward_after_posture=left_forward,
                        base_forward_is_final_approach=True,
                        arm_already_neutral=True,
                        close_command_override=float(
                            self._value('left_drawer_close_angle')
                        ),
                        grasp_settle_seconds=float(
                            self._value('left_drawer_grasp_settle_seconds')
                        ),
                    ):
                        return False
                    self.get_logger().info('left drawer: 完了しました。')
                    self._prepare_for_navigation()
                return True
            self.get_logger().info('Bottom drawer-open task completed.')
            return True
        except Exception as error:
            self.get_logger().error(f'引き出し操作に失敗しました: {error}')
            try:
                self.stop_base()
            except Exception as stop_error:
                self.get_logger().error(f'台車の停止にも失敗しました: {stop_error}')
            try:
                if self.hsrif is not None and manipulation_started:
                    self._release_and_retreat()
            except Exception as retreat_error:
                self.get_logger().error(f'退避にも失敗しました: {retreat_error}')
            return False
        finally:
            # The teammate grasp/manipulation states reuse this interface.
            # Restore their normal planner weights before leaving this state.
            if whole_body is not None:
                try:
                    if original_linear_weight is not None:
                        whole_body.linear_weight = original_linear_weight
                    if original_angular_weight is not None:
                        whole_body.angular_weight = original_angular_weight
                except Exception as error:
                    self.get_logger().warn(
                        f'全身プランナ重みの復元に失敗しました: {error}'
                    )


class OpenDrawersState(State):
    """Open all drawers, clear the cabinet, and hand off to navigation."""

    def __init__(self, node: Node, drawer_task: DrawerOpenTask):
        """Store the shared node and reusable drawer controller."""
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = node
        self.drawer_task = drawer_task

    def execute(self, blackboard: Blackboard) -> str:
        """Run the drawer sequence once before the object-grasp states."""
        self.node.get_logger().info('Executing state OpenDrawers')
        blackboard.drawers_opened = False

        if not self.drawer_task.run_task(require_full_sequence=True):
            self.node.get_logger().error(
                '引き出しタスクに失敗したため、物体把持へは進みません。'
            )
            return 'failed'

        blackboard.drawers_opened = True
        self.node.get_logger().info(
            '引き出しタスク完了。Move2GraspPoint へ遷移します。'
        )
        return 'succeeded'


class DrawerOpenNode(Node):
    """Thin standalone ROS wrapper around :class:`DrawerOpenTask`."""

    def __init__(self) -> None:
        """Create the standalone node used by the open_drawer executable."""
        super().__init__('drawer_open')
        self.task = DrawerOpenTask(self)

    def run_task(self) -> bool:
        """Run the reusable drawer task in standalone mode."""
        return self.task.run_task()


def main(args=None) -> None:
    """ROS 2 console entry point."""
    rclpy.init(args=args)
    node = DrawerOpenNode()
    try:
        node.run_task()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
