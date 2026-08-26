#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""じゃんけん中の HSR 動作だけを担当するモジュール.

ROS の購読や勝敗判定をこのファイルに混ぜないことで、実機を使わない
``--dry-run`` テストや、将来のモーション変更を簡単にしています。
"""

import math
import time

import rclpy
from geometry_msgs.msg import Twist

from .config import (
    ACTION_WAIT_SECONDS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    INITIAL_JOINTS,
    INITIAL_HOLD_SECONDS,
    PRESENTATION_JOINTS,
    WIN_ARM_JOINTS,
    WIN_WRIST_SWEEP,
    HAND_POSES,
)


class JankenRobot:
    """HSR の腕・手・台車を、じゃんけん用 API として提供する."""

    def __init__(self, node, dry_run: bool = False):
        self.node = node
        self.hsr = None
        if not dry_run:
            # HSRInterfaces は hsrb_interface の初期化時に controller の
            # service を待つため、bringup/Isaac Sim を先に起動してください。
            from carrobo_manipulation_pkg.hsrif import HSRInterfaces

            self.hsr = HSRInterfaces()

    def _wait(self, seconds: float = ACTION_WAIT_SECONDS) -> None:
        """動作が安定するまで待つ。dry-run でも時間関係を再現する."""

        time.sleep(seconds)

    def prepare(self) -> None:
        """ゲーム開始姿勢: 手を閉じ、腕と頭を下げて 3 秒保持する."""

        self.node.publish_state('initial_pose')
        if self.hsr is None:
            self.node.get_logger().info('dry-run: initial pose')
            self._wait(INITIAL_HOLD_SECONDS)
            return
        self.hsr.gripper.command(GRIPPER_CLOSED)
        self._wait()
        self.hsr.whole_body.move_to_joint_positions(INITIAL_JOINTS, sync=True)
        self._wait(INITIAL_HOLD_SECONDS)

    def show_choice(self, choice: str) -> None:
        """ロボットの選んだ手を HSR の手で表示する."""

        pose = HAND_POSES[choice]
        self.node.publish_state('showing_robot_choice')
        if self.hsr is None:
            self.node.get_logger().info('dry-run: robot choice=%s' % choice)
            return

        # いったん開いてから腕を前へ出し、最後にグー／チョキ／パーを
        # 表示します。各 command の後に 1 秒待つことで動作を見やすくします。
        self.hsr.gripper.command(GRIPPER_OPEN)
        self._wait()
        self.hsr.whole_body.move_to_joint_positions(
            PRESENTATION_JOINTS, sync=True)
        self._wait()
        self.hsr.whole_body.move_to_joint_positions(
            {'wrist_roll_joint': pose.wrist_roll}, sync=True)
        self._wait()
        self.hsr.gripper.command(pose.gripper)
        self._wait()

    def win(self) -> None:
        """勝ち動作: 腕を上げ、可動範囲内で手首をスイープする."""

        self.node.publish_state('robot_win_motion')
        if self.hsr is None:
            self.node.get_logger().info('dry-run: win motion')
            return
        self.hsr.whole_body.move_to_joint_positions(WIN_ARM_JOINTS, sync=True)
        self._wait()
        for roll in WIN_WRIST_SWEEP:
            self.hsr.whole_body.move_to_joint_positions(
                {'wrist_roll_joint': roll}, sync=True)
            self._wait()

    def lose(self) -> None:
        """負け動作: 半メートル後退して、台車を 180 度回転する."""

        self.node.publish_state('robot_lose_motion')
        if self.hsr is None:
            self.node.get_logger().info('dry-run: lose motion')
            return
        try:
            moved = self.hsr.omni_base.go_rel(
                -0.5, 0.0, math.pi, timeout=8.0, sync=True)
            if moved is True:
                self._wait()
                return
            self.node.get_logger().warning(
                'go_rel が完了しなかったため cmd_vel に切り替えます。')
        except Exception as error:
            self.node.get_logger().warning('go_rel に失敗しました: %s' % error)

        # Nav2 の Action が無い Isaac Sim でも動くように、HSRInterfaces が
        # 持つ ROS ノードから速度指令を送るフォールバックを用意します。
        ros_node = getattr(self.hsr, '_node', None)
        if ros_node is None:
            return
        publisher = ros_node.create_publisher(
            Twist, '/omni_base_controller/cmd_vel', 10)
        for linear_x, angular_z, duration in (
                (-0.15, 0.0, 3.4), (0.0, 0.6, math.pi / 0.6)):
            deadline = time.monotonic() + duration
            while rclpy.ok() and time.monotonic() < deadline:
                command = Twist()
                command.linear.x = linear_x
                command.angular.z = angular_z
                publisher.publish(command)
                rclpy.spin_once(ros_node, timeout_sec=0.05)
            publisher.publish(Twist())
            self._wait()

    def cleanup(self) -> None:
        """終了時に手を開き、whole_body を neutral に戻す."""

        if self.hsr is None:
            return
        try:
            self.hsr.gripper.command(GRIPPER_OPEN)
            self.hsr.whole_body.move_to_neutral(sync=True)
        except Exception as error:
            self.node.get_logger().warning('終了姿勢に失敗しました: %s' % error)
