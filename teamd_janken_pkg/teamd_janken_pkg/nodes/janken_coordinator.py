#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Whisper と MediaPipe と HSR を接続する ROS 2 ノード.

このノードは「認識アルゴリズム」そのものを実装しません。チームメンバー
の Whisper ノードと MediaPipe ノードが標準メッセージを publish し、この
ノードが状態に応じて受信・ロボット動作・勝敗通知を調整します。
"""

import argparse
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import Bool, String

from yasmin_viewer import YasminViewerPub

from ..config import (
    DEFAULT_DETECTOR_ENABLE_TOPIC,
    DEFAULT_GESTURE_TOPIC,
    DEFAULT_WHISPER_TOPIC,
    RESULT_TOPIC,
    STATE_TOPIC,
)
from ..janken_state_machine import JankenStateMachine
from ..protocol import contains_phrase, parse_hand
from ..robot_motion import JankenRobot
from ..speaker import JapaneseSpeaker


class JankenCoordinatorNode(Node):
    """じゃんけんの ROS 通信、音声、ロボットをまとめるノード."""

    def __init__(self, args):
        super().__init__('teamd_jyanken')
        self.args = args

        # 最新値 1 個ではなくキューを使います。Whisper が短時間に複数の
        # 結果を送っても、ステートが順番に確認できるようにします。
        self._whisper_messages = deque(maxlen=100)
        self._gesture_messages = deque(maxlen=100)

        self.whisper_sub = self.create_subscription(
            String, args.whisper_topic, self._on_whisper, 10)
        self.gesture_sub = self.create_subscription(
            String, args.gesture_topic, self._on_gesture, 10)
        self.detector_pub = None
        if args.detector_enable_topic:
            self.detector_pub = self.create_publisher(
                Bool, args.detector_enable_topic, 10)
        self.state_pub = self.create_publisher(String, STATE_TOPIC, 10)
        self.result_pub = self.create_publisher(String, RESULT_TOPIC, 10)

        # HSRInterfaces は controller service が必要です。ロボットを起動
        # せず接続だけ確認したい場合は --dry-run を指定してください。
        self.robot = JankenRobot(self, dry_run=args.dry_run)
        self.speaker = JapaneseSpeaker(
            enabled=not args.no_speech, speed=args.speech_speed)
        self.state_machine = JankenStateMachine(self)

        self.viewer = None
        if not args.no_viewer:
            self.viewer = YasminViewerPub(
                fsm_name='TEAMD_JYANKEN', fsm=self.state_machine.fsm)

        self.get_logger().info('janken coordinator is ready')
        self.get_logger().info('Whisper: %s' % args.whisper_topic)
        self.get_logger().info('MediaPipe: %s' % args.gesture_topic)

    def _on_whisper(self, message: String) -> None:
        """Whisper の全文をキューへ保存するコールバック."""

        self._whisper_messages.append(message.data)

    def _on_gesture(self, message: String) -> None:
        """MediaPipe の文字列を正規化してからキューへ保存する."""

        choice = parse_hand(message.data)
        if choice is not None:
            self._gesture_messages.append(choice)

    def publish_state(self, state: str) -> None:
        """現在の状態を他のノードとデバッグ画面へ通知する."""

        self.state_pub.publish(String(data=state))
        self.get_logger().info('state: %s' % state)

    def clear_whisper_messages(self) -> None:
        """ステート遷移前に古い Whisper 結果を破棄する."""

        self._whisper_messages.clear()

    def clear_gesture_messages(self) -> None:
        """ステート遷移前に古い MediaPipe 結果を破棄する."""

        self._gesture_messages.clear()

    def wait_for_whisper(self, aliases):
        """指定フレーズを受信するまで spin する."""

        deadline = time.monotonic() + self.args.phrase_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            while self._whisper_messages:
                text = self._whisper_messages.popleft()
                if contains_phrase(text, aliases):
                    return text
            rclpy.spin_once(self, timeout_sec=0.1)
        return None

    def wait_for_gesture(self, timeout_seconds):
        """MediaPipe の最初の有効な手を受信するまで spin する."""

        deadline = time.monotonic() + timeout_seconds
        while rclpy.ok() and time.monotonic() < deadline:
            if self._gesture_messages:
                return self._gesture_messages.popleft()
            rclpy.spin_once(self, timeout_sec=0.1)
        return None

    def set_detector_enabled(self, enabled: bool) -> None:
        """MediaPipe ノードへ検出開始／停止を伝える."""

        if self.detector_pub is not None:
            self.detector_pub.publish(Bool(data=enabled))
        self.get_logger().info('MediaPipe detector: %s' % (
            'ON' if enabled else 'OFF'))

    def run(self) -> str:
        """状態機械を 1 回実行する."""

        return self.state_machine.run()

    def cleanup(self) -> None:
        """検出停止、ロボット安全姿勢、YASMIN Viewer 停止を行う."""

        try:
            self.set_detector_enabled(False)
            self.robot.cleanup()
        finally:
            if self.viewer is not None:
                self.viewer.shutdown()


def build_argument_parser() -> argparse.ArgumentParser:
    """ROS 以外のコマンドライン引数を定義する."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--whisper-topic', default=DEFAULT_WHISPER_TOPIC)
    parser.add_argument('--gesture-topic', default=DEFAULT_GESTURE_TOPIC)
    parser.add_argument(
        '--detector-enable-topic', default=DEFAULT_DETECTOR_ENABLE_TOPIC,
        help='空文字にすると MediaPipe 制御信号を publish しません',
    )
    parser.add_argument(
        '--phrase-timeout', type=float, default=60.0,
        help='最初はグー／じゃんけんぽいを待つ秒数',
    )
    parser.add_argument(
        '--gesture-timeout', type=float, default=30.0,
        help='プレイヤーの手形を待つ秒数',
    )
    parser.add_argument('--speech-speed', type=int, default=120)
    parser.add_argument('--no-speech', action='store_true')
    parser.add_argument('--no-viewer', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main(args=None) -> None:
    """ROS 2 を初期化し、じゃんけん状態機械を実行する."""

    parser = build_argument_parser()
    # launch から渡される --ros-args を argparse が誤って解釈しないよう
    # に、ROS 専用引数を取り除いてからアプリ引数を解析します。
    parsed = parser.parse_args(remove_ros_args(args))
    rclpy.init(args=args)
    node = JankenCoordinatorNode(parsed)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('じゃんけんを中断します。')
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
