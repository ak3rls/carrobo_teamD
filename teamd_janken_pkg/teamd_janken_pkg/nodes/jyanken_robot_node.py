#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""じゃんけん専用の HSR ロボット動作ノード.

このノードは Whisper、MediaPipe、勝敗判定を知りません。``jyanken_sm``
から JSON 要求を受け取り、HSR の腕・手・台車を動かして完了応答を返す
だけです。そのため、状態機械や認識ノードを別の実装へ交換できます。

要求 topic:
    ``/teamd_jyanken/robot/request`` (``std_msgs/msg/String``)

要求の例:
    ``{"request_id":"abc","action":"choose_and_show"}``

``choose_and_show`` を受けた場合、選択はこのノード内で行われます。
応答には ``"choice":"rock"`` のように選択結果が含まれます。

応答 topic:
    ``/teamd_jyanken/robot/response`` (``std_msgs/msg/String``)
"""

import argparse
import json
import math
import random
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import String
from teamd_janken_interfaces.srv import janken_robot_node 

REQUEST_TOPIC = '/teamd_jyanken/robot/request'
RESPONSE_TOPIC = '/teamd_jyanken/robot/response'

# このファイルだけをコピーしても動作するように、姿勢と hand 定義を
# 外部 config.py へ依存させず、ここにまとめています。
ACTION_WAIT = 1.0
INITIAL_HOLD = 3.0
OPEN = 1.0
CLOSED = 0.0

INITIAL_JOINTS = {
    'arm_lift_joint': 0.0, 'arm_flex_joint': 0.0,
    'arm_roll_joint': 0.0, 'wrist_flex_joint': -1.57,
    'wrist_roll_joint': 0.0, 'head_pan_joint': -0.5,
    'head_tilt_joint': -0.5,
}
PRESENTATION_JOINTS = {
    'arm_lift_joint': 0.20, 'arm_flex_joint': 0.0,
    'arm_roll_joint': 0.0, 'wrist_flex_joint': -1.57,
    'wrist_roll_joint': 0.0, 'head_pan_joint': 0.0,
    'head_tilt_joint': 0.0,
}
WIN_ARM_JOINTS = {
    'arm_lift_joint': 0.55, 'arm_flex_joint': -0.45,
    'arm_roll_joint': 0.0, 'wrist_flex_joint': -1.0,
}
WIN_WRIST_SWEEP = (3.4, -1.8, 0.0)
HAND_POSES = {
    'rock': {'wrist_roll': 0.0, 'gripper': CLOSED},
    'scissors': {'wrist_roll': 1.57, 'gripper': OPEN},
    'paper': {'wrist_roll': 0.0, 'gripper': OPEN},
}


class JankenRobot:
    """HSR の腕・手・台車の動作をこのノード内だけで完結させる."""

    def __init__(self, node, dry_run=False):
        self.node = node
        self.hsr = None

        # ここの処理は実機の時の話なので無視してok
        if not dry_run:
            # 実機または Isaac Sim の HSR bringup が先に必要です。
            from carrobo_manipulation_pkg.hsrif import HSRInterfaces

            self.hsr = HSRInterfaces()


    #ロボットの動き中の停止関数
    def _wait(self, seconds=ACTION_WAIT):
        time.sleep(seconds)

    

    def prepare(self):
        """手を閉じ、腕と頭を下げた初期姿勢を 3 秒保持する."""
        self.node.publish_state('initial_pose')

        #ロボットの接続確認(もしできてなかったら少し待ってから関数から抜け出す)
        if self.hsr is None:
            self.node.get_logger().info('dry-run: initial pose')
            self._wait(INITIAL_HOLD)
            return
        #初期状態(ぐー)に持っていく
        self.hsr.gripper.command(CLOSED)

        self._wait()
        #ロボットの関節を初期状態に持っていく
        self.hsr.whole_body.move_to_joint_positions(INITIAL_JOINTS, sync=True)
        #長めに待つ
        self._wait(INITIAL_HOLD)

    #choiceには'rock' 'scissor' 'paper'を入れる
    def show_choice(self, choice):
        """指定されたグー／チョキ／パーを HSR で表示する."""

         #辞書を参考に手の動きを決める
        pose = HAND_POSES[choice]

        self.node.publish_state('showing_robot_choice')

        #ロボットの接続確認(もしできてなかったら少し待ってから関数から抜け出す)
        if self.hsr is None:
            self.node.get_logger().info('dry-run: robot choice=%s' % choice)
            return

        #手を一旦パーにする
        self.hsr.gripper.command(OPEN)

        self._wait()
        #じゃんけんの姿勢にする
        self.hsr.whole_body.move_to_joint_positions(
            PRESENTATION_JOINTS, sync=True)

        self._wait()
        #手の捻り具合の調整
        self.hsr.whole_body.move_to_joint_positions(
            {'wrist_roll_joint': pose['wrist_roll']}, sync=True)

        self._wait()
        #手の形を変える
        self.hsr.gripper.command(pose['gripper'])

        self._wait()

    def win(self):
        # """勝ったときに腕を上げて手首を安全範囲で動かす."""

        # self.node.publish_state('robot_win_motion')
        # if self.hsr is None:
        #     self.node.get_logger().info('dry-run: win motion')
        #     return
        # self.hsr.whole_body.move_to_joint_positions(WIN_ARM_JOINTS, sync=True)
        # self._wait()
        # for roll in WIN_WRIST_SWEEP:
        #     self.hsr.whole_body.move_to_joint_positions(
        #         {'wrist_roll_joint': roll}, sync=True)
        #     self._wait()

    def lose(self):
        # """負けたときに半メートル後退し、台車を 180 度回す."""

        # self.node.publish_state('robot_lose_motion')
        # if self.hsr is None:
        #     self.node.get_logger().info('dry-run: lose motion')
        #     return
        # try:
        #     moved = self.hsr.omni_base.go_rel(
        #         -0.5, 0.0, math.pi, timeout=8.0, sync=True)
        #     if moved is True:
        #         self._wait()
        #         return
        #     self.node.get_logger().warning(
        #         'go_rel が完了しなかったため cmd_vel に切り替えます。')
        # except Exception as error:
        #     self.node.get_logger().warning('go_rel に失敗しました: %s' % error)
        # ros_node = getattr(self.hsr, '_node', None)
        # if ros_node is None:
        #     return
        # publisher = ros_node.create_publisher(
        #     Twist, '/omni_base_controller/cmd_vel', 10)
        # for linear_x, angular_z, duration in (
        #         (-0.15, 0.0, 3.4), (0.0, 0.6, math.pi / 0.6)):
        #     deadline = time.monotonic() + duration
        #     while rclpy.ok() and time.monotonic() < deadline:
        #         command = Twist()
        #         command.linear.x = linear_x
        #         command.angular.z = angular_z
        #         publisher.publish(command)
        #         rclpy.spin_once(ros_node, timeout_sec=0.05)
        #     publisher.publish(Twist())
        #     self._wait()

    def cleanup(self):
        """終了時に手を開き、whole_body を neutral に戻す."""

        if self.hsr is None:
            return
        try:
            self.hsr.gripper.command(OPEN)
            self.hsr.whole_body.move_to_neutral(sync=True)
        except Exception as error:
            self.node.get_logger().warning('終了姿勢に失敗しました: %s' % error)


    class JankenRobotNode(Node):
        """この 1 ファイル内の HSR 動作を ROS request/response に変換する."""

        def __init__(self, args):
            super().__init__('teamd_jyanken_robot')
            self.robots = JankenRobot()
            self.srv = self.create_service (
                janken_robot_node,
                'robot_move',
                self.move_janken
            )
        def move_janken(self, request, response):
            self.robots.show_choice(request.sentaku)
            print("ok!")
            response.result = "sucsses"  
            return response
#         self.response_pub = self.create_publisher(
#             String, args.response_topic, 10)
#         self.request_sub = self.create_subscription(
#             String, args.request_topic, self._on_request, 10)
#         self.robot = JankenRobot(self, dry_run=args.dry_run)
#         self._motion_lock = threading.Lock()
#         self._initial_done = False

#         # ロボットノード単体で起動しても安全な初期姿勢になるようにします。
#         # jyanken_sm から initial 要求が来た場合は二重動作を避けます。
#         if not args.no_auto_initial:
#             self._execute_action('initial', 'startup')

#     def _publish_response(self, request_id, action, status, message='', **fields):
#         response = {
#             'request_id': request_id,
#             'action': action,
#             'status': status,
#         }
#         if message:
#             response['message'] = message
#         response.update(fields)
#         self.response_pub.publish(String(data=json.dumps(
#             response, ensure_ascii=False)))

#     def publish_state(self, state):
#         """このファイル内の動作クラスが使う簡易状態ログ."""

#         self.get_logger().info('robot state: %s' % state)

#     def _on_request(self, message):
#         """要求 JSON を検証して、1 件ずつ直列に実行する."""

#         try:
#             request = json.loads(message.data)
#         except (TypeError, json.JSONDecodeError) as error:
#             self.get_logger().error('ロボット要求の JSON が不正です: %s' % error)
#             self._publish_response('', '', 'error', 'invalid JSON')
#             return
#         if not isinstance(request, dict):
#             self._publish_response('', '', 'error', 'request must be an object')
#             return

#         request_id = str(request.get('request_id', ''))
#         action = str(request.get('action', '')).lower()
#         choice = str(request.get('choice', '')).lower()
#         if action == 'show' and choice not in HAND_POSES:
#             self._publish_response(
#                 request_id, action, 'error', 'choice must be rock/scissors/paper')
#             return
#         self._execute_action(action, request_id, choice=choice)

#     def _execute_action(self, action, request_id, choice=''):
#         """要求された 1 動作を実行し、必ず done/error を返す."""

#         with self._motion_lock:
#             try:
#                 if action == 'initial':
#                     if not self._initial_done:
#                         self.robot.prepare()
#                         self._initial_done = True
#                 elif action in ('choose', 'choose_and_show', 'start'):
#                     # 手の選択は robot node だけが行います。state machine は
#                     # response の choice を受け取って勝敗判定に使います。
#                     choice = random.choice(tuple(HAND_POSES))
#                     self.robot.show_choice(choice)
#                 elif action == 'show':
#                     self.robot.show_choice(choice)
#                 elif action == 'win':
#                     self.robot.win()
#                 elif action == 'lose':
#                     self.robot.lose()
#                 elif action == 'cleanup':
#                     self.robot.cleanup()
#                 else:
#                     raise ValueError(
#                         'action must be initial/choose_and_show/show/win/lose/cleanup')
#             except Exception as error:
#                 self.get_logger().error('robot action failed: %s' % error)
#                 self._publish_response(request_id, action, 'error', str(error))
#                 return
#             response_fields = {'choice': choice} if action in (
#                 'choose', 'choose_and_show', 'start', 'show') else {}
#             self._publish_response(
#                 request_id, action, 'done', **response_fields)


# def build_parser():
#     """ロボットノードの topic とテスト用引数を定義する."""

#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument('--request-topic', default=REQUEST_TOPIC)
#     parser.add_argument('--response-topic', default=RESPONSE_TOPIC)
#     parser.add_argument('--dry-run', action='store_true')
#     parser.add_argument('--no-auto-initial', action='store_true')
#     return parser


def main(args=None):
    """ROS 2 を初期化し、ロボット要求を待ち続ける."""

    parsed = build_parser().parse_args(remove_ros_args(args))
    rclpy.init(args=args)
    node = JankenRobotNode(parsed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('ロボットノードを停止します。')
    finally:
        node.robot.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
