#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""じゃんけんの YASMIN ステートマシンを構築するファイル.

ROS 通信を担当する ``JankenCoordinatorNode`` と、各ステートをつなぐ
処理を分離しています。ここを読むとゲームの全体の流れが分かります。
"""
import rclpy
from navigation_tools.navlib import NavModule
from rclpy.node import Node
from tf2_ros import Buffer
from tf2_ros import TransformListener
from yasmin import Blackboard
from yasmin import StateMachine
from yasmin_viewer import YasminViewerPub

from yasmin import Blackboard, StateMachine

from .config import START_ALIASES, THROW_ALIASES
from .states import linear, move_backward, move_forward, janken
from .nodes import hand_recog, jyanken_robot_node


class JankenStateMachine(Node):
    """じゃんけん 1 回分の状態遷移と blackboard を保持する."""

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.blackboard = Blackboard()
        # 各ステートが参照する値を初期化します。
        self.blackboard.robot_choice = None
        self.blackboard.player_choice = None

        sm = StateMachine(outcomes=["EXIT"])
        sm.add_state(
            name = "LISTENER",
            state = linear.Whisper_state(self),
            transitions = {
                "success": "JANKEN"
            }
        )
        sm.add_state(
            name = "JANKEN",
            state = 
            
        )
        sm.add_state(
            'EnableMediaPipe', DetectorControlState(node, True),
            {'succeeded': 'WaitThrow', 'failed': 'FAILED'},
        )
        sm.add_state(
            'WaitThrow',
            WaitWhisperState(node, THROW_ALIASES, 'waiting_throw_phrase'),
            {'succeeded': 'ShowRobotChoice', 'failed': 'FAILED'},
        )
        sm.add_state(
            'ShowRobotChoice', ShowRobotChoiceState(node),
            {'succeeded': 'WaitPlayerGesture', 'failed': 'FAILED'},
        )
        sm.add_state(
            'WaitPlayerGesture',
            WaitPlayerGestureState(node, node.args.gesture_timeout),
            {'succeeded': 'Judge', 'failed': 'FAILED'},
        )
        sm.add_state(
            'Judge', JudgeState(node),
            {'succeeded': 'Finish', 'failed': 'FAILED'},
        )
        sm.add_state(
            'Finish', FinishState(node),
            {'succeeded': 'SUCCEEDED', 'failed': 'FAILED'},
        )

    def run(self) -> str:
        """YASMIN を実行し、``SUCCEEDED`` または ``FAILED`` を返す."""

        outcome = self.fsm(blackboard=self.blackboard)
        self.node.get_logger().info('Janken state machine finished: %s' % outcome)
        return outcome
