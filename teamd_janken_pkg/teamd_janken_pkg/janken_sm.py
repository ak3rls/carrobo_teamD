#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""じゃんけんの YASMIN ステートマシンを構築するファイル.

ROS 通信を担当する ``JankenCoordinatorNode`` と、各ステートをつなぐ
処理を分離しています。ここを読むとゲームの全体の流れが分かります。
"""

from yasmin import Blackboard, StateMachine

from .config import START_ALIASES, THROW_ALIASES
from .states import (
    DetectorControlState,
    FinishState,
    InitialPoseState,
    JudgeState,
    ShowRobotChoiceState,
    WaitPlayerGestureState,
    WaitWhisperState,
)


class JankenStateMachine:
    """じゃんけん 1 回分の状態遷移と blackboard を保持する."""

    def __init__(self, node):
        self.node = node
        self.blackboard = Blackboard()
        # 各ステートが参照する値を初期化します。
        self.blackboard.robot_choice = None
        self.blackboard.player_choice = None

        self.fsm = StateMachine(outcomes=['SUCCEEDED', 'FAILED'])
        self.fsm.add_state(
            'InitialPose', InitialPoseState(node),
            {'succeeded': 'WaitStart', 'failed': 'FAILED'},
        )
        self.fsm.add_state(
            'WaitStart',
            WaitWhisperState(node, START_ALIASES, 'waiting_start_phrase'),
            {'succeeded': 'EnableMediaPipe', 'failed': 'FAILED'},
        )
        self.fsm.add_state(
            'EnableMediaPipe', DetectorControlState(node, True),
            {'succeeded': 'WaitThrow', 'failed': 'FAILED'},
        )
        self.fsm.add_state(
            'WaitThrow',
            WaitWhisperState(node, THROW_ALIASES, 'waiting_throw_phrase'),
            {'succeeded': 'ShowRobotChoice', 'failed': 'FAILED'},
        )
        self.fsm.add_state(
            'ShowRobotChoice', ShowRobotChoiceState(node),
            {'succeeded': 'WaitPlayerGesture', 'failed': 'FAILED'},
        )
        self.fsm.add_state(
            'WaitPlayerGesture',
            WaitPlayerGestureState(node, node.args.gesture_timeout),
            {'succeeded': 'Judge', 'failed': 'FAILED'},
        )
        self.fsm.add_state(
            'Judge', JudgeState(node),
            {'succeeded': 'Finish', 'failed': 'FAILED'},
        )
        self.fsm.add_state(
            'Finish', FinishState(node),
            {'succeeded': 'SUCCEEDED', 'failed': 'FAILED'},
        )

    def run(self) -> str:
        """YASMIN を実行し、``SUCCEEDED`` または ``FAILED`` を返す."""

        outcome = self.fsm(blackboard=self.blackboard)
        self.node.get_logger().info('Janken state machine finished: %s' % outcome)
        return outcome
