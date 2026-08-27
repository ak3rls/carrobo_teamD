#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""じゃんけんを実行する YASMIN ステートマシン."""

import rclpy
from navigation_tools.navlib import NavModule
from rclpy.node import Node
from yasmin import Blackboard
from yasmin import StateMachine
from yasmin_viewer import YasminViewerPub

from .states.continue_game import ContinueState
from .states.human_hand import RecognizeHumanHandState
from .states.janken import JankenState
from .states.listener import WhisperState
from .states.move_backward import (
    MovebackwardState as MoveBackwardState,
)
from .states.move_forward import MoveForwardState
from .states.robot_hand import ShowRobotHandState


class JankenStateMachineNode(Node):
    """じゃんけんステートマシンを構築して実行するノード."""

    def __init__(self):
        super().__init__('teamd_janken')

        self.nav = NavModule()

        self.blackboard = Blackboard()
        self.blackboard.human_hand = None
        self.blackboard.robot_hand = None
        self.blackboard.janken_result = None

        self.state_machine = StateMachine(
            outcomes=['SUCCEEDED', 'FAILED']
        )

        self.state_machine.add_state(
            name='Whisper',
            state=WhisperState(self),
            transitions={
                'success': 'RecognizeHumanHand', # 'succeeded'? 'success'?
                'failed': 'FAILED',
            },
        )

        self.state_machine.add_state(
            name='RecognizeHumanHand',
            state=RecognizeHumanHandState(self, self.hand_recognizer),
            transitions={
                'succeeded': 'ShowRobotHand',
                'failed': 'RecognizeHumanHand',
            },
        )

        self.state_machine.add_state(
            name='ShowRobotHand',
            state=ShowRobotHandState(self),
            transitions={
                'succeeded': 'Janken',
                'failed': 'ShowRobotHand',
            },
        )

        self.state_machine.add_state(
            name='Janken',
            state=JankenState(self),
            transitions={
                'win': 'MoveForward',
                'lose': 'MoveBackward',
                'draw': 'RecognizeHumanHand',###
                'failed': 'RecognizeHumanHand',
            },
        )

        self.state_machine.add_state(
            name='MoveForward',
            state=MoveForwardState(self, self.nav),
            transitions={
                'succeeded': 'Continue',
                'failed': 'Continue',
            },
        )

        self.state_machine.add_state(
            name='MoveBackward',
            state=MoveBackwardState(self, self.nav),
            transitions={
                'succeeded': 'Continue',
                'failed': 'Continue',
            },
        )

        self.state_machine.add_state(
            name='Continue',
            state=ContinueState(self),
            transitions={
                'yes': 'Whisper',
                'no': 'SUCCEEDED',
                'failed': 'FAILED',
            },
        )

        # 最初に実行するステート
        self.state_machine.set_start_state('Whisper')

        self.viewer = YasminViewerPub(
            fsm_name='TEAMD_JANKEN',
            fsm=self.state_machine,
            node=self,
        )

    def run(self) -> str:
        """ステートマシンを実行する."""
        outcome = self.state_machine(
            blackboard=self.blackboard
        )
        self.get_logger().info(
            f'Janken state machine finished: {outcome}'
        )
        return outcome

    def cleanup(self) -> None:
        """使用したバックグラウンド処理を終了する."""
        self.viewer.shutdown()
        self.nav.shutdown()


def main(args=None):
    """ROS 2を初期化してステートマシンを実行する."""
    rclpy.init(args=args)
    node = None

    try:
        node = JankenStateMachineNode()
        node.run()
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info(
                'じゃんけんを中断します。'
            )
    finally:
        if node is not None:
            node.cleanup()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()