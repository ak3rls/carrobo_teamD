#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""片付けタスクを実行する YASMIN ステートマシン."""

import rclpy
from navigation_tools.navlib import NavModule
from rclpy.node import Node
from tf2_ros import Buffer
from tf2_ros import TransformListener
from yasmin import Blackboard
from yasmin import StateMachine
from yasmin_viewer import YasminViewerPub

from carrobo_manipulation_pkg.hsrif import HSRInterfaces

from .states.grasp import GraspState
from .states.move_to_grasp_point import Move2GraspPointState
from .states.move_to_place_point import Move2PlacePointState
from .states.place import PlaceState
from .states.recog import RecogState


class TidyupStateMachineNode(Node):
    """片付けの5ステートを構築して実行する ROS 2 ノード."""

    def __init__(self):
        """ロボットインターフェースとステートマシンを初期化する."""
        super().__init__('teamd_tidyup')

        # carrobo_nav と carrobo_manipulation_pkg の例と同じインターフェースです。
        self.nav = NavModule()
        self.hsrif = HSRInterfaces()

        # Recog でカメラ座標系から base_link へ変換するために使います。
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.state_machine = StateMachine(outcomes=['SUCCEEDED', 'FAILED'])
        self.state_machine.add_state(
            name='Move2GraspPoint',
            state=Move2GraspPointState(self, self.nav),
            transitions={
                'succeeded': 'Recog',
                'failed': 'FAILED',
            },
        )
        self.state_machine.add_state(
            name='Recog',
            state=RecogState(self, self.hsrif, self.tf_buffer),
            transitions={
                'succeeded': 'Grasp',
                'failed': 'FAILED',
            },
        )
        self.state_machine.add_state(
            name='Grasp',
            state=GraspState(self, self.hsrif),
            transitions={
                'succeeded': 'Move2PlacePoint',
                'failed': 'FAILED',
            },
        )
        self.state_machine.add_state(
            name='Move2PlacePoint',
            state=Move2PlacePointState(self, self.nav),
            transitions={
                'succeeded': 'Place',
                'failed': 'FAILED',
            },
        )
        self.state_machine.add_state(
            name='Place',
            state=PlaceState(self, self.hsrif),
            transitions={
                'succeeded': 'Move2GraspPoint',
                'failed': 'FAILED',
            },
        )
        #drawerのステートマシン
        # self.state_machine.add_state(
        #     name='drawer',
        #     state=drawerawerState(self),
        #     transitions={
        #         'succeeded': 'Move2GraspPoint',
        #     }
        # )

        self.viewer = YasminViewerPub(
            fsm_name='TEAMD_TIDYUP',
            fsm=self.state_machine,
        )

        self.blackboard = Blackboard()
        self.blackboard.grasp_pose = None
        self.blackboard.grasp_approach = 0.0
        self.blackboard.target_name = ''

    def run(self) -> str:
        """ステートマシンを実行し、最終 outcome を返す."""
        outcome = self.state_machine(blackboard=self.blackboard)
        self.get_logger().info(f'State machine finished: {outcome}')
        return outcome

    def cleanup(self):
        """ナビゲーションと Viewer のバックグラウンド処理を止める."""
        if self.nav.is_navigating:
            self.nav.cancel_nav_action()
        self.nav.shutdown()
        self.viewer.shutdown()


def main(args=None):
    """ROS 2 を初期化し、片付けステートマシンを実行する."""
    rclpy.init(args=args)
    node = TidyupStateMachineNode()

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('片付けタスクを中断します。')
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
