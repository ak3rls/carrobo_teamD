#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""objectlist.yaml の YOLOE を使う片付けタスクの YASMIN ステートマシン."""

import rclpy
from carrobo_manipulation_pkg.hsrif import HSRInterfaces
from navigation_tools.navlib import NavModule
from rclpy.node import Node
from tf2_ros import Buffer
from tf2_ros import TransformListener
from yasmin import Blackboard
from yasmin import StateMachine
from yasmin_viewer import YasminViewerPub

from .states.drawer_open import DrawerOpenTask
from .states.drawer_open import OpenDrawersState
from .states.grasp import GraspState
from .states.move_to_box import Move2BoxState
from .states.move_to_grasp_point import Move2GraspPointState
from .states.move_to_room import Move2RoomState
from .states.place import PlaceState
from .states.recog2 import Recog2State
from .states.select_next_room import SelectNextRoomState


class TidyupStateMachineNode(Node):
    """YOLOE版の片付けステートマシンを構築して実行する ROS 2 ノード."""

    def __init__(self):
        """ロボットインターフェースとステートマシンを初期化する."""
        super().__init__('teamd_tidyup_sm2')

        self.nav = NavModule()
        self.hsrif = HSRInterfaces()

        # Recog2 でカメラ座標系から base_link へ変換するために使います。
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.drawer_task = DrawerOpenTask(
            self,
            hsrif=self.hsrif,
            tf_buffer=self.tf_buffer,
        )

        self.state_machine = StateMachine(outcomes=['SUCCEEDED', 'FAILED'])
        self.state_machine.add_state(
            name='Drawer',
            state=OpenDrawersState(self, self.drawer_task),
            transitions={
                'succeeded': 'MoveRoomF2A',
                'failed': 'FAILED',
            },
        )
        self.state_machine.add_state(
            name='MoveRoomF2A',
            state=Move2RoomState(
                self,
                self.nav,
                source_room='roomF',
                target_room='roomA',
            ),
            transitions={
                'succeeded': 'Recog2',
                'failed': 'FAILED',
            },
        )
        self.state_machine.add_state(
            name='Recog2',
            state=Recog2State(self, self.hsrif, self.tf_buffer, self.nav),
            transitions={
                'succeeded': 'Grasp',
                'failed': 'Recog2',
                'none': 'SelectNextRoom',
            },
        )
        self.state_machine.add_state(
            name='SelectNextRoom',
            state=SelectNextRoomState(self),
            transitions={
                'move_to_room_b': 'MoveRoomA2B',
                'finished': 'SUCCEEDED',
                'failed': 'FAILED',
            },
        )
        self.state_machine.add_state(
            name='Grasp',
            state=GraspState(self, self.hsrif),
            transitions={
                'succeeded': 'Move2Box',
                'failed': 'Recog2',
            },
        )
        self.state_machine.add_state(
            name='Move2Box',
            state=Move2BoxState(self, self.nav),
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
        self.state_machine.add_state(
            name='Move2GraspPoint',
            state=Move2GraspPointState(self, self.nav),
            transitions={
                'succeeded': 'Recog2',
                'failed': 'FAILED',
            },
        )
        self.state_machine.add_state(
            name='MoveRoomA2B',
            state=Move2RoomState(
                self,
                self.nav,
                source_room='roomA',
                target_room='roomB',
            ),
            transitions={
                'succeeded': 'Recog2',
                'failed': 'FAILED',
            },
        )

        self.viewer = YasminViewerPub(
            fsm_name='TEAMD_TIDYUP_YOLOE',
            fsm=self.state_machine,
            node=self,
        )

        self.blackboard = Blackboard()
        self.blackboard.grasp_pose = None
        self.blackboard.grasp_approach = 0.0
        self.blackboard.grasp_wrist_roll = None
        self.blackboard.target_name = ''
        self.blackboard.current_room = 'roomF'

    def run(self) -> str:
        """YOLOE版ステートマシンを実行し、最終 outcome を返す."""
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
    """ROS 2 を初期化し、YOLOE版片付けタスクを実行する."""
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
