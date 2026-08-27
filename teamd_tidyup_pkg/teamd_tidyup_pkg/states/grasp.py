#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blackboard に保存された姿勢で物体を把持するステート."""

import math
import time

import numpy as np
import rclpy
import tf_transformations as tft
from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State

from hsrb_interface import geometry

from carrobo_manipulation_pkg.hsrif import HSRInterfaces


WRIST_ROLL_JOINT = 'wrist_roll_joint'
# hsrb_description の URDF の値。joint_limits が引けないときに使います。
WRIST_ROLL_LIMITS = (-1.92, 3.67)

# 各動作の間に置く静定時間 [s]。sync=True でも指令が返った直後は
# 腕がまだ揺れているため、次の動作を重ねないように待ちます。
SETTLE_TIME = 0.5
# グリッパを閉じるのにかける時間 [s]。短いと薄い物体を弾いてしまいます。
GRIPPER_CLOSE_TIME = 2.0
# グリッパを閉じるときの目標角 [rad]。負の値ほど強く握り込みます。
GRIPPER_CLOSE_ANGLE = -0.1
# 物体ごとに握り込み量を変えたいときに指定します。
# 太くて滑る物体は強く握らないと持ち上げられません。
# ここに無い物体は GRIPPER_CLOSE_ANGLE を使います。
# キーは小文字で書いてください。
GRIPPER_CLOSE_ANGLES = {
    'cleanser bottle': -0.3,
}
# 握った後、走行姿勢へ移る前に真上へ引き上げる距離 [m]。
# move_to_go は腕を大きく振るので、先に物体を持ち上げて逃がします。
LIFT_DISTANCE = 0.05

# 上把持のとき、引き上げた後に台車を後退させる距離 [m]。
# 手を真下に向けて掴むと机や棚のすぐ上に腕がいるので、離れてから
# move_to_go で腕を振ります。
BACKUP_DISTANCE = 0.10
# 後退の基準フレームです。ここから見た -X がまっすぐ後ろになります。
BASE_FRAME = 'base_footprint'
# 後退の完了を待つ最大時間 [s]。
BACKUP_TIMEOUT = 15.0


def _hand_yaw(orientation) -> float:
    """手のひらが真下のときの、指の間を通り抜ける向き [rad] を返す.

    hand_palm_link のローカル Y 軸が、左右の指が並ぶ向き
    (＝グリッパが閉じる方向) です。返すのはその向きそのものではなく、
    そこから 90 度回した「指の間を通り抜ける辺の向き」です。
    手のひらが真下の姿勢は必ず rpy(pi, 0, yaw) の形になり、この関数は
    その yaw を返します。指が閉じるのは yaw - 90 度の向きです。

    目標にしたい閉じ方向と直接比べるには、閉じ方向に 90 度足した向きを
    渡します。長辺方向に閉じたい場合は、物体の長辺の向き + 90 度を
    target_hand_yaw に設定します。これで指は短辺側の面に接触します。
    """
    rotation = tft.quaternion_matrix([
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    ])[:3, :3]
    closing_axis = rotation[:, 1]
    return math.atan2(closing_axis[0], -closing_axis[1])


def _fold_to_half_pi(angle: float) -> float:
    """グリッパの 180 度対称性を使い、角度を [-pi/2, pi/2) へ畳む.

    指の並びは 180 度回しても同じ挟み方になるので、
    回転量は必ず 90 度以内に収められます。
    """
    return (angle + math.pi / 2.0) % math.pi - math.pi / 2.0


def _wrist_roll_goal(
    current_roll: float,
    current_hand_yaw: float,
    target_hand_yaw: float,
    limits,
) -> float:
    """長辺方向に閉じて短辺側の面を挟むための目標角を求める.

    hand_palm_link は wrist_roll_link に rpy(0,0,pi) で付いているだけなので、
    ハンドの Z 軸は wrist_roll の回転軸そのものです。手のひらが真下を向く
    ときこの軸は base_link の -Z を指すため、wrist_roll を +d 回すと
    ハンドは鉛直まわりに -d 回ります。符号が反転する点に注意。

    Args:
        current_roll: いまの wrist_roll_joint の角度 [rad]。
        current_hand_yaw: いまの指の間を通り抜ける向き [rad]。
        target_hand_yaw: 合わせたい指の間を通り抜ける向き [rad]。
            長辺方向に閉じる場合は、物体の長辺の向き + 90 度です。
        limits: (下限, 上限) [rad]。

    Returns:
        wrist_roll_joint の目標角 [rad]。
    """
    delta = _fold_to_half_pi(target_hand_yaw - current_hand_yaw)
    lower, upper = limits
    candidates = [
        current_roll - delta + n * math.pi
        for n in (-2, -1, 0, 1, 2)
    ]
    reachable = [
        value for value in candidates if lower <= value <= upper
    ]
    if not reachable:
        raise ValueError(
            f'{WRIST_ROLL_JOINT} を '
            f'{math.degrees(current_roll):.1f} deg から '
            f'{math.degrees(delta):.1f} deg 回す指令が可動域外です。'
        )
    return min(reachable, key=lambda value: abs(value - current_roll))


class GraspState(State):
    """既存の carrobo_manipulation_pkg と同じ手順で把持する."""

    def __init__(self, node: Node, hsrif: HSRInterfaces):
        """ステートを初期化する."""
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = node
        self.hsrif = hsrif

    def _step(self, description: str) -> None:
        """直前の動作が静定するのを待ってから、次の動作を知らせる."""
        time.sleep(SETTLE_TIME)
        self.node.get_logger().info(f'把持動作: {description}')

    def _wrist_roll_limits(self):
        """URDF から wrist_roll_joint の可動域を取る。取れなければ既定値."""
        try:
            return self.hsrif.whole_body.joint_limits[WRIST_ROLL_JOINT]
        # joint_limits は URDF 取得に失敗すると様々な例外を投げます。
        except Exception as error:
            self.node.get_logger().warning(
                f'{WRIST_ROLL_JOINT} の可動域を取得できませんでした'
                f' ({error})。既定値 {WRIST_ROLL_LIMITS} を使います。'
            )
            return WRIST_ROLL_LIMITS

    def _rotate_wrist_to_long_edge(self, target_hand_yaw: float) -> None:
        """手のひらを真下に向けたまま、長辺方向に閉じる向きへ回す."""
        # 指令値ではなく実際の手先姿勢から測ります。IK が姿勢を厳密に
        # 出せていなくても、そのぶんを含めて合わせられます。
        _, orientation = self.hsrif.whole_body.get_end_effector_pose(
            'base_link'
        )
        current_hand_yaw = _hand_yaw(orientation)
        current_roll = self.hsrif.whole_body.joint_positions[
            WRIST_ROLL_JOINT
        ]
        goal = _wrist_roll_goal(
            current_roll,
            current_hand_yaw,
            target_hand_yaw,
            self._wrist_roll_limits(),
        )
        self.node.get_logger().info(
            f'長辺方向に閉じる向き={math.degrees(target_hand_yaw):.1f} deg, '
            f'いまの指の向き={math.degrees(current_hand_yaw):.1f} deg -> '
            f'{WRIST_ROLL_JOINT}: {math.degrees(current_roll):.1f} deg -> '
            f'{math.degrees(goal):.1f} deg'
        )
        self.hsrif.whole_body.move_to_joint_positions(
            {WRIST_ROLL_JOINT: goal},
            sync=True,
        )

        # 実際に合ったかを残します。ずれが大きければ符号か軸の取り違えです。
        _, moved = self.hsrif.whole_body.get_end_effector_pose('base_link')
        error = _fold_to_half_pi(_hand_yaw(moved) - target_hand_yaw)
        self.node.get_logger().info(
            f'回転後の指の向き={math.degrees(_hand_yaw(moved)):.1f} deg '
            f'(目標向きとのずれ={math.degrees(error):.1f} deg)'
        )

    def _lift(self, distance: float) -> None:
        """base_link 基準の真上へ、まっすぐ引き上げる.

        move_end_effector_by_line の軸は手先ローカルです。上把持では
        ローカル -Z がたまたま真上になりますが、横把持では水平方向を
        向いてしまい持ち上がりません。base_link の +Z を手先ローカルへ
        変換して渡すことで、どちらの把持でも真上へ上げます。

        Args:
            distance: 引き上げる距離 [m]。
        """
        _, orientation = self.hsrif.whole_body.get_end_effector_pose(
            'base_link'
        )
        rotation = tft.quaternion_matrix([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ])[:3, :3]
        # base_link の +Z を手先ローカルで表した向きです。
        axis = rotation.T @ np.array([0.0, 0.0, 1.0])
        self.node.get_logger().info(
            f'真上へ {distance:.3f} m 引き上げます '
            f'(手先ローカル軸={np.round(axis, 2)})。'
        )
        self.hsrif.whole_body.move_end_effector_by_line(
            (float(axis[0]), float(axis[1]), float(axis[2])),
            distance,
            sync=True,
        )

    def _back_up(self, distance: float) -> None:
        """台車を、向きを変えずにまっすぐ後退させる.

        物体を掴んだ直後は机や棚のすぐ上に腕があります。その場で
        move_to_go を呼ぶと腕が大きく振られて周囲に当たるので、
        先に離れます。

        go_rel は Nav2 の /move_base/move を使うため、プランナが
        「目標へ向く→走る→向き直す」経路を作り、把持姿勢のまま台車が
        回ってしまいます。follow_trajectory は地図を無視して台車
        コントローラへ直接軌道を渡すので、純粋な並進になります。
        HSR は全方向移動なので、後ろ向きのまま真後ろへ下がれます。

        follow_trajectory は sync=True にできません。hsrb_interface の
        MobileBase.follow_trajectory が wait_controllers へ Node ではなく
        自分自身を渡しており、中の rclpy.spin_once で必ず
        'MobileBase' object has no attribute '_subscriptions' になります。
        非同期で送ってから、こちらのノードを回しつつ完了を待ちます。

        後退できなくても把持自体は成功しているので、失敗しても続けます。

        Args:
            distance: 後退する距離 [m]。
        """
        self.node.get_logger().info(
            f'台車を向きを変えずに {distance:.3f} m 後退させます。'
        )
        try:
            self.hsrif.omni_base.follow_trajectory(
                [geometry.pose(x=-distance, y=0.0, ek=0.0)],
                ref_frame_id=BASE_FRAME,
                sync=False,
            )
        # 台車コントローラの例外型は環境で異なります。
        except Exception as error:
            self.node.get_logger().warning(
                f'台車を後退させられませんでした: {error}'
            )
            return

        deadline = time.monotonic() + BACKUP_TIMEOUT
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            try:
                if not self.hsrif.omni_base.is_moving():
                    break
            # 状態を取れないときは待つ意味が無いので抜けます。
            except Exception as error:
                self.node.get_logger().warning(
                    f'後退の状態を取得できませんでした: {error}'
                )
                return
        else:
            self.node.get_logger().warning(
                f'後退が {BACKUP_TIMEOUT:.0f} 秒で完了しませんでした。'
            )
            return

        self.node.get_logger().info('後退が完了しました。')

    def execute(self, blackboard: Blackboard) -> str:
        """把持前姿勢へ移動し、物体を掴んで走行姿勢へ戻る."""
        self.node.get_logger().info('Executing state Grasp')

        if 'grasp_pose' not in blackboard or 'grasp_approach' not in blackboard:
            self.node.get_logger().error('Blackboard に把持姿勢がありません。')
            return 'failed'

        # 上把持のときだけ設定されます。横把持では姿勢に向きが入っています。
        target_hand_yaw = (
            blackboard.grasp_wrist_roll
            if 'grasp_wrist_roll' in blackboard
            else None
        )

        # 動作は必ず1つずつ完了させます。重ねると、握りの浅い平たい物体は
        # 閉じ切る前に腕が動いて弾き飛ばされます。
        try:
            self._step('グリッパを開く')
            self.hsrif.gripper.command(1.0, sync=True)

            self._step('把持前姿勢へ移動')
            self.hsrif.whole_body.move_end_effector_pose(
                blackboard.grasp_pose,
                'base_link',
                sync=True,
            )

            if target_hand_yaw is not None:
                self._step('長辺方向に閉じる向きへ手首を回す')
                self._rotate_wrist_to_long_edge(target_hand_yaw)

            # 手首を回しても手のひらの向き (ローカル +Z) は真下のままです。
            self._step(f'{blackboard.grasp_approach:.3f} m 手を伸ばす')
            self.hsrif.whole_body.move_end_effector_by_line(
                (0, 0, 1),
                blackboard.grasp_approach,
                sync=True,
            )

            # 降下の揺れが止まってから閉じます。
            close_angle = GRIPPER_CLOSE_ANGLES.get(
                blackboard.target_name.lower(),
                GRIPPER_CLOSE_ANGLE,
            )
            self._step(f'グリッパを閉じる (目標 {close_angle:.2f} rad)')
            self.hsrif.gripper.command(
                close_angle,
                GRIPPER_CLOSE_TIME,
                sync=True,
            )

            # 走行姿勢へ移る前に、握ったままその場で真上へ引き上げます。
            self._step(f'{LIFT_DISTANCE:.3f} m 真上へ引き上げる')
            self._lift(LIFT_DISTANCE)

            # 上把持は机や棚のすぐ上で掴んでいるので、腕を振る前に離れます。
            # 横把持は既に正面から水平に入っているので後退しません。
            if target_hand_yaw is not None:
                self._step(f'台車を {BACKUP_DISTANCE:.3f} m 後退させる')
                self._back_up(BACKUP_DISTANCE)

            self._step('走行姿勢へ戻る')
            self.hsrif.whole_body.move_to_go(sync=True)
        # hsrb_interface の動作例外をステート失敗へ変換します。
        except Exception as error:
            self.node.get_logger().error(f'把持動作に失敗しました: {error}')
            return 'failed'

        self.node.get_logger().info(f'{blackboard.target_name} を把持しました。')
        return 'succeeded'
