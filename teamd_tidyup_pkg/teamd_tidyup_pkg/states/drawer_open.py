#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""画像認識でつまみを掴み、台車の後退で引き出しを開けるタスク。"""

import math
import time
from dataclasses import dataclass

import numpy as np
import rclpy
import tf2_ros
import tf_transformations as tft
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState
from yasmin import Blackboard
from yasmin import State
from yolov8_detection_interfaces.srv import ObjectDetectionService

from carrobo_manipulation_pkg.hsrif import HSRInterfaces


DRAWER_PARAMETER_DEFAULTS = {
    # 棚への接近
    'first_forward_distance': 1.10,
    'right_turn_degrees': 90.0,
    'second_forward_distance': 0.90,
    'navigation_timeout': 60.0,
    'require_entrance_start': True,
    'entrance_tolerance': 0.25,

    # つまみの検出
    'knob_service': '/drawer_knob_detection/service',
    'knob_confidence': 0.03,
    # 実物は 36-45 px。12-22 px の誤検出を除く。
    'knob_min_box_pixels': 25.0,
    'knob_max_box_pixels': 120.0,
    # つまみ検出時の首角度。YOLOEはこの角度だけで実行する。
    'knob_search_tilts': [-0.35],
    # 最初の検出だけ、より下向きにする。
    'first_detection_tilt': -0.55,
    # 同じ首角度で複数フレームを取得し、外れ値を中央値で抑える。
    'knob_detection_samples': 3,
    'knob_min_detection_votes': 2,
    'knob_detection_sample_interval': 0.20,
    'knob_max_distance': 2.0,
    'knob_min_height': 0.10,
    'knob_max_height': 0.95,
    # マスク位置から実際に挟む軸までの補正値。
    'knob_depth_bias': 0.020,
    'knob_height_bias': -0.010,
    'detection_settle_seconds': 1.5,
    # 同一つまみとみなす距離（検出結果の整理用）。
    'same_knob_radius': 0.12,
    # 把持前の再検出で、最初に選んだつまみと照合するための最大距離。
    'final_alignment_match_radius': 0.10,
    'final_alignment_detection': True,
    'knob_level_split': 0.40,
    # 一番奥のつまみより手前にある、開いた引き出しを除く。
    'opened_protrusion': 0.08,

    # base_footprint 基準で実測した把持姿勢
    'grasp_reach_x': 0.6858,
    'grasp_offset_y': 0.0792,
    'grasp_height_at_zero_lift': 0.2513,
    'arm_lift_gain': 0.945,
    'arm_lift_min': 0.0,
    'arm_lift_max': 0.66,
    'preparation_arm_flex': -1.7500,
    'preparation_wrist_flex': 0.1792,
    'preparation_wrist_roll': 0.10,
    # 上側の縁を避けてから腕を伸ばし、台車を前進させて把持する距離。
    'grasp_preparation_backoff': 0.10,
    # 指先が目標高さに届かなければ姿勢を再送する。
    'arm_settle_tolerance': 0.010,
    'arm_settle_attempts': 3,

    # 台車
    'align_tolerance': 0.008,
    'align_speed': 0.06,
    'align_timeout': 25.0,

    # 把持と引き出し
    'gripper_open_angle': 0.90,
    'gripper_open_duration': 0.70,
    'gripper_close_angle': 0.0,
    'gripper_close_duration': 2.0,
    # 閉じ角度との差で空振りか判定する。
    'grasp_angle_margin': 0.03,
    'grasp_settle_seconds': 0.50,
    # 把持確認による失敗判定は一旦無効化する。
    'verify_grasp': False,
    'pull_distance': 0.20,
    'base_pull_speed': 0.08,
    'base_motion_weight': 100.0,
    'retreat_distance': 0.10,

    # 手順
    'max_drawers': 3,
    'grasp_attempts': 2,
    # 右側のドロワーを優先する回数。右側は2つで、その後は高さを優先する。
    'right_priority_drawers': 2,
    # base_footprint の +Y 側を左側とみなす境界 [m]。
    'left_side_min_y': 0.0,
}

DEPTH_SCALES = {'16UC1': 0.001, '32FC1': 1.0}


def wrap_angle(angle: float) -> float:
    """角度を [-pi, pi] に正規化する。"""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class Knob:
    """検出した引き出しのつまみ1つ。base_footprint と odom の両方で保持する。"""

    x: float
    y: float
    z: float
    score: float
    odom: tuple = None

    def __post_init__(self):
        """座標とスコアを通常の float に揃える。"""
        self.x, self.y, self.z, self.score = map(
            float, (self.x, self.y, self.z, self.score)
        )

    def __repr__(self):
        """ログ用の短い表示を返す。"""
        return (f'Knob(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, '
                f'score={self.score:.3f})')


class DrawerOpenTask:
    """既存の ROS ノードと HSR インターフェースの上で動く引き出しタスク。"""

    def __init__(
        self,
        node: Node,
        hsrif: HSRInterfaces = None,
        tf_buffer=None,
    ) -> None:
        """共有の ROS / ロボットインターフェースにタスクを接続する。"""
        self.node = node
        self.logger = node.get_logger()
        for name, default in DRAWER_PARAMETER_DEFAULTS.items():
            self.node.declare_parameter(name, default)
        if tf_buffer is None:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(
                self.tf_buffer, self.node
            )
        else:
            self.tf_buffer = tf_buffer
            self.tf_listener = None
        self.base_cmd_pub = self.node.create_publisher(
            Twist, '/omni_base_controller/cmd_vel', 10
        )
        self.bridge = CvBridge()
        self._joint_state = None
        self.node.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10
        )
        self.knob_client = self.node.create_client(
            ObjectDetectionService, str(self._value('knob_service'))
        )
        self.hsrif = hsrif

    def _on_joint_state(self, msg) -> None:
        self._joint_state = msg

    def _value(self, name: str):
        return self.node.get_parameter(name).value

    def _float(self, name: str) -> float:
        return float(self._value(name))

    @staticmethod
    def _rotate(x: float, y: float, yaw: float):
        """2次元座標を yaw だけ回転する。"""
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        return x * cos_yaw - y * sin_yaw, x * sin_yaw + y * cos_yaw

    def _spin(self, seconds: float) -> None:
        """指定した実時間だけサービスのコールバックと TF を回す。"""
        end = time.monotonic() + max(0.0, seconds)
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def stop_base(self) -> None:
        """後始末や引き渡しのために、速度ゼロを明示的に送る。"""
        self.base_cmd_pub.publish(Twist())

    def _base_pose(self):
        """base_footprint の現在の odom 姿勢を返す。"""
        last_error = None
        for _ in range(30):
            try:
                transform = self.tf_buffer.lookup_transform(
                    'odom', 'base_footprint', rclpy.time.Time()
                )
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                yaw = tft.euler_from_quaternion(
                    [rotation.x, rotation.y, rotation.z, rotation.w]
                )[2]
                return translation.x, translation.y, yaw
            except Exception as error:
                last_error = error
                rclpy.spin_once(self.node, timeout_sec=0.1)
        raise RuntimeError(f'odom/base_footprint TF を取得できません: {last_error}')

    def _drive_base(
        self, forward: float, left: float, label: str, speed: float = None
    ) -> None:
        """odom の位置を見ながら台車を平行移動する。"""
        tolerance = self._float('align_tolerance')
        if math.hypot(forward, left) < tolerance:
            return
        speed = abs(self._float('align_speed') if speed is None else speed)
        timeout = self._float('align_timeout')
        start_x, start_y, start_yaw = self._base_pose()
        offset_x, offset_y = self._rotate(forward, left, start_yaw)
        goal_x, goal_y = start_x + offset_x, start_y + offset_y
        self.logger.info(
            f'{label}: 前後 {forward:+.3f} m, 左右 {left:+.3f} m 台車を平行移動します。'
        )
        started = time.monotonic()
        command = Twist()
        try:
            while rclpy.ok() and time.monotonic() - started < timeout:
                current_x, current_y, yaw = self._base_pose()
                error_x, error_y = goal_x - current_x, goal_y - current_y
                distance = math.hypot(error_x, error_y)
                if distance < tolerance:
                    break
                base_forward, base_left = self._rotate(error_x, error_y, -yaw)
                gain = min(1.0, distance / 0.05)
                command.linear.x = speed * gain * base_forward / distance
                command.linear.y = speed * gain * base_left / distance
                self.base_cmd_pub.publish(command)
                rclpy.spin_once(self.node, timeout_sec=0.02)
            else:
                self.logger.warn(f'{label}: 整定前にタイムアウトしました。')
        finally:
            self.base_cmd_pub.publish(Twist())

    def _approach(self) -> None:
        """初期 odom 姿勢から 前進 -> 右回転 -> 前進 で棚の前まで移動する。"""
        start_x, start_y, start_yaw = self._base_pose()
        if bool(self._value('require_entrance_start')):
            tolerance = self._float('entrance_tolerance')
            if math.hypot(start_x, start_y) > tolerance or abs(start_yaw) > 0.20:
                raise RuntimeError(
                    'ロボットが入口の初期位置にいません。'
                    'Isaac Sim をリセットしてから再実行してください。'
                )
        timeout = self._float('navigation_timeout')

        def go(x, y, yaw, label):
            self.logger.info(f'{label}: odom=({x:.3f}, {y:.3f})')
            reached = self.hsrif.omni_base.go_abs(
                x, y, yaw, timeout=timeout, sync=True
            )
            if reached is False:
                raise RuntimeError(f'{label} へ到達できませんでした。')

        first = self._float('first_forward_distance')
        offset_x, offset_y = self._rotate(first, 0.0, start_yaw)
        first_x, first_y = start_x + offset_x, start_y + offset_y
        go(first_x, first_y, start_yaw, '1/3 前進')
        turn_yaw = wrap_angle(
            start_yaw - math.radians(self._float('right_turn_degrees'))
        )
        go(first_x, first_y, turn_yaw, '2/3 右回転')
        second = self._float('second_forward_distance')
        offset_x, offset_y = self._rotate(second, 0.0, turn_yaw)
        go(
            first_x + offset_x,
            first_y + offset_y,
            turn_yaw,
            '3/3 前進',
        )
        self.logger.info('棚の前に到着しました。ここからは検出で位置を合わせます。')

    def _look(self, tilt: float) -> None:
        """腕でカメラを塞がないようにしたまま、首を棚に向ける。"""
        self.hsrif.whole_body.move_to_joint_positions(
            {'head_pan_joint': 0.0, 'head_tilt_joint': float(tilt)}, sync=True
        )
        self._spin(self._float('detection_settle_seconds'))

    def _detect_once(self):
        """つまみ検出サービスを1回呼び、マスクを base_footprint に投影する。"""
        service = str(self._value('knob_service'))
        if not self.knob_client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError(f'{service} が見つかりません。')
        request = ObjectDetectionService.Request()
        request.confidence_th = self._float('knob_confidence')
        future = self.knob_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=30.0)
        response = future.result()
        if response is None:
            raise RuntimeError(f'{service} から応答がありません。')
        detections = response.detections
        if not detections.bbox:
            return []

        scale = DEPTH_SCALES.get(detections.depth.encoding)
        if scale is None:
            raise RuntimeError(
                f'未対応の深度エンコーディングです: {detections.depth.encoding}'
            )
        depth = self.bridge.imgmsg_to_cv2(
            detections.depth, desired_encoding=detections.depth.encoding
        ).astype(np.float64) * scale
        matrix = detections.camera_info.k
        fx, fy, cx, cy = matrix[0], matrix[4], matrix[2], matrix[5]
        if fx == 0.0 or fy == 0.0:
            raise RuntimeError('camera_info が空です。')

        transform = self.tf_buffer.lookup_transform(
            'base_footprint', detections.camera_info.header.frame_id,
            rclpy.time.Time(),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        camera_to_base = tft.quaternion_matrix(
            [rotation.x, rotation.y, rotation.z, rotation.w]
        )
        camera_to_base[:3, 3] = [translation.x, translation.y, translation.z]

        min_box = self._float('knob_min_box_pixels')
        max_box = self._float('knob_max_box_pixels')
        max_distance = self._float('knob_max_distance')
        min_height = self._float('knob_min_height')
        max_height = self._float('knob_max_height')

        knobs = []
        for bbox, segment in zip(detections.bbox, detections.segments):
            width, height = float(bbox.w), float(bbox.h)
            if not (min_box <= width <= max_box and min_box <= height <= max_box):
                continue
            mask = self.bridge.imgmsg_to_cv2(
                segment, desired_encoding='mono8'
            )
            if mask.shape[:2] != depth.shape[:2]:
                continue
            valid = (
                (mask != 0) & np.isfinite(depth)
                & (depth > 0.10) & (depth <= max_distance)
            )
            ys, xs = np.nonzero(valid)
            if ys.size < 20:
                continue
            distances = depth[ys, xs]
            # マスクはつまみの丸い面全体に広がる。その手前半分が前面の板で、
            # 位置の基準にするのはこの面。
            near = distances <= np.percentile(distances, 40.0)
            ys, xs, distances = ys[near], xs[near], distances[near]
            points = np.column_stack(
                ((xs - cx) * distances / fx,
                 (ys - cy) * distances / fy,
                 distances)
            )
            in_base = (camera_to_base[:3, :3] @ points.T).T + camera_to_base[:3, 3]
            centre = in_base.mean(axis=0)
            if min_height <= centre[2] <= max_height:
                knobs.append(Knob(centre[0], centre[1], centre[2], bbox.score))
        return knobs

    def _detect_knobs(self, first_detection: bool = False):
        """首を固定して複数回検出し、中央値でodom上の候補を整理する。"""
        radius = self._float('same_knob_radius')
        clusters = []
        if first_detection:
            search_tilts = [self._float('first_detection_tilt')]
            self.logger.info(
                '1回目のドロワー検出では、首をより下向きにして実行します。'
            )
        else:
            configured_tilts = list(self._value('knob_search_tilts'))
            if not configured_tilts:
                raise RuntimeError('knob_search_tilts が空です。')
            # 複数角度の探索は行わず、先頭の角度でだけ検出する。
            search_tilts = [configured_tilts[0]]
        for tilt in search_tilts:
            self._look(float(tilt))
            samples = max(1, int(self._value('knob_detection_samples')))
            for sample_index in range(samples):
                try:
                    knobs = self._detect_once()
                except Exception as error:
                    self.logger.warn(
                        f'tilt={float(tilt):.2f} の検出 '
                        f'({sample_index + 1}/{samples}) に失敗: {error}'
                    )
                    continue
                for knob in knobs:
                    anchored = self._to_odom(knob)
                    for cluster in clusters:
                        if math.dist(cluster[0].odom, anchored.odom) < radius:
                            cluster.append(anchored)
                            break
                    else:
                        clusters.append([anchored])
                if sample_index + 1 < samples:
                    self._spin(self._float('knob_detection_sample_interval'))

        base_x, base_y, yaw = self._base_pose()
        fused = []
        min_votes = max(1, int(self._value('knob_min_detection_votes')))
        for cluster in clusters:
            if len(cluster) < min_votes:
                self.logger.info(
                    f'つまみ候補を除外します: {len(cluster)} フレームだけで検出 '
                    f'(必要 {min_votes} フレーム)'
                )
                continue
            # YOLO のマスクが棚の縁まで広がると、score にかかわらず大きな
            # 位置外れ値になる。各軸の中央値なら、その1フレームに把持姿勢を
            # 引っ張られず、手前からの再検出でも安定する。
            centre = np.median([knob.odom for knob in cluster], axis=0)
            # まとめた odom 上の点を、現在の台車座標系に戻す。
            dx, dy = centre[0] - base_x, centre[1] - base_y
            x, y = self._rotate(dx, dy, -yaw)
            knob = Knob(x, y, centre[2], max(item.score for item in cluster))
            knob.odom = tuple(centre)
            fused.append(knob)
        self.logger.info(f'つまみ {len(fused)} 個を検出 -> {fused}')
        return fused

    def _to_odom(self, knob: Knob) -> Knob:
        """台車が動いても追えるように、つまみを odom 上に固定する。"""
        base_x, base_y, yaw = self._base_pose()
        x, y = self._rotate(knob.x, knob.y, yaw)
        knob.odom = (
            base_x + x,
            base_y + y,
            knob.z,
        )
        return knob

    def _in_current_base(self, knob: Knob):
        """odom に固定したつまみを、今いる台車座標系で表す。"""
        base_x, base_y, yaw = self._base_pose()
        dx, dy = knob.odom[0] - base_x, knob.odom[1] - base_y
        x, y = self._rotate(dx, dy, -yaw)
        return x, y, knob.odom[2]

    def _select_knob(self, knobs, prefer_right: bool = True):
        """開いた候補を除き、右または高さを優先してつまみを選ぶ。"""
        if not knobs:
            return None
        split = self._float('knob_level_split')
        margin = self._float('opened_protrusion')
        groups = (
            ('下段', [knob for knob in knobs if knob.z < split]),
            ('上段', [knob for knob in knobs if knob.z >= split]),
        )
        available = []
        for label, group in groups:
            if not group:
                continue
            # 開いた引き出しはつまみを手前へ連れてくるため、同じ段の中で
            # 一番手前に出た候補を開放済みとして除外する。
            deepest = max(knob.x for knob in group)
            closed = [knob for knob in group if knob.x > deepest - margin]
            if len(closed) < len(group):
                self.logger.info(
                    f'{label}: 手前に出ている {len(group) - len(closed)} 個は '
                    '開いた引き出しとみなして除外します。'
                )
            for knob in closed:
                available.append((knob.y, knob.z, label, knob))

        if not available:
            return None

        if prefer_right:
            # 台車座標の y が小さい側が右側。
            _, _, label, selected = min(
                available, key=lambda item: (item[0], item[1])
            )
            selection_rule = '右側優先'
        else:
            # 右側を2つ開けた後の3つ目は左側だけを候補にし、その中で
            # 高さ(z)の低いものを優先する。
            left_min_y = self._float('left_side_min_y')
            left_available = [
                item for item in available if item[0] > left_min_y
            ]
            if not left_available:
                self.logger.warn('左側のつまみが見つかりません。')
                return None
            _, _, label, selected = min(
                left_available, key=lambda item: (item[1], item[0])
            )
            selection_rule = '左側かつ低い位置優先'
        self.logger.info(
            f'{label}から{selection_rule}でつまみを選択します: '
            f'y={selected.y:.3f}, z={selected.z:.3f}'
        )
        return selected

    def _fingertip(self):
        """base_footprint 基準で、2本の指先の中点を返す。"""
        try:
            points = []
            for frame in ('hand_l_finger_tip_frame', 'hand_r_finger_tip_frame'):
                translation = self.tf_buffer.lookup_transform(
                    'base_footprint', frame, rclpy.time.Time()
                ).transform.translation
                points.append([translation.x, translation.y, translation.z])
            return np.mean(points, axis=0)
        except Exception as error:
            self.logger.warn(f'指先 TF を取得できません: {error}')
            return None

    def _set_arm(self, knob: Knob) -> None:
        """ハンドを開き、つまみの高さに合わせた較正済み姿勢を保つ。"""
        corrected_z = knob.z + self._float('knob_height_bias')
        base_height = self._float('grasp_height_at_zero_lift')
        gain = self._float('arm_lift_gain')
        lift = min(
            max((corrected_z - base_height) / gain, self._float('arm_lift_min')),
            self._float('arm_lift_max'),
        )
        target_z = base_height + gain * lift
        self.logger.info(
            f'つまみ高さ {knob.z:.3f} m (補正 {corrected_z:.3f} m) '
            f'-> arm_lift={lift:.3f}'
        )
        self.hsrif.gripper.command(
            self._float('gripper_open_angle'),
            self._float('gripper_open_duration'),
        )
        self._spin(0.5)
        # 軌道コントローラはときどき手前で止まるので、指が本当に指令した
        # 高さに来るまで同じ姿勢を送り直す。
        for attempt in range(int(self._value('arm_settle_attempts'))):
            self.hsrif.whole_body.move_to_joint_positions(
                {
                    'arm_lift_joint': lift,
                    'arm_flex_joint': self._float('preparation_arm_flex'),
                    'arm_roll_joint': 0.0,
                    'wrist_flex_joint': self._float('preparation_wrist_flex'),
                    'wrist_roll_joint': self._float('preparation_wrist_roll'),
                },
                sync=True,
            )
            self._spin(1.0)
            tip = self._fingertip()
            if tip is None:
                return
            error = abs(float(tip[2]) - target_z)
            self.logger.info(
                f'腕の整定 {attempt + 1}: 指先 z={tip[2]:.3f} '
                f'(目標 {target_z:.3f}, ずれ {error:.4f} m)'
            )
            if error <= self._float('arm_settle_tolerance'):
                return

    def _refine_knob_for_grasp(self, original: Knob) -> Knob:
        """把持直前に同じつまみだけを再検出し、位置を更新する。"""
        if not bool(self._value('final_alignment_detection')):
            return original
        try:
            candidates = self._detect_knobs()
        except Exception as error:
            self.logger.warn(f'把持前の再検出に失敗したため初回座標を使います: {error}')
            return original
        if not candidates:
            self.logger.warn('把持前の再検出で候補が無いため初回座標を使います。')
            return original
        refined = min(
            candidates, key=lambda candidate: math.dist(
                candidate.odom, original.odom
            )
        )
        delta = math.dist(refined.odom, original.odom)
        if delta > self._float('final_alignment_match_radius'):
            self.logger.warn(
                f'把持前の候補が初回位置から {delta:.3f} m 離れているため、'
                '別のつまみとみなして初回座標を使います。'
            )
            return original
        self.logger.info(
            f'把持前の再検出でつまみを更新します: {delta:.3f} m 補正'
        )
        return refined

    def _grasp(self, knob: Knob) -> bool:
        """少し後退して腕を伸ばし、前進して検出したつまみを掴む。"""
        bias = self._float('knob_depth_bias')
        backoff = max(0.0, self._float('grasp_preparation_backoff'))
        knob_x, knob_y, _ = self._in_current_base(knob)
        grasp_forward = knob_x + bias - self._float('grasp_reach_x')
        grasp_left = knob_y - self._float('grasp_offset_y')

        # 腕を出す軌道で引き出し上側の縁に当たらないよう、把持位置より
        # 少し手前に台車を置いてから腕を伸ばす。
        self._drive_base(
            grasp_forward - backoff,
            grasp_left,
            '把持前に後退して上側の縁を回避',
        )

        # 台車を近づけると、初回検出の視差・深度誤差がそのまま把持誤差に
        # なる。ここで同じつまみだけを照合して、最後の base 移動量を更新する。
        knob = self._refine_knob_for_grasp(knob)
        knob_x, knob_y, _ = self._in_current_base(knob)
        grasp_forward = knob_x + bias - self._float('grasp_reach_x')
        grasp_left = knob_y - self._float('grasp_offset_y')
        self._set_arm(knob)

        # 腕を伸ばした姿勢を保ったまま、検出したつまみの位置まで
        # 台車を前進させて把持する。
        self._drive_base(
            grasp_forward, grasp_left,
            '腕を伸ばしたまま前進して把持位置へ',
        )
        self._log_residual(knob)
        self.logger.info('つまみを挟むためハンドを閉じます。')
        self.hsrif.gripper.command(
            self._float('gripper_close_angle'),
            self._float('gripper_close_duration'),
        )
        self._spin(self._float('grasp_settle_seconds'))
        return self._holds_knob()

    def _log_residual(self, knob: Knob) -> None:
        """検出したつまみに対して指先がどこに来たかをログに出す。"""
        tip = self._fingertip()
        if tip is None:
            return
        knob_x, knob_y, knob_z = self._in_current_base(knob)
        target = (
            knob_x + self._float('knob_depth_bias'),
            knob_y,
            knob_z + self._float('knob_height_bias'),
        )
        self.logger.info(
            f'把持直前: 指先=({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}) '
            f'つまみ=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) '
            f'残差=({tip[0] - target[0]:+.3f}, {tip[1] - target[1]:+.3f}, '
            f'{tip[2] - target[2]:+.3f})'
        )

    def _holds_knob(self) -> bool:
        """hand_motor_joint の角度から空振りか確認する。"""
        if not bool(self._value('verify_grasp')):
            return True
        self._spin(0.4)
        angle = None
        for _ in range(40):
            state = self._joint_state
            if state is not None and 'hand_motor_joint' in state.name:
                angle = float(state.position[state.name.index('hand_motor_joint')])
                break
            rclpy.spin_once(self.node, timeout_sec=0.05)
        if angle is None:
            self.logger.warn('hand_motor_joint を取得できず把持確認を省略します。')
            return True
        commanded = self._float('gripper_close_angle')
        held = angle > commanded + self._float('grasp_angle_margin')
        self.logger.info(
            f'把持確認: hand_motor={angle:+.4f} rad -> {"把持" if held else "空振り"}'
        )
        return held

    def _release(self) -> None:
        """台車も腕も動かさずにハンドだけ開く。"""
        self.hsrif.gripper.command(1.0, self._float('gripper_open_duration'))
        self._spin(0.5)

    def _open_one_drawer(self, knob: Knob) -> bool:
        """掴む -> 掴んだまま後退 -> 離す。"""
        if not self._grasp(knob):
            self.logger.warn('つまみを掴めませんでした。退避します。')
            self._release()
            self._drive_base(-self._float('retreat_distance'), 0.0, '失敗後の後退')
            return False
        self._drive_base(
            -max(0.0, self._float('pull_distance')), 0.0,
            '掴んだまま後退して引き出しを開ける',
            speed=self._float('base_pull_speed'),
        )
        self._release()
        self.logger.info('引き出しを1つ開けました。')
        return True

    def run_task(self) -> bool:
        """低い検出座標の順に 認識 -> 掴む -> 後退 -> 再認識を繰り返す。"""
        whole_body = None
        weights = None
        try:
            if self.hsrif is None:
                self.hsrif = HSRInterfaces()
            self._approach()

            # 全身プランナが台車を使って解かないようにする。このタスクの
            # 台車移動はすべて明示的に指令する。
            whole_body = self.hsrif.whole_body._wb
            weights = (whole_body.linear_weight, whole_body.angular_weight)
            weight = self._float('base_motion_weight')
            whole_body.linear_weight = whole_body.angular_weight = weight

            opened_count = 0
            attempts = max(1, int(self._value('grasp_attempts')))
            right_priority_drawers = max(
                0, int(self._value('right_priority_drawers'))
            )
            for drawer_index in range(int(self._value('max_drawers'))):
                for attempt in range(attempts):
                    # 認識のたびに腕をどける (move_to_go はカメラを腕で塞ぐ)。
                    self.hsrif.whole_body.move_to_neutral(sync=True)
                    knobs = self._detect_knobs(
                        first_detection=drawer_index == 0 and attempt == 0
                    )
                    target = self._select_knob(
                        knobs, prefer_right=drawer_index < right_priority_drawers
                    )
                    if target is None:
                        self.logger.warn(
                            '次のつまみが見つかりません。次へ進みます。'
                        )
                        break
                    self.logger.info(
                        f'{drawer_index + 1}個目の引き出しを開けます '
                        f'({attempt + 1}/{attempts}): {target}'
                    )
                    if self._open_one_drawer(target):
                        opened_count += 1
                        break

            if opened_count == 0:
                raise RuntimeError('引き出しを1つも開けられませんでした。')
            self._drive_base(
                -self._float('retreat_distance'), 0.0, '完了後の後退',
                speed=self._float('base_pull_speed'),
            )
            self.hsrif.whole_body.move_to_go(sync=True)
            self.stop_base()
            self.logger.info(f'{opened_count} 個の引き出しを開けました。')
            return True
        except Exception as error:
            self.logger.error(f'引き出し操作に失敗しました: {error}')
            # 棚から離れておく。つまみを掴んだままだと後続のステートが
            # すべて動けなくなる。
            try:
                self.stop_base()
                if self.hsrif is not None:
                    self._release()
                    self._drive_base(
                        -self._float('retreat_distance'), 0.0, '異常時の退避'
                    )
                    self.hsrif.whole_body.move_to_go(sync=True)
            except Exception as cleanup_error:
                self.logger.error(f'退避にも失敗しました: {cleanup_error}')
            return False
        finally:
            # このインターフェースは他メンバーの把持ステートでも使い回す。
            if whole_body is not None and weights is not None:
                try:
                    whole_body.linear_weight, whole_body.angular_weight = weights
                except Exception as error:
                    self.logger.warn(f'プランナ重みの復元に失敗しました: {error}')


class OpenDrawersState(State):
    """引き出しを全部開け、棚から離れてナビゲーションへ引き渡すステート。"""

    def __init__(self, node: Node, drawer_task: DrawerOpenTask):
        """共有ノードと使い回す引き出しタスクを保持する。"""
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = node
        self.drawer_task = drawer_task

    def execute(self, blackboard: Blackboard) -> str:
        """物体把持のステートに入る前に、引き出しの手順を1回実行する。"""
        self.node.get_logger().info('Executing state OpenDrawers')
        blackboard.drawers_opened = False
        if not self.drawer_task.run_task():
            self.node.get_logger().error(
                '引き出しタスクに失敗したため、物体把持へは進みません。'
            )
            return 'failed'
        blackboard.drawers_opened = True
        return 'succeeded'


class DrawerOpenNode(Node):
    """:class:`DrawerOpenTask` を単体実行するための薄い ROS ノード。"""

    def __init__(self) -> None:
        """open_drawer 実行ファイルが使う単体実行用ノードを作る。"""
        super().__init__('drawer_open')
        self.task = DrawerOpenTask(self)

    def run_task(self) -> bool:
        """使い回し可能な引き出しタスクを単体モードで実行する。"""
        return self.task.run_task()


def main(args=None) -> None:
    """ROS 2 のコンソールエントリポイント。"""
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
