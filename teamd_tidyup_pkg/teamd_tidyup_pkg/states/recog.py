#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOv8 の検出結果から把持姿勢を求めるステート."""

import math
import time

import cv2
import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401
import tf_transformations as tft
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from grasp_point_detection_interfaces.srv import GraspPointService
from hma_object_detection2_interfaces.srv import (
    ObjectDetectionService as RexObjectDetectionService,
)
from hma_rclpy_extension.cv_bridge import CvBridgeUtils
from rclpy.duration import Duration
from rclpy.node import Node
from std_srvs.srv import Trigger
from tf2_ros import Buffer
from tf2_ros import TransformException
from yasmin import Blackboard
from yasmin import State
from yolov8_detection_interfaces.msg import BBox as YoloBBox
from yolov8_detection_interfaces.msg import ObjectDetection
from yolov8_detection_interfaces.srv import ObjectDetectionService

from carrobo_manipulation_pkg.hsrif import HSRInterfaces


# 掴みたい物体名を指定します。
# 既存の把持例と同じ対象を初期値にしています。
# 空文字にするとロボットから最も近い物体を選びます。
# ロボット自身を選ぶ可能性があるため、通常は物体名を指定してください。
TARGET_NAME = ''

# 検出されても把持対象にしない物体名です。
# 比較は大文字小文字を無視するので、小文字で書いてください。
EXCLUDED_NAMES = {
    'rubiks',
    '9-peg',
    'pitcher base',
    'plate',
}

CONFIDENCE_THRESHOLD = 0.6
MAX_GRASP_DISTANCE = 2.0
# これより近い物体は把持候補にしません [m]。
# 足元にあると腕を振り込めず、台車やロボット自身と干渉します。
MIN_GRASP_DISTANCE = 0.8
DEPTH_SCALES = {
    '16UC1': 0.001,
    '32FC1': 1.0,
}
TALL_THRESHOLD = 0.15
# 横把持で、物体の手前どれだけの位置にプリグラスプを置くか [m]。
# 物体に近い位置で把持姿勢を作ると、腕を振り込む途中で物体に当たります。
# 遠くで姿勢を作ってからまっすぐ寄せるほうが当たりにくくなります。
SIDE_GRASP_PREGRASP_DISTANCE = 0.15
# 横把持で、プリグラスプからまっすぐ伸ばす距離 [m]。
# プリグラスプ距離との差が、hand_palm_link と把持点のすき間になります。
# 伸ばしすぎると指が物体を突き抜けて GOAL_IN_COLLISION になるので、
# 当たるようなら小さくしてください。
SIDE_GRASP_APPROACH_DISTANCE = 0.15
# プリグラスプは接近方向に沿ってロボット側へ下げるので、下げすぎると
# 手が base_link に近づきすぎ、ロボット自身と干渉して
# START_STATE_IN_COLLISION になります。これより近くには置きません。
MIN_PREGRASP_RADIUS = 0.40
# 把持位置の床からの高さ [m] の下限です。base_link の z=0 が床面なので、
# 推定値をそのまま使うと平たい物体でグリッパが床に食い込みます。
MIN_GRASP_HEIGHT = 0.0215
# 長辺方向に沿って把持位置を測り直す物体と、中点からさらに細い側へ
# 寄せる量 [m] です。キーは小文字で書いてください。
#
# 把持点推定サービスが返す位置は有効深度点の重心です。スプーンのように
# 片側 (受け皿) の面積が大きい物体では重心がそちらへ引っ張られ、端を
# 掴んで落とします。長辺方向の広がりの中点なら形が偏っていても物体の
# 真ん中になるので、そこを基準にして細い側 (柄) へ寄せます。
#
# ここに無い物体は、サービスが返す重心をそのまま使います。
# 左右対称な物体 (Bowl など) では細い側を判定できないので、その場合は
# ロボット側の縁へ寄せます。負の値にすると奥側の縁へ寄ります。
GRASP_LONG_AXIS_OFFSETS = {
    'spoon': 0.040,
    'bowl': 0.030,
}
# 上把持で、手首を回すときの高さを物体ごとに追加で上げる量 [m] です。
# お椀のように縁が高い物体は、把持点の 10 cm 上でも手首を回すと縁に
# 当たります。ここで上げた分は降りる距離にも足すので、最終的に掴む
# 位置は変わりません。ここに無い物体は既定の 10 cm のままです。
# キーは小文字で書いてください。
WRIST_ROTATION_EXTRA_LIFT = {
    'bowl': 0.050,
}
# 長辺方向の広がりを測るときに、両端で無視する割合 [%]。
# マスクのはみ出しで端が伸びても中点がずれないようにします。
LONG_AXIS_TRIM_PERCENT = 2.0
# 長辺の前後で短辺方向の幅がこれ以下しか違わなければ、左右対称とみなして
# 細い側の判定を諦めます [m]。ノイズで向きが毎回反転するのを防ぎます。
SYMMETRIC_WIDTH_MARGIN = 0.010
# wrist_roll_joint の可動域 [rad] (hsrb_description の URDF より)。
# whole_body.joint_limits が引けなかったときのフォールバックです。
WRIST_ROLL_LIMITS = (-1.92, 3.67)
HEAD_TILT = math.radians(-50.0)
# 認識するときのグリッパの開き角 [rad]。0.0 で閉じます。
# 開いたままだと指がカメラの視界に入ります。Grasp が最初に開くので、
# ここで閉じておいても把持には影響しません。
RECOG_GRIPPER_ANGLE = 0.0
# YOLO がこの回数だけ連続で物体を見つけられなかったら Rex-Omni へ移ります。
# 1 なら 1 回目の未検出で即座に Rex-Omni を試します。
# 下の向き直しを使い切ってからこの回数を数えます。
MAX_EMPTY_DETECTIONS = 1

# YOLO が空振りしたとき、その場で体の向きを変えてもう一度試す部屋です。
ROTATION_RETRY_ROOMS = {'roomA'}
# 1 つの部屋で向きを変えて試す回数です。
MAX_ROTATION_RETRIES = 1
# 1 回あたりの回頭量 [rad]。正で左回り、負で右回りです。
ROTATION_RETRY_YAW = math.radians(60.0)

# False にすると Rex-Omni を一切呼ばず、YOLO と下の正面向き再試行だけで
# 探索します。モデルのロードを伴わないぶん速く終わります。
USE_REX_OMNI = False

# 他で駄目だったとき、机の上を見るために首を正面に向けて YOLO を
# 1度だけ試す部屋です。
FORWARD_RETRY_ROOMS = {'roomA', 'roomB'}
# そのときのヘッドの傾き [rad]。0.0 で正面です。全部屋で共通です。
FORWARD_HEAD_TILT = 0.0
# 机の上を見に行くときの立ち位置 (map 座標系) を部屋ごとに指定します。
# 指定が無い部屋では移動せず、回した分だけ戻して 0 度で見ます。
FORWARD_RETRY_POSES = {
    'roomB': {'x': 6.9, 'y': 3.59, 'yaw': 0.0,
    },
}

# 検出できなかったときに移動して観測し直す位置です (map 座標系)。
# 上から順に1回ずつ使います。部屋の指定が無ければ移動しません。
EXTRA_VIEWPOINTS = {
    'roomB': [
        {'x': 6.848, 'y':  4.283, 'yaw': -1.060},
    ],
}
# 観測位置への移動の制限時間 [s]。0.0 は到着するまで待ちます。
VIEWPOINT_NAVIGATION_TIMEOUT = 0.0
REX_OMNI_PROMPT = 'object'
REX_OMNI_DETECTION_SERVICE = '/rex_omni_sam2/object_detection'
REX_OMNI_START_SERVICE = '/rex_omni/start'
REX_OMNI_STOP_SERVICE = '/rex_omni/stop'
REX_OMNI_SERVICE_TIMEOUT = 10.0
# TF 変換を待つ最大時間 [s]。
TF_WAIT_TIMEOUT = 2.0
# TF 変換の失敗だけで候補が無くなったとき、画像を撮り直して選び直す回数。
# 撮り直すとタイムスタンプが更新されるので、時刻ズレが原因なら解消します。
MAX_TF_RETRY_DETECTIONS = 2


def _horizontal_long_axis(box_orientation: Quaternion):
    """物体の水平な長辺方向の単位ベクトルを返す.

    把持点推定サービスは、箱のローカル X 軸を長辺、Y 軸を短辺と定めています。
    上把持では、この向きをグリッパの閉じ方向に合わせて短辺側の面を挟みます。
    """
    box_quaternion = np.array(
        [
            box_orientation.x,
            box_orientation.y,
            box_orientation.z,
            box_orientation.w,
        ],
        dtype=np.float64,
    )
    box_quaternion /= np.linalg.norm(box_quaternion)

    long_axis = tft.quaternion_matrix(box_quaternion)[:3, 0].copy()
    long_axis[2] = 0.0
    norm = np.linalg.norm(long_axis)
    if norm < np.finfo(np.float64).eps:
        raise ValueError('物体の水平長辺方向を計算できません。')
    return long_axis / norm


def _horizontal_short_axis(box_orientation: Quaternion):
    """物体の水平な短辺方向の単位ベクトルを返す."""
    box_quaternion = np.array(
        [
            box_orientation.x,
            box_orientation.y,
            box_orientation.z,
            box_orientation.w,
        ],
        dtype=np.float64,
    )
    box_quaternion /= np.linalg.norm(box_quaternion)

    short_axis = tft.quaternion_matrix(box_quaternion)[:3, 1].copy()
    short_axis[2] = 0.0
    norm = np.linalg.norm(short_axis)
    if norm < np.finfo(np.float64).eps:
        raise ValueError('物体の水平短辺方向を計算できません。')
    return short_axis / norm


def _side_grasp_orientation(approach_xy):
    """指定した水平軸方向から接近する横把持の姿勢を作る.

    背の高い物体では短辺方向から接近し、グリッパの閉じ方向を長辺方向に
    合わせます。これにより、グリッパの指が物体の短辺側の面に接触します。

    できあがる手先姿勢は rpy(pi, -pi/2, yaw) で、
    ローカル +Z (接近方向) が物体の方向、
    ローカル Y (グリッパが閉じる方向) が水平でそれに直交、
    ローカル X が鉛直上向きになります。

    Args:
        approach_xy: base_link 基準での接近方向の水平成分 (x, y)。

    Returns:
        (手先姿勢, 接近方向の単位ベクトル)。
    """
    direction = np.array(
        [float(approach_xy[0]), float(approach_xy[1]), 0.0],
        dtype=np.float64,
    )
    norm = np.linalg.norm(direction)
    if norm < np.finfo(np.float64).eps:
        raise ValueError('物体がロボットの真上または真下にあります。')
    direction /= norm

    yaw = math.atan2(direction[1], direction[0])
    quaternion = tft.quaternion_from_euler(math.pi, -math.pi / 2.0, yaw)
    orientation = Quaternion(
        x=float(quaternion[0]),
        y=float(quaternion[1]),
        z=float(quaternion[2]),
        w=float(quaternion[3]),
    )
    return orientation, direction


class RecogState(State):
    """物体検出と把持点推定を行うステート."""

    def __init__(
        self,
        node: Node,
        hsrif: HSRInterfaces,
        tf_buffer: Buffer,
        nav=None,
    ):
        """サービスクライアントを生成する.

        Args:
            node: このステートを動かすノード。
            hsrif: ロボットのインターフェース。
            tf_buffer: カメラ座標系から base_link へ変換するための Buffer。
            nav: 観測位置へ移動するための NavModule。None なら移動しません。
        """
        super().__init__(outcomes=['succeeded', 'failed', 'none'])
        self.node = node
        self.hsrif = hsrif
        self.tf_buffer = tf_buffer
        self.nav = nav
        self.empty_detection_count = 0
        # 向き直しは部屋ごとに数え直すので、いまいる部屋を覚えておきます。
        self.current_room = None
        self.rotation_retry_count = 0
        self.forward_retry_done = False
        # この部屋で既に把持対象にした物体名です (小文字)。
        # 把持に失敗しても同じ物体を掴み直さないよう、候補から外します。
        self.attempted_names = set()
        # 直前の _select_target で TF 変換に失敗した候補があったかどうか。
        self.tf_error_in_selection = False
        # この部屋で次に使う観測位置の添字です。
        self.viewpoint_index = 0
        # この部屋でこれまでに回した合計角度 [rad]。把持後に
        # Move2GraspPoint が固定姿勢へ戻すので、戻された分を掛け直します。
        self.applied_rotation = 0.0
        # この部屋で最後に移動した観測位置。把持後に Move2GraspPoint が
        # 既定の位置へ戻すので、次に来たときはここへ戻ります。
        self.current_viewpoint = None
        # 直前にこのステートが succeeded を返したか。真なら把持サイクルを
        # 一周して台車の位置と向きが戻されています。
        self.returned_after_grasp = False
        self.bridge = CvBridgeUtils()

        self.detect_client = self.node.create_client(
            ObjectDetectionService, '/yolov8_detection/service'
        )
        self.rex_detect_client = self.node.create_client(
            RexObjectDetectionService,
            REX_OMNI_DETECTION_SERVICE,
        )
        self.rex_start_client = self.node.create_client(
            Trigger,
            REX_OMNI_START_SERVICE,
        )
        self.rex_stop_client = self.node.create_client(
            Trigger,
            REX_OMNI_STOP_SERVICE,
        )
        self.grasp_client = self.node.create_client(
            GraspPointService, '/grasp_point_detection/service'
        )

    def _wait_for_service(self, client, service_name: str) -> bool:
        """サービスが利用可能になるまで待つ."""
        while rclpy.ok():
            if client.wait_for_service(timeout_sec=1.0):
                return True
            self.node.get_logger().info(
                f'{service_name} が見つかりません。起動を待っています...'
            )
        return False

    def _wait_for_rex_service(self, client, service_name: str) -> bool:
        """Rex-Omni関連サービスを有限時間だけ待つ."""
        if client.wait_for_service(timeout_sec=REX_OMNI_SERVICE_TIMEOUT):
            return True
        self.node.get_logger().error(
            f'{service_name} が {REX_OMNI_SERVICE_TIMEOUT:.0f} 秒以内に'
            '見つかりませんでした。'
        )
        return False

    def _set_rex_model_state(self, start: bool) -> bool:
        """Rex-Omniモデルをロードまたはアンロードする."""
        client = self.rex_start_client if start else self.rex_stop_client
        service_name = (
            REX_OMNI_START_SERVICE if start else REX_OMNI_STOP_SERVICE
        )
        operation = '起動' if start else '停止'
        if not self._wait_for_rex_service(client, service_name):
            return False

        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()
        if response is None:
            self.node.get_logger().error(
                f'Rex-Omniモデルの{operation}サービスから応答がありません。'
            )
            return False
        if not response.success:
            self.node.get_logger().error(
                f'Rex-Omniモデルの{operation}に失敗しました: '
                f'{response.message}'
            )
            return False

        self.node.get_logger().info(
            f'Rex-Omniモデルの{operation}完了: {response.message}'
        )
        return True

    def _decode_rex_depth(self, compressed_depth):
        """Rex-Omniの圧縮深度画像を復元する。使えなければ None を返す.

        depth_is_compressed=false で起動していると、Rex-Omni 側は
        cv2_to_compressed_imgmsg(dst_format='png') で詰め直すため、
        format が単なる 'png' になり深度のエンコーディングが失われます。
        その場合 8bit へ落ちていることがあり、深度としては使えません。
        """
        if not compressed_depth.data:
            self.node.get_logger().warning(
                'Rex-Omni検出結果に深度画像がありません。'
            )
            return None

        if 'compresseddepth' in compressed_depth.format.lower():
            depth = self.bridge.compressed_imgmsg_to_depth(compressed_depth)
        else:
            depth = cv2.imdecode(
                np.frombuffer(compressed_depth.data, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
        if depth is None:
            self.node.get_logger().warning(
                'Rex-Omniの圧縮深度画像を復元できませんでした '
                f'(format={compressed_depth.format!r})。'
            )
            return None

        depth = np.asarray(depth)
        if depth.dtype == np.uint16:
            depth_encoding = '16UC1'
        elif depth.dtype == np.float32:
            depth_encoding = '32FC1'
        else:
            # 8bit まで落ちた深度は mm 単位の距離として復元できません。
            self.node.get_logger().warning(
                'Rex-Omniの深度画像型に対応していません: '
                f'{depth.dtype} (format={compressed_depth.format!r})。'
            )
            return None

        depth_msg = self.bridge.cv2_to_imgmsg(
            depth,
            encoding=depth_encoding,
        )
        depth_msg.header = compressed_depth.header
        return depth_msg

    def _convert_rex_detections(
        self,
        detections,
        fallback_depth=None,
    ) -> ObjectDetection:
        """Rex-Omni+SAM2の圧縮画像結果を既存YOLO形式へ変換する.

        Args:
            detections: Rex-Omni+SAM2 の検出結果。
            fallback_depth: Rex-Omni の深度が使えなかったときに代わりに使う
                sensor_msgs/Image。YOLO 応答の深度を渡します。ロボットは
                Recog の間停止しているので、同じ画角のものとして扱えます。
        """
        converted = ObjectDetection()
        converted.header = detections.header
        converted.is_detected = detections.is_detected
        converted.camera_info = detections.camera_info

        if not detections.bbox:
            return converted

        depth_msg = self._decode_rex_depth(detections.depth)
        if depth_msg is None:
            if fallback_depth is None or not fallback_depth.data:
                raise ValueError(
                    'Rex-Omniの深度画像が使えず、代わりの深度もありません。'
                )
            self.node.get_logger().warning(
                'Rex-Omniの深度画像が使えないため、'
                'YOLO応答の深度画像で代用します '
                f'(encoding={fallback_depth.encoding})。'
            )
            depth_msg = fallback_depth

        if depth_msg.encoding not in DEPTH_SCALES:
            raise ValueError(
                '代用した深度画像のエンコーディングに対応していません: '
                f'{depth_msg.encoding}'
            )

        depth = self.bridge.imgmsg_to_cv2(
            depth_msg,
            desired_encoding=depth_msg.encoding,
        )
        converted.depth = depth_msg

        if len(detections.segments) < len(detections.bbox):
            raise ValueError(
                'Rex-Omni検出結果のbbox数よりmask数が少ないです。'
            )

        for rex_bbox, compressed_mask in zip(
            detections.bbox,
            detections.segments,
        ):
            bbox = YoloBBox()
            bbox.id = rex_bbox.id
            bbox.name = rex_bbox.name
            bbox.score = rex_bbox.score
            bbox.x = rex_bbox.x
            bbox.y = rex_bbox.y
            bbox.w = rex_bbox.w
            bbox.h = rex_bbox.h
            converted.bbox.append(bbox)

            mask = cv2.imdecode(
                np.frombuffer(compressed_mask.data, dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
            if mask is None:
                raise ValueError(
                    f'{rex_bbox.name} の圧縮maskを変換できません。'
                )
            if mask.shape[:2] != depth.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (depth.shape[1], depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask_msg = self.bridge.cv2_to_imgmsg(
                mask.astype(np.uint8),
                encoding='mono8',
            )
            mask_msg.header = compressed_mask.header
            converted.segments.append(mask_msg)

        return converted

    def _detect_with_rex_omni(self, fallback_depth=None):
        """Rex-Omniを1視点・1フレームだけ実行する.

        Args:
            fallback_depth: Rex-Omni の深度が使えなかったときに使う
                sensor_msgs/Image。YOLO 応答の深度を渡します。
        """
        if not self._wait_for_rex_service(
            self.rex_detect_client,
            REX_OMNI_DETECTION_SERVICE,
        ):
            return None
        if not self._set_rex_model_state(start=True):
            return None

        try:
            request = RexObjectDetectionService.Request()
            request.confidence_th = 0.0
            request.iou_th = 0.0
            request.use_latest_image = True
            request.max_distance = MAX_GRASP_DISTANCE
            request.specific_id = REX_OMNI_PROMPT

            self.node.get_logger().info(
                'YOLOで物体を検出できなかったため、'
                f'Rex-Omniを prompt="{REX_OMNI_PROMPT}" で1回実行します。'
            )
            future = self.rex_detect_client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future)
            response = future.result()
            if response is None:
                self.node.get_logger().error(
                    'Rex-Omni検出サービスから応答がありません。'
                )
                return None

            try:
                return self._convert_rex_detections(
                    response.detections,
                    fallback_depth=fallback_depth,
                )
            except (TypeError, ValueError) as error:
                self.node.get_logger().error(
                    f'Rex-Omni検出結果を変換できませんでした: {error}'
                )
                return None
        finally:
            if not self._set_rex_model_state(start=False):
                self.node.get_logger().warning(
                    'Rex-Omniモデルを停止できませんでした。'
                )

    def _reset_for_room(self, blackboard: Blackboard) -> None:
        """部屋が変わったら、未検出回数と向き直し回数を数え直す."""
        room = (
            blackboard.current_room
            if 'current_room' in blackboard
            else None
        )
        if room != self.current_room:
            self.current_room = room
            self.empty_detection_count = 0
            self.rotation_retry_count = 0
            self.forward_retry_done = False
            self.attempted_names = set()
            self.viewpoint_index = 0
            self.applied_rotation = 0.0
            self.current_viewpoint = None
            self.returned_after_grasp = False

    def _rotate_and_retry(self) -> bool:
        """向きを変えてもう一度 YOLO を試せるなら回頭して True を返す.

        Rex-Omni はモデルのロードを伴って重いので、その前に体の向きを
        変えて安い YOLO をもう一度試します。回頭できたら失敗として戻り、
        ステートマシンの自己ループで Recog をやり直します。
        """
        if self.current_room not in ROTATION_RETRY_ROOMS:
            return False
        if self.rotation_retry_count >= MAX_ROTATION_RETRIES:
            return False

        self.rotation_retry_count += 1
        degrees = math.degrees(ROTATION_RETRY_YAW)
        self.node.get_logger().info(
            f'{self.current_room} で物体を検出できなかったため、'
            f'{degrees:.0f} 度向きを変えてもう一度 YOLO を試します '
            f'({self.rotation_retry_count}/{MAX_ROTATION_RETRIES})。'
        )
        if not self._rotate_base(ROTATION_RETRY_YAW):
            return False
        self.applied_rotation += ROTATION_RETRY_YAW
        return True

    def _rotate_base(self, yaw: float) -> bool:
        """台車をその場で回す。失敗しても止めずに False を返す."""
        try:
            self.hsrif.omni_base.go_rel(yaw=yaw, sync=True)
        # 回頭できなくても先の探索へ進めるよう、失敗は握りつぶします。
        except Exception as error:
            self.node.get_logger().warning(
                f'向きを変えられませんでした: {error}'
            )
            return False
        return True

    def _restore_observation_pose(self) -> None:
        """把持サイクルで戻された観測姿勢を、前回検出できた状態に戻す.

        Move2GraspPoint が既定の位置へナビゲーションし直すため、前回この
        部屋で移動した観測位置と回した角度は失われています。毎回そこから
        確認し直すのは無駄なので、検出する前に同じ状態まで戻します。
        """
        if not self.returned_after_grasp:
            return
        self.returned_after_grasp = False
        self._restore_viewpoint()
        self._restore_rotation()

    def _restore_viewpoint(self) -> None:
        """前回検出できた観測位置へ戻る (roomB など)."""
        if self.current_viewpoint is None:
            return
        if self.nav is None:
            return

        goal = self.current_viewpoint
        self.node.get_logger().info(
            f'{self.current_room} では前回 '
            f'x={goal.x:.2f}, y={goal.y:.2f}, yaw={goal.theta:.2f} で検出'
            'できたので、検出前にそこへ戻ります。'
        )
        if not self.nav.nav_goal(
            goal=goal,
            timeout=VIEWPOINT_NAVIGATION_TIMEOUT,
        ):
            self.node.get_logger().warning(
                f'観測位置へ戻れませんでした: {self.nav.nav_status.message}'
            )

    def _restore_rotation(self) -> None:
        """前回検出できた向きまで回し直す (roomA など)."""
        if self.applied_rotation == 0.0:
            return

        degrees = math.degrees(self.applied_rotation)
        self.node.get_logger().info(
            f'{self.current_room} では前回 {degrees:.0f} 度回して検出'
            'できたので、検出前に同じ角度まで回します。'
        )
        self._rotate_base(self.applied_rotation)

    def _transform_pose(
        self,
        stamped: PoseStamped,
        target_frame: str,
        timeout: float = TF_WAIT_TIMEOUT,
    ):
        """ノードを回しながら姿勢を target_frame へ変換する.

        このノードは rclpy.spin() で回り続けていません。tf2 側の timeout に
        任せると、待っている間に /tf を受け取れず、画像のタイムスタンプが
        最新 TF より数十 ms 新しいだけで必ず extrapolation エラーになります。
        自分で spin しながら、変換できるようになるまで待ちます。

        Args:
            stamped: 変換したい姿勢。
            target_frame: 変換先のフレーム。
            timeout: あきらめるまでの時間 [s]。

        Returns:
            変換後の geometry_msgs/Pose。

        Raises:
            TransformException: 時間内に変換できなかった場合。
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.tf_buffer.transform(
                    stamped,
                    target_frame,
                    timeout=Duration(seconds=0.0),
                ).pose
            except TransformException:
                if time.monotonic() >= deadline:
                    raise
            # /tf を受け取るために回します。これが無いと永久に届きません。
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _look_and_detect(self, head_tilt: float, move_head: bool = True):
        """ヘッドを指定の傾きに向けて YOLO を1回実行する.

        Args:
            head_tilt: head_tilt_joint の目標角 [rad]。0.0 で正面。
            move_head: False なら姿勢はそのままで撮り直しだけ行います。
                同じ画角で新しいタイムスタンプの画像が欲しいときに使います。

        Returns:
            検出結果。サービスから応答が無ければ None。
        """
        if move_head:
            try:
                # 腕は動かしません。move_to_go で作られた初期姿勢のまま、
                # 首だけを向けます。認識のたびに腕を出すと邪魔になります。
                self.hsrif.gripper.command(
                    RECOG_GRIPPER_ANGLE,
                    sync=True,
                )
                self.hsrif.whole_body.move_to_joint_positions(
                    {'head_pan_joint': 0.0, 'head_tilt_joint': head_tilt},
                    sync=True,
                )
            # 直前の動作で衝突状態のまま止まっていると
            # START_STATE_IN_COLLISION で例外が飛びます。ここで受けないと
            # ステートマシンごと落ちるので、失敗として扱います。
            except Exception as error:
                self.node.get_logger().error(
                    f'認識姿勢へ移動できませんでした: {error}'
                )
                return None
            time.sleep(2.0)

        # TF を受信して Buffer に溜めてからサービスを呼びます。
        for _ in range(20):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        detect_req = ObjectDetectionService.Request()
        detect_req.confidence_th = CONFIDENCE_THRESHOLD
        future = self.detect_client.call_async(detect_req)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()
        if response is None:
            self.node.get_logger().error(
                '物体検出サービスから応答がありません。'
            )
            return None
        return response.detections

    def _move_to_forward_pose(self) -> None:
        """机の上を見る姿勢を作る.

        立ち位置が指定されている部屋ではそこへ移動します。指定が無ければ
        移動せず、回した分だけ戻して体の向きを 0 度にします。
        """
        pose = FORWARD_RETRY_POSES.get(self.current_room)
        if pose is not None and self.nav is not None:
            if (
                pose['x'] is None
                or pose['y'] is None
                or pose['yaw'] is None
            ):
                self.node.get_logger().warning(
                    f'{self.current_room} の机を見る立ち位置が未設定です。'
                    'その場で首だけ正面に向けます。'
                )
            else:
                goal = Pose2D(
                    x=float(pose['x']),
                    y=float(pose['y']),
                    theta=float(pose['yaw']),
                )
                self.node.get_logger().info(
                    f'机の上を見るため '
                    f'x={goal.x:.2f}, y={goal.y:.2f}, '
                    f'yaw={goal.theta:.2f} へ移動します。'
                )
                if self.nav.nav_goal(
                    goal=goal,
                    timeout=VIEWPOINT_NAVIGATION_TIMEOUT,
                ):
                    # 移動で向きも決まったので、回した角度は忘れます。
                    self.applied_rotation = 0.0
                    # ここで見つかったなら、次のサイクルもここから始めます。
                    self.current_viewpoint = goal
                    return
                self.node.get_logger().warning(
                    f'机を見る立ち位置へ移動できませんでした: '
                    f'{self.nav.nav_status.message}'
                )

        # 机は既定の向きの正面にあるので、回した分を戻してから見ます。
        if self.applied_rotation != 0.0:
            degrees = math.degrees(self.applied_rotation)
            self.node.get_logger().info(
                f'机の上を見るため、回した {degrees:.0f} 度を戻して'
                '体の向きを 0 度にします。'
            )
            if self._rotate_base(-self.applied_rotation):
                # 以降はこの向きが基準になります。把持サイクルから
                # 戻ったときに古い角度を掛け直さないよう忘れます。
                self.applied_rotation = 0.0

    def _forward_yolo_retry(self):
        """首を正面に向けた YOLO を1度だけ試す.

        Rex-Omni でも見つからなかったときの最後の手段です。俯いた姿勢では
        画角から外れる、離れた場所の物体を拾うことを狙います。ここで駄目でも
        Rex-Omni は再実行しません。

        Returns:
            (検出結果, 添字)。試さなかった、または見つからなければ None。
        """
        if self.current_room not in FORWARD_RETRY_ROOMS:
            return None
        if self.forward_retry_done:
            return None

        self.forward_retry_done = True
        self._move_to_forward_pose()

        self.node.get_logger().info(
            f'{self.current_room} なので、首を正面に向けて '
            'YOLO をもう一度だけ試します。'
        )
        detections = self._look_and_detect(FORWARD_HEAD_TILT)
        if detections is None:
            return None

        names = [bbox.name for bbox in detections.bbox]
        self.node.get_logger().info(f'正面向きで検出した物体: {names}')

        index = self._select_target(detections)
        if index < 0:
            self.node.get_logger().info(
                '正面向きでも物体を検出できませんでした。'
            )
            return None
        return detections, index

    def _move_to_next_viewpoint(self) -> bool:
        """次の観測位置へ移動できたら True を返す.

        その部屋に残っている観測位置を上から順に1回ずつ使います。
        移動できたら失敗として戻り、ステートマシンの自己ループで
        Recog を最初からやり直します。
        """
        viewpoints = EXTRA_VIEWPOINTS.get(self.current_room, [])
        if self.viewpoint_index >= len(viewpoints):
            return False
        if self.nav is None:
            self.node.get_logger().warning(
                'NavModule が渡されていないため、観測位置へ移動できません。'
            )
            return False

        viewpoint = viewpoints[self.viewpoint_index]
        self.viewpoint_index += 1
        if (
            viewpoint['x'] is None
            or viewpoint['y'] is None
            or viewpoint['yaw'] is None
        ):
            self.node.get_logger().warning(
                f'{self.current_room} の {self.viewpoint_index} 番目の'
                '観測位置が未設定なので飛ばします。'
            )
            return False

        goal = Pose2D(
            x=float(viewpoint['x']),
            y=float(viewpoint['y']),
            theta=float(viewpoint['yaw']),
        )
        self.node.get_logger().info(
            f'{self.current_room} で検出できなかったため、'
            f'{self.viewpoint_index}/{len(viewpoints)} 番目の観測位置 '
            f'x={goal.x:.2f}, y={goal.y:.2f}, yaw={goal.theta:.2f} へ'
            '移動して検出し直します。'
        )
        if not self.nav.nav_goal(
            goal=goal,
            timeout=VIEWPOINT_NAVIGATION_TIMEOUT,
        ):
            self.node.get_logger().warning(
                f'観測位置へ移動できませんでした: '
                f'{self.nav.nav_status.message}'
            )
            return False
        # 次に把持サイクルから戻ってきたときは、ここから検出を始めます。
        self.current_viewpoint = goal
        return True

    def _is_excluded(self, name: str) -> bool:
        """把持対象から外す物体かどうかを返す.

        EXCLUDED_NAMES の指定と、この部屋で既に把持対象にしたものが対象です。
        """
        lowered = name.lower()
        return lowered in EXCLUDED_NAMES or lowered in self.attempted_names

    def _long_axis_center(self, detections, index, grasp, extra_shift):
        """把持位置を、物体の長辺方向の中点から細い側へ寄せた点にする.

        サービスが返す位置は有効深度点の重心です。スプーンのように片側だけ
        面積が大きい物体では、重心が広いほう (受け皿側) へ寄ってしまい、
        端を掴んで落とします。長辺方向の広がりの中点なら、形が偏っていても
        物体の真ん中になります。そこからさらに細い側へ寄せて、細い柄を
        挟めるようにします。

        Args:
            detections: 検出結果 (深度・マスク・camera_info を持つ)。
            index: 対象物体の添字。
            grasp: 把持点推定サービスの結果 (カメラ座標系)。
            extra_shift: 中点から細い側へ寄せる距離 [m]。

        Returns:
            カメラ座標系の位置 (x, y, z)。計算できなければ None。
        """
        depth_scale = DEPTH_SCALES.get(detections.depth.encoding)
        if depth_scale is None:
            return None

        try:
            depth = self.bridge.imgmsg_to_cv2(
                detections.depth,
                desired_encoding=detections.depth.encoding,
            ).astype(np.float64) * depth_scale
            mask = self.bridge.imgmsg_to_cv2(
                detections.segments[index],
                desired_encoding='mono8',
            )
        except Exception as error:  # cv_bridge の例外型は環境で異なる
            self.node.get_logger().warning(
                f'長辺中点の計算用に画像を変換できませんでした: {error}'
            )
            return None

        if depth.shape[:2] != mask.shape[:2]:
            return None

        k = detections.camera_info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        if fx == 0.0 or fy == 0.0:
            return None

        valid = (
            (mask != 0)
            & np.isfinite(depth)
            & (depth > 0.0)
            & (depth <= MAX_GRASP_DISTANCE)
        )
        ys, xs = np.nonzero(valid)
        if ys.size < 10:
            return None

        distances = depth[ys, xs]
        # サービスと同じ基準で、マスクの縁が拾った背景を落とします。
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median))) * 1.4826
        inliers = np.abs(distances - median) <= max(3.0 * mad, 0.005)
        if np.count_nonzero(inliers) < 10:
            return None
        distances = distances[inliers]
        xs = xs[inliers]
        ys = ys[inliers]

        points = np.stack(
            [
                (xs - cx) * distances / fx,
                (ys - cy) * distances / fy,
                distances,
            ],
            axis=1,
        )

        box_quaternion = np.array(
            [
                grasp.pose.orientation.x,
                grasp.pose.orientation.y,
                grasp.pose.orientation.z,
                grasp.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        norm = np.linalg.norm(box_quaternion)
        if norm < np.finfo(np.float64).eps:
            return None
        # 箱のローカル X が長辺、Y が短辺です (カメラ座標系のまま扱います)。
        rotation = tft.quaternion_matrix(box_quaternion / norm)[:3, :3]
        long_axis = rotation[:, 0]

        centroid = np.array(
            [
                grasp.pose.position.x,
                grasp.pose.position.y,
                grasp.pose.position.z,
            ],
            dtype=np.float64,
        )
        projection = (points - centroid) @ long_axis
        lower = float(np.percentile(projection, LONG_AXIS_TRIM_PERCENT))
        upper = float(
            np.percentile(projection, 100.0 - LONG_AXIS_TRIM_PERCENT)
        )
        midpoint = (lower + upper) * 0.5

        shift = midpoint
        if extra_shift != 0.0:
            short_projection = (points - centroid) @ rotation[:, 1]
            sign = self._thin_side_sign(
                projection,
                short_projection,
                midpoint,
            )
            if sign == 0.0:
                # 左右対称で細い側を決められない物体 (お椀など) では、
                # 手前の縁へ寄せます。カメラ座標系の +Z が奥行きなので、
                # 長辺軸の Z 成分と逆向きが手前になります。
                sign = -math.copysign(1.0, long_axis[2])
                self.node.get_logger().info(
                    '細い側を決められないので、手前側の縁へ寄せます。'
                )
            shift += extra_shift * sign

        self.node.get_logger().info(
            f'長辺方向: 重心 +0 mm / 中点 {midpoint * 1000.0:+.0f} mm / '
            f'掴む位置 {shift * 1000.0:+.0f} mm '
            f'(広がり {upper - lower:.3f} m)。'
        )
        if not lower <= shift <= upper:
            self.node.get_logger().warning(
                f'掴む位置 {shift * 1000.0:+.0f} mm が物体の範囲 '
                f'[{lower * 1000.0:+.0f}, {upper * 1000.0:+.0f}] mm を'
                'はみ出しています。寄せる量が大きすぎます。'
            )
        return centroid + shift * long_axis

    def _thin_side_sign(
        self,
        projection,
        short_projection,
        midpoint: float,
    ) -> float:
        """長辺の中点から見て、細いほうの端の符号を返す.

        PCA の長辺軸は符号が任意なので、中点の前後それぞれで短辺方向の
        広がりを測り、狭いほうを選びます。判定できないときは 0.0 です。
        """
        def spread(selected) -> float:
            if np.count_nonzero(selected) < 10:
                return float('inf')
            values = short_projection[selected]
            return float(
                np.percentile(values, 100.0 - LONG_AXIS_TRIM_PERCENT)
                - np.percentile(values, LONG_AXIS_TRIM_PERCENT)
            )

        positive = projection > midpoint
        width_positive = spread(positive)
        width_negative = spread(~positive)
        if not np.isfinite(width_positive) and not np.isfinite(
            width_negative
        ):
            self.node.get_logger().warning(
                '長辺の細い側を判定できませんでした。中点を掴みます。'
            )
            return 0.0
        # 差が小さいときは対称とみなして判定を諦めます。ノイズで
        # 向きが毎回反転するのを防ぐためです。
        narrow = min(width_positive, width_negative)
        wide = max(width_positive, width_negative)
        if not np.isfinite(wide) or wide - narrow < SYMMETRIC_WIDTH_MARGIN:
            self.node.get_logger().info(
                f'長辺の前後で幅の差が小さいです '
                f'({narrow:.3f} m 対 {wide:.3f} m)。'
            )
            return 0.0

        self.node.get_logger().info(
            f'細い側の判定: 幅 {narrow:.3f} m 対 {wide:.3f} m。'
        )
        return 1.0 if width_positive < width_negative else -1.0

    def _select_target(self, detections) -> int:
        """対象名に一致する検出のうち、ロボットに最も近い添字を返す.

        EXCLUDED_NAMES の物体と、この部屋で既に把持対象にした物体は、
        検出されていても候補から外します。後者があるおかげで、Grasp が
        失敗して Recog に戻っても同じ物体を掴み直さずに済みます。
        """
        excluded_names = [
            bbox.name
            for bbox in detections.bbox
            if bbox.name.lower() in EXCLUDED_NAMES
        ]
        if excluded_names:
            self.node.get_logger().info(
                f'除外指定の物体なので把持対象から外します: {excluded_names}'
            )

        attempted = [
            bbox.name
            for bbox in detections.bbox
            if bbox.name.lower() in self.attempted_names
        ]
        if attempted:
            self.node.get_logger().info(
                f'既に把持対象にしたので候補から外します: {attempted}'
            )

        self.tf_error_in_selection = False
        candidate_indices = [
            index
            for index, bbox in enumerate(detections.bbox)
            if not self._is_excluded(bbox.name)
            and (not TARGET_NAME or bbox.name == TARGET_NAME)
        ]
        if not candidate_indices:
            return -1

        depth_scale = DEPTH_SCALES.get(detections.depth.encoding)
        if depth_scale is None:
            self.node.get_logger().error(
                '最も近い物体の選択では未対応の'
                '深度エンコーディングです: '
                f'{detections.depth.encoding}'
            )
            return -1

        try:
            depth = self.bridge.imgmsg_to_cv2(
                detections.depth,
                desired_encoding=detections.depth.encoding,
            ).astype(np.float64) * depth_scale
        except Exception as error:  # cv_bridge の例外型は環境で異なる
            self.node.get_logger().error(
                f'距離選択用の深度画像を変換できませんでした: {error}'
            )
            return -1

        k = detections.camera_info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        if fx == 0.0 or fy == 0.0:
            self.node.get_logger().error(
                '距離選択に必要な camera_info が空です。'
            )
            return -1

        candidates = []
        for index in candidate_indices:
            bbox = detections.bbox[index]
            if index >= len(detections.segments):
                self.node.get_logger().warning(
                    f'{bbox.name} に対応するマスクがありません。'
                )
                continue

            try:
                mask = self.bridge.imgmsg_to_cv2(
                    detections.segments[index],
                    desired_encoding='mono8',
                )
            except Exception as error:  # cv_bridge の例外型は環境で異なる
                self.node.get_logger().warning(
                    f'{bbox.name} のマスクを変換できませんでした: {error}'
                )
                continue

            if depth.shape[:2] != mask.shape[:2]:
                self.node.get_logger().warning(
                    f'{bbox.name} の深度画像 {depth.shape[:2]} と'
                    f'マスク {mask.shape[:2]} の解像度が異なります。'
                )
                continue

            valid = (
                (mask != 0)
                & np.isfinite(depth)
                & (depth > 0.0)
                & (depth <= MAX_GRASP_DISTANCE)
            )
            ys, xs = np.nonzero(valid)
            if ys.size == 0:
                self.node.get_logger().warning(
                    f'{bbox.name} の有効な深度を取得できませんでした。'
                )
                continue

            distances = depth[ys, xs]
            median = float(np.median(distances))
            mad = float(np.median(np.abs(distances - median))) * 1.4826
            inliers = np.abs(distances - median) <= max(3.0 * mad, 0.005)
            distances = distances[inliers]
            xs = xs[inliers]
            ys = ys[inliers]

            camera_x = np.mean((xs - cx) * distances / fx)
            camera_y = np.mean((ys - cy) * distances / fy)
            camera_z = np.mean(distances)

            stamped = PoseStamped()
            stamped.header = detections.camera_info.header
            stamped.pose.position.x = float(camera_x)
            stamped.pose.position.y = float(camera_y)
            stamped.pose.position.z = float(camera_z)
            stamped.pose.orientation.w = 1.0
            try:
                base_pose = self._transform_pose(stamped, 'base_link')
            except TransformException as error:
                self.tf_error_in_selection = True
                self.node.get_logger().warning(
                    f'{bbox.name} の距離を base_link 基準に変換'
                    f'できませんでした: {error}'
                )
                continue

            position = base_pose.position
            distance = math.sqrt(
                position.x ** 2 + position.y ** 2 + position.z ** 2
            )
            if distance < MIN_GRASP_DISTANCE:
                self.node.get_logger().info(
                    f'{bbox.name} はロボットから {distance:.3f} m と'
                    f'近すぎるので無視します '
                    f'(下限 {MIN_GRASP_DISTANCE:.3f} m)。'
                )
                continue

            self.node.get_logger().info(
                f'把持候補 {bbox.name}: ロボットから {distance:.3f} m'
            )
            candidates.append((distance, index))

        if not candidates:
            return -1

        distance, index = min(candidates, key=lambda item: item[0])
        self.node.get_logger().info(
            f'最も近い物体 {detections.bbox[index].name} '
            f'({distance:.3f} m) を選択しました。'
        )
        return index

    def execute(self, blackboard: Blackboard) -> str:
        """対象物体を認識し、把持前姿勢を Blackboard に保存する."""
        self.node.get_logger().info('Executing state Recog')
        self._reset_for_room(blackboard)

        if not self._wait_for_service(
            self.detect_client, '/yolov8_detection/service'
        ):
            return 'failed'
        if not self._wait_for_service(
            self.grasp_client, '/grasp_point_detection/service'
        ):
            return 'failed'

        self._restore_observation_pose()

        detections = self._look_and_detect(HEAD_TILT)
        if detections is None:
            return 'failed'

        names = [bbox.name for bbox in detections.bbox]
        self.node.get_logger().info(f'検出した物体: {names}')

        index = self._select_target(detections)

        # 候補はあったのに TF 変換の失敗だけで選べなかった場合は、
        # 画像を撮り直してタイムスタンプを更新し、選び直します。
        # 姿勢はそのままなので、撮り直しは 1 秒程度で終わります。
        for retry in range(1, MAX_TF_RETRY_DETECTIONS + 1):
            if index >= 0 or not self.tf_error_in_selection:
                break
            self.node.get_logger().warning(
                'TF 変換に失敗して候補が無くなりました。'
                f'撮り直して選び直します ({retry}/'
                f'{MAX_TF_RETRY_DETECTIONS})。'
            )
            detections = self._look_and_detect(HEAD_TILT, move_head=False)
            if detections is None:
                return 'failed'
            names = [bbox.name for bbox in detections.bbox]
            self.node.get_logger().info(f'撮り直して検出した物体: {names}')
            index = self._select_target(detections)

        if index < 0:
            # Rex-Omni へ移る前に、向きを変えて YOLO をもう一度試します。
            if self._rotate_and_retry():
                return 'failed'

            # 次に、別の位置へ移動して検出し直します。
            if self._move_to_next_viewpoint():
                return 'failed'

            self.empty_detection_count = min(
                self.empty_detection_count + 1,
                MAX_EMPTY_DETECTIONS,
            )
            target = TARGET_NAME or '物体'
            self.node.get_logger().warning(
                f'{target} が見つかりません '
                f'({self.empty_detection_count}/{MAX_EMPTY_DETECTIONS})。'
            )
            if self.empty_detection_count >= MAX_EMPTY_DETECTIONS:
                rex_detections = None
                rex_index = -1
                if not USE_REX_OMNI:
                    self.node.get_logger().info(
                        'Rex-Omni は無効なので実行しません。'
                    )
                else:
                    self.node.get_logger().info(
                        f'{MAX_EMPTY_DETECTIONS}回連続で物体を検出'
                        'できなかったため、Rex-Omniで最終確認します。'
                    )
                    # Rex-Omni は最終確認なので、駄目でも再実行はしません。
                    rex_detections = self._detect_with_rex_omni(
                        fallback_depth=detections.depth,
                    )
                    if rex_detections is None:
                        self.node.get_logger().warning(
                            'Rex-Omniを実行できませんでした。'
                        )
                    elif not rex_detections.bbox:
                        self.node.get_logger().info(
                            'Rex-Omniでも物体を検出しませんでした。'
                        )
                    else:
                        rex_names = [
                            bbox.name for bbox in rex_detections.bbox
                        ]
                        self.node.get_logger().info(
                            f'Rex-Omniが検出した物体: {rex_names}'
                        )
                        rex_index = self._select_target(rex_detections)
                        if rex_index < 0:
                            self.node.get_logger().warning(
                                'Rex-Omniは物体を検出しましたが、'
                                '把持可能な深度またはmaskを'
                                '取得できませんでした。'
                            )

                if rex_index < 0:
                    # 最後に、首を正面に向けた YOLO を1度だけ試します。
                    # これで駄目なら none を返します。
                    forward = self._forward_yolo_retry()
                    if forward is None:
                        self.empty_detection_count = 0
                        self.node.get_logger().info(
                            '物体を検出できなかったため、none を返します。'
                        )
                        return 'none'
                    detections, index = forward
                    self.node.get_logger().info(
                        '正面向きの検出結果を把持処理へ渡します。'
                    )
                else:
                    detections = rex_detections
                    index = rex_index
                    self.node.get_logger().info(
                        'Rex-Omniの検出結果を把持処理へ渡します。'
                    )
            else:
                return 'failed'

        # 物体を検出できたので、未検出回数をリセットします。
        # 回頭回数は部屋ごとに数えるのでここでは戻しません。戻すと、
        # _restore_rotation() で 60 度に戻したところからさらに 60 度
        # 回ってしまいます。1 つの部屋で回すのは 1 回だけです。
        self.empty_detection_count = 0
        self.forward_retry_done = False
        if index >= len(detections.segments):
            self.node.get_logger().error(
                '検出物体に対応するマスクがありません。'
            )
            return 'failed'

        grasp_req = GraspPointService.Request()
        grasp_req.depth = detections.depth
        grasp_req.mask = detections.segments[index]
        grasp_req.camera_info = detections.camera_info
        grasp_req.max_distance = MAX_GRASP_DISTANCE

        future = self.grasp_client.call_async(grasp_req)
        rclpy.spin_until_future_complete(self.node, future)
        grasp_response = future.result()

        if grasp_response is None:
            self.node.get_logger().error(
                '把持点推定サービスから応答がありません。'
            )
            return 'failed'
        if not grasp_response.success:
            self.node.get_logger().error(
                f'把持点を推定できませんでした: {grasp_response.message}'
            )
            return 'failed'

        # 推定結果を、腕を動かす base_link 基準へ変換します。
        stamped = PoseStamped()
        stamped.header = detections.camera_info.header
        stamped.pose = grasp_response.grasp.pose

        # 指定された物体は、重心ではなく長辺方向の中点を基準に掴みます。
        target_name = detections.bbox[index].name
        extra_shift = GRASP_LONG_AXIS_OFFSETS.get(target_name.lower())
        if extra_shift is not None:
            center = self._long_axis_center(
                detections,
                index,
                grasp_response.grasp,
                extra_shift,
            )
            if center is None:
                self.node.get_logger().warning(
                    f'{target_name} の長辺方向の中点を計算できなかったため、'
                    '重心のまま掴みます。'
                )
            else:
                stamped.pose.position.x = float(center[0])
                stamped.pose.position.y = float(center[1])
                stamped.pose.position.z = float(center[2])
        try:
            object_pose = self._transform_pose(stamped, 'base_link')
        except TransformException as error:
            self.node.get_logger().error(
                f'把持姿勢の TF 変換に失敗しました: {error}'
            )
            return 'failed'

        # 高さで上／横把持を選び、短辺側の面をグリッパで挟みます。
        height = grasp_response.grasp.size.z
        long_edge = grasp_response.grasp.size.x
        short_edge = grasp_response.grasp.size.y
        box_yaw = tft.euler_from_quaternion([
            object_pose.orientation.x,
            object_pose.orientation.y,
            object_pose.orientation.z,
            object_pose.orientation.w,
        ])[2]
        self.node.get_logger().info(
            '推定した物体寸法: '
            f'長辺={long_edge:.3f} m, 短辺={short_edge:.3f} m, '
            f'高さ={height:.3f} m / '
            f'長辺方向(base_link)={math.degrees(box_yaw):.1f} deg / '
            f'長短比={long_edge / short_edge if short_edge > 0.0 else float("inf"):.2f} / '
            f'把持={"横" if height > TALL_THRESHOLD else "上"}'
        )

        if object_pose.position.z < MIN_GRASP_HEIGHT:
            self.node.get_logger().info(
                f'把持位置が床に近すぎます '
                f'(z={object_pose.position.z:.3f} m)。'
                f'{MIN_GRASP_HEIGHT:.3f} m まで持ち上げます。'
            )
            object_pose.position.z = MIN_GRASP_HEIGHT


        if height > TALL_THRESHOLD:
            short_axis = _horizontal_short_axis(object_pose.orientation)
            object_xy = np.array(
                [object_pose.position.x, object_pose.position.y],
                dtype=np.float64,
            )
            # PCA の短辺軸には180度の曖昧さがあるため、ロボットから物体へ
            # 向かう側を選び、短辺方向から物体へ接近できるようにします。
            if np.dot(short_axis[:2], object_xy) < 0.0:
                short_axis *= -1.0
            orientation, approach_axis = _side_grasp_orientation(short_axis[:2])
            # 手のひらを把持点の手前どれだけで止めるかは保ったまま、
            # ロボットに近づきすぎない範囲でプリグラスプを下げます。
            clearance = (
                SIDE_GRASP_PREGRASP_DISTANCE - SIDE_GRASP_APPROACH_DISTANCE
            )
            object_distance = math.hypot(
                object_pose.position.x,
                object_pose.position.y,
            )
            standoff = min(
                SIDE_GRASP_PREGRASP_DISTANCE,
                max(0.0, object_distance - MIN_PREGRASP_RADIUS),
            )
            if standoff < SIDE_GRASP_PREGRASP_DISTANCE:
                self.node.get_logger().warning(
                    f'物体がロボットから {object_distance:.3f} m と近いため、'
                    f'把持前姿勢を下げる量を '
                    f'{SIDE_GRASP_PREGRASP_DISTANCE:.3f} m から '
                    f'{standoff:.3f} m に減らします。'
                )
            object_pose.position.x -= standoff * approach_axis[0]
            object_pose.position.y -= standoff * approach_axis[1]
            approach = standoff - clearance
            wrist_roll = None
            approach_yaw = math.atan2(approach_axis[1], approach_axis[0])
            self.node.get_logger().info(
                '背が高い物体なので、短辺方向から接近する横把持にします。'
                f'接近方向={math.degrees(approach_yaw):.1f} deg '
                f'(物体の短辺方向、ロボットから物体へ向かう向き)、'
                'グリッパは長辺方向に閉じて短辺側の面を挟みます。'
                f'把持点の {standoff * 100.0:.0f} cm 手前 '
                f'(base_link から {object_distance - standoff:.2f} m) '
                f'から {approach * 100.0:.0f} cm 伸ばして'
                f'把持点の {clearance * 100.0:.0f} cm 手前で止めます。'
            )
        else:
            long_axis = _horizontal_long_axis(object_pose.orientation)
            # 手首を回す高さを物体ごとに上げられるようにします。上げた分は
            # 降りる距離にも足すので、掴む位置そのものは変わりません。
            extra_lift = WRIST_ROTATION_EXTRA_LIFT.get(
                target_name.lower(),
                0.0,
            )
            object_pose.position.z += 0.1 + extra_lift
            approach = 0.05 + extra_lift
            if extra_lift != 0.0:
                self.node.get_logger().info(
                    f'{target_name} は手首を回す高さを '
                    f'{extra_lift * 1000.0:.0f} mm 上げます '
                    f'(把持点の {(0.1 + extra_lift) * 100.0:.0f} cm 上で回し、'
                    f'{approach * 100.0:.0f} cm 降ります)。'
                )

            # 手のひらを真下に向ける姿勢だけを IK に渡します。
            # yaw を姿勢に含めると arm_roll など別の関節へ逃げることがあるので、
            # 長辺に合わせる回転は wrist_roll_joint へ明示的に指令します。
            # ここで渡す wrist_roll は「グリッパが閉じる方向に直交する向き」、
            # つまり指の間を通り抜ける向きです。長辺に90度足した向きを渡すと、
            # グリッパの閉じ方向が長辺と一致し、短辺側の面を挟めます。実際の
            # 関節角への変換と符号は GraspState._rotate_wrist_to_long_edge が持ちます。
            qx, qy, qz, qw = tft.quaternion_from_euler(math.pi, 0.0, 0.0)
            orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            wrist_roll = math.atan2(long_axis[1], long_axis[0]) + math.pi / 2.0
            self.node.get_logger().info(
                '平たい物体なので、長辺方向にグリッパを閉じる上把持にします。'
                f'指の間を通す向き={math.degrees(wrist_roll):.1f} deg '
                'に合わせ、短辺側の面を挟みます。'
            )

        object_pose.orientation = orientation

        blackboard.grasp_pose = object_pose
        blackboard.grasp_approach = approach
        # 上把持のときだけ、base_link 基準での目標ハンド yaw [rad]。
        # 横把持では姿勢そのものに向きが入っているので None です。
        blackboard.grasp_wrist_roll = wrist_roll
        blackboard.target_name = detections.bbox[index].name
        # 把持の成否によらず、この部屋ではもうこの物体を狙いません。
        # Grasp が失敗して Recog へ戻っても、次は別の物体を選びます。
        self.attempted_names.add(blackboard.target_name.lower())
        # このあと Move2GraspPoint が台車の向きを戻すので、次に来たときは
        # 覚えている角度まで回し直します。
        self.returned_after_grasp = True
        self.node.get_logger().info(
            f'{blackboard.target_name} の把持姿勢を Blackboard に保存しました。'
        )
        return 'succeeded'
