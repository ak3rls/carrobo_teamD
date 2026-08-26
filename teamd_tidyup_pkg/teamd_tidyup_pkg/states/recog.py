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

CONFIDENCE_THRESHOLD = 0.25
MAX_GRASP_DISTANCE = 2.0
DEPTH_SCALES = {
    '16UC1': 0.001,
    '32FC1': 1.0,
}
TALL_THRESHOLD = 0.15
HEAD_TILT = math.radians(-50.0)
MAX_EMPTY_DETECTIONS = 3
REX_OMNI_PROMPT = 'object'
REX_OMNI_DETECTION_SERVICE = '/rex_omni_sam2/object_detection'
REX_OMNI_START_SERVICE = '/rex_omni/start'
REX_OMNI_STOP_SERVICE = '/rex_omni/stop'
REX_OMNI_SERVICE_TIMEOUT = 10.0


def _box_aligned_grasp_orientation(
    box_orientation: Quaternion,
    side_grasp: bool,
    object_xy,
):
    """
    Align the gripper closing axis with the object's horizontal short edge.

    The grasp-point service defines the box-local X axis as the long edge and
    Y as the short edge.  The HSR hand closes along its local Y axis and
    approaches along local +Z.

    For a side grasp, the PCA long axis has an arbitrary sign.  Select the
    sign pointing from the robot toward the object so the pre-grasp offset is
    placed on the robot-facing end of the object.
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

    box_rotation = tft.quaternion_matrix(box_quaternion)[:3, :3]
    long_axis = box_rotation[:, 0].copy()
    long_axis[2] = 0.0
    long_axis_norm = np.linalg.norm(long_axis)
    if long_axis_norm < np.finfo(np.float64).eps:
        raise ValueError('物体の水平長辺方向を計算できません。')
    long_axis /= long_axis_norm

    if side_grasp and np.dot(long_axis[:2], object_xy) < 0.0:
        half_turn = tft.quaternion_from_euler(0.0, 0.0, math.pi)
        box_quaternion = tft.quaternion_multiply(
            box_quaternion,
            half_turn,
        )
        long_axis *= -1.0

    if side_grasp:
        grasp_offset = tft.quaternion_from_euler(
            math.pi,
            -math.pi / 2.0,
            0.0,
        )
    else:
        grasp_offset = tft.quaternion_from_euler(math.pi, 0.0, 0.0)

    grasp_quaternion = tft.quaternion_multiply(
        box_quaternion,
        grasp_offset,
    )
    grasp_quaternion /= np.linalg.norm(grasp_quaternion)
    orientation = Quaternion(
        x=float(grasp_quaternion[0]),
        y=float(grasp_quaternion[1]),
        z=float(grasp_quaternion[2]),
        w=float(grasp_quaternion[3]),
    )
    return orientation, long_axis


class RecogState(State):
    """物体検出と把持点推定を行うステート."""

    def __init__(
        self,
        node: Node,
        hsrif: HSRInterfaces,
        tf_buffer: Buffer,
    ):
        """サービスクライアントを生成する."""
        super().__init__(outcomes=['succeeded', 'failed', 'none'])
        self.node = node
        self.hsrif = hsrif
        self.tf_buffer = tf_buffer
        self.empty_detection_count = 0
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

    def _convert_rex_detections(self, detections) -> ObjectDetection:
        """Rex-Omni+SAM2の圧縮画像結果を既存YOLO形式へ変換する."""
        converted = ObjectDetection()
        converted.header = detections.header
        converted.is_detected = detections.is_detected
        converted.camera_info = detections.camera_info

        if not detections.bbox:
            return converted
        if not detections.depth.data:
            raise ValueError('Rex-Omni検出結果に深度画像がありません。')

        if 'compresseddepth' in detections.depth.format.lower():
            depth = self.bridge.compressed_imgmsg_to_depth(detections.depth)
        else:
            depth = cv2.imdecode(
                np.frombuffer(detections.depth.data, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
        if depth is None:
            raise ValueError('Rex-Omniの圧縮深度画像を変換できません。')
        depth = np.asarray(depth)
        if depth.dtype == np.uint16:
            depth_encoding = '16UC1'
        elif depth.dtype == np.float32:
            depth_encoding = '32FC1'
        else:
            raise ValueError(
                'Rex-Omniの深度画像型に対応していません: '
                f'{depth.dtype}'
            )

        converted.depth = self.bridge.cv2_to_imgmsg(
            depth,
            encoding=depth_encoding,
        )
        converted.depth.header = detections.depth.header

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

    def _detect_with_rex_omni(self):
        """Rex-Omniを1視点・1フレームだけ実行する."""
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
                return self._convert_rex_detections(response.detections)
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

    def _select_target(self, detections) -> int:
        """対象名に一致する検出のうち、ロボットに最も近い添字を返す."""
        candidate_indices = [
            index
            for index, bbox in enumerate(detections.bbox)
            if not TARGET_NAME or bbox.name == TARGET_NAME
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
                base_pose = self.tf_buffer.transform(
                    stamped,
                    'base_link',
                    timeout=Duration(seconds=2.0),
                ).pose
            except TransformException as error:
                self.node.get_logger().warning(
                    f'{bbox.name} の距離を base_link 基準に変換'
                    f'できませんでした: {error}'
                )
                continue

            position = base_pose.position
            distance = math.sqrt(
                position.x ** 2 + position.y ** 2 + position.z ** 2
            )
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

        if not self._wait_for_service(
            self.detect_client, '/yolov8_detection/service'
        ):
            return 'failed'
        if not self._wait_for_service(
            self.grasp_client, '/grasp_point_detection/service'
        ):
            return 'failed'

        # move_to_go はカメラを腕で隠すため、認識時は neutral 姿勢です。
        self.hsrif.whole_body.move_to_neutral(sync=True)
        self.hsrif.whole_body.move_to_joint_positions(
            {'head_pan_joint': 0.0, 'head_tilt_joint': HEAD_TILT},
            sync=True,
        )
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
            return 'failed'

        detections = response.detections
        names = [bbox.name for bbox in detections.bbox]
        self.node.get_logger().info(f'検出した物体: {names}')

        index = self._select_target(detections)
        if index < 0:
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
                self.node.get_logger().info(
                    f'{MAX_EMPTY_DETECTIONS}回連続で物体を検出できなかったため、'
                    'Rex-Omniで最終確認します。'
                )
                rex_detections = self._detect_with_rex_omni()
                if rex_detections is None:
                    return 'failed'

                rex_names = [bbox.name for bbox in rex_detections.bbox]
                self.node.get_logger().info(
                    f'Rex-Omniが検出した物体: {rex_names}'
                )
                if not rex_detections.bbox:
                    self.empty_detection_count = 0
                    self.node.get_logger().info(
                        'Rex-Omniでも物体を検出しなかったため、'
                        'none を返します。'
                    )
                    return 'none'

                rex_index = self._select_target(rex_detections)
                if rex_index < 0:
                    self.node.get_logger().warning(
                        'Rex-Omniは物体を検出しましたが、'
                        '把持可能な深度またはmaskを取得できませんでした。'
                    )
                    return 'failed'

                detections = rex_detections
                index = rex_index
                self.node.get_logger().info(
                    'Rex-Omniの検出結果を把持処理へ渡します。'
                )
            else:
                return 'failed'

        # 物体を検出できた場合、連続未検出回数をリセットします。
        self.empty_detection_count = 0
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
        try:
            object_pose = self.tf_buffer.transform(
                stamped,
                'base_link',
                timeout=Duration(seconds=2.0),
            ).pose
        except TransformException as error:
            self.node.get_logger().error(
                f'把持姿勢の TF 変換に失敗しました: {error}'
            )
            return 'failed'

        # 高さで上／横把持を選び、水平の短辺をグリッパで挟みます。
        height = grasp_response.grasp.size.z
        long_edge = grasp_response.grasp.size.x
        short_edge = grasp_response.grasp.size.y
        self.node.get_logger().info(
            '推定した物体寸法: '
            f'長辺={long_edge:.3f} m, 短辺={short_edge:.3f} m, '
            f'高さ={height:.3f} m'
        )

        if height > TALL_THRESHOLD:
            orientation, approach_axis = _box_aligned_grasp_orientation(
                object_pose.orientation,
                side_grasp=True,
                object_xy=(object_pose.position.x, object_pose.position.y),
            )
            object_pose.position.x -= 0.1 * approach_axis[0]
            object_pose.position.y -= 0.1 * approach_axis[1]
            approach = 0.1
            approach_yaw = math.atan2(approach_axis[1], approach_axis[0])
            self.node.get_logger().info(
                '背が高い物体なので、短辺を挟む横把持にします。'
                f'長辺方向からの接近角={math.degrees(approach_yaw):.1f} deg'
            )
        else:
            orientation, _ = _box_aligned_grasp_orientation(
                object_pose.orientation,
                side_grasp=False,
                object_xy=(object_pose.position.x, object_pose.position.y),
            )
            object_pose.position.z += 0.1
            approach = 0.05
            self.node.get_logger().info(
                '平たい物体なので、短辺を挟む上把持にします。'
            )

        object_pose.orientation = orientation

        blackboard.grasp_pose = object_pose
        blackboard.grasp_approach = approach
        blackboard.target_name = detections.bbox[index].name
        self.node.get_logger().info(
            f'{blackboard.target_name} の把持姿勢を Blackboard に保存しました。'
        )
        return 'succeeded'
