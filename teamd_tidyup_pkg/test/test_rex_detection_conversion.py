"""Tests for converting Rex-Omni detections into the existing YOLO format."""

import numpy as np
from hma_object_detection2_interfaces.msg import BBox as RexBBox
from hma_object_detection2_interfaces.msg import (
    ObjectDetection as RexDetection,
)
from hma_rclpy_extension.cv_bridge import CvBridgeUtils

from teamd_tidyup_pkg.states.recog import RecogState


def _state_with_bridge():
    state = RecogState.__new__(RecogState)
    state.bridge = CvBridgeUtils()
    return state


def test_empty_rex_detection_does_not_require_depth():
    """Return an empty YOLO-compatible detection without decoding depth."""
    rex = RexDetection()
    rex.is_detected = False

    converted = _state_with_bridge()._convert_rex_detections(rex)

    assert not converted.is_detected
    assert not converted.bbox
    assert not converted.segments


def test_rex_detection_converts_depth_mask_and_bbox():
    """Preserve a Rex bbox and convert its depth and mask to raw images."""
    state = _state_with_bridge()
    rex = RexDetection()
    rex.is_detected = True

    depth = np.full((6, 8), 1200, dtype=np.uint16)
    rex.depth = state.bridge.cv2_to_compressed_imgmsg(
        depth,
        dst_format='png',
    )

    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[1:, 1:3] = 255
    rex.segments.append(
        state.bridge.cv2_to_compressed_imgmsg(mask, dst_format='png')
    )

    bbox = RexBBox()
    bbox.id = 7
    bbox.name = 'object'
    bbox.score = 1.0
    bbox.x = 4
    bbox.y = 3
    bbox.w = 2
    bbox.h = 2
    rex.bbox.append(bbox)

    converted = state._convert_rex_detections(rex)

    assert converted.is_detected
    assert converted.depth.encoding == '16UC1'
    assert converted.depth.height == 6
    assert converted.depth.width == 8
    assert converted.bbox[0].id == 7
    assert converted.bbox[0].name == 'object'
    assert converted.segments[0].encoding == 'mono8'
    assert converted.segments[0].height == 6
    assert converted.segments[0].width == 8
