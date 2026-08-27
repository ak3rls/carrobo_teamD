"""物体リスト prompt の YOLOE 検出結果から把持姿勢を求めるステート."""

from __future__ import annotations

from carrobo_manipulation_pkg.hsrif import HSRInterfaces
from rclpy.node import Node
from tf2_ros import Buffer

from teamd_tidyup_pkg.objectlist import load_objectlist

from .recog import REX_OMNI_PROMPT
from .recog import RecogState


YOLOE_DETECTION_SERVICE = '/yoloe_detection/service'


class Recog2State(RecogState):
    """objectlist.yaml を prompt に使う YOLOE 用の認識ステート."""

    def __init__(
        self,
        node: Node,
        hsrif: HSRInterfaces,
        tf_buffer: Buffer,
        nav=None,
    ):
        """物体リストを読み込み、YOLOEサービスへ接続する."""
        self.prompt_names, self.category_by_name = load_objectlist()
        self._prompt_name_keys = {
            name.casefold() for name in self.prompt_names
        }
        super().__init__(
            node,
            hsrif,
            tf_buffer,
            nav=nav,
            detect_service_name=YOLOE_DETECTION_SERVICE,
        )
        self.node.get_logger().info(
            'YOLOE認識で objectlist.yaml の prompt を使用します: '
            f'{len(self.prompt_names)}種類'
        )

    def _is_excluded(self, name: str) -> bool:
        """物体リストにないラベルを把持候補から外す."""
        # Rex-Omniの最終確認だけは汎用ラベル ``object`` を返すため、
        # objectlist外でも親クラスの除外規則に任せます。
        if name.casefold() == REX_OMNI_PROMPT.casefold():
            return super()._is_excluded(name)
        if name.casefold() not in self._prompt_name_keys:
            return True
        return super()._is_excluded(name)


__all__ = ['Recog2State', 'YOLOE_DETECTION_SERVICE']
