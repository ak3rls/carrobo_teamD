#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS 2サービス呼び出しの共通処理."""

import rclpy
from rclpy.node import Node


def call_service(node: Node, client, request, service_name: str, service_wait_timeout: float = 5.0, response_timeout: float = 30.0,):
    """サービスを呼び出し、応答またはNoneを返す."""
    if not client.wait_for_service(
        timeout_sec=service_wait_timeout
    ):
        node.get_logger().error(
            f'サービスが見つかりません: {service_name}'
        )
        return None

    future = client.call_async(request)

    rclpy.spin_until_future_complete(
        node,
        future,
        timeout_sec=response_timeout,
    )

    if not future.done():
        future.cancel()
        node.get_logger().error(
            f'サービスがタイムアウトしました: {service_name}'
        )
        return None

    try:
        return future.result()
    except Exception as error:
        node.get_logger().error(
            f'サービス呼び出しに失敗しました: '
            f'{service_name}: {error}'
        )
        return None