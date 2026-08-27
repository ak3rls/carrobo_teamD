 #client_node
import rclpy
from rclpy.node import Node
from teamd_janken_interfaces.srv import hand_recog, jyanken_robot_node

class JankenClient(Node):
    def __init__(self):
        super().__init__('janken_client')
        self.cli_hand = self.create_client(hand_recog, 'hand_recog_jg')
        while not self.cli_hand.wait_for_service(timeout_sec = 1.0):
            self.get_logger().info('service_hand not available, wait...')
        self.cli_robot = self.create_client(jyanken_robot_node, 'robot_move')
        while not self.cli_robot.wait_for_service(timeout_sec = 1.0):
            self.get_logger().info('service_robot not available, wait...')
        
        self.req_hand = hand_recog.Request()
        self.req_robot = jyanken_robot_node.Request()

    def send_hand_request(self, kara:int):
        self.req_hand.kara = int(kara)
        future_hand = self.cli_hand.call_async(self.req_hand)
        rclpy.spin_until_future_complete(self,future_hand)
        return future_hand.result()
        
    def send_robot_request(self, sentaku:str):
        self.req_robot.sentaku = str(sentaku)
        future_robot = self.cli_robot.call_async(self.req_robot)
        rclpy.spin_until_future_complete (self, future_robot)
        return future_robot.result ()

def mein(args=None):
    rclpy.init(args=args)
    node =  JankenClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node ()
        if rclpy.ok():
             rclpy.shutdown ()
