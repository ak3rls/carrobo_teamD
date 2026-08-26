import rclpy
from rclpy.node import Node
import cv2
import mediapipe as mp


class Hand_Recog(Node):
    def __init__(self):
        super().__init__()

        #わかりやすく見せるために点や線を表示させるサポート導入
        self.mp_drawing = mp.solutions.drawing_utils
        #手の骨格を判定させる
        self.mp_hands = mp.solutions.mp_hands

        #表示させる手のせってい
        self.hands = self.mp_hands.Hands(
            max_num_hands = 1,  #認識する手の数
            min_detection_confidence = 0.7,  #初期の手を認識する際の最低信頼度
            min_tracking_confidence = 0.7,   #手が認識できている際の最低信頼度
        )

def main():
    cap = cv2.VideoCapture(0)

    print("カメラを起動")
    print("qを押してカメラを終了")

    while cap.isOpened():

        success, frame = cap.read()

        if not succeeded



