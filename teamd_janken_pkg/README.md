# carrobo_tidyup_pkg

ステートは次の順に遷移します

```text
Move2GraspPoint -> Recog -> Grasp -> Move2PlacePoint -> Place
```


## ビルドと実行

```bash
cd ~/hma2_ws
colcon_build_release_single teamd_janken_pkg
source install/setup.bash
ros2 launch teamd_janken_pkg tidyup.launch.py
```

`tidyup.launch.py` は YOLOv8 検出サービス（yolov8_detection）、把持点推定サービス（grasp_detection_point）、YASMIN
Viewer と本ステートマシンを起動します。
bring upとnavigation、Issac Simは立ち上げておいてください。
