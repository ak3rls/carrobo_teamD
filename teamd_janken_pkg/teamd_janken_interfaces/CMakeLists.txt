cmake_minimum_required(VERSION 3.8)
project(hma2_carrobo_tutorial_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)


set(srv_files
    "srv/janken_robot_node.srv"
    "srv/hand_recog.srv"
)

set(action_files
)

# ここに .action / .srv を列挙
rosidl_generate_interfaces(${PROJECT_NAME}
  ${srv_files}
  ${action_files}
  DEPENDENCIES geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

