cmake_minimum_required(VERSION 3.8)
project(world_model_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config demo gazebo
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/nav2_world_model_params.py
  scripts/smoke_nav2_loopback.py
  scripts/build_demo_bag.py
  scripts/gz_ped_mover.py
  scripts/gz_nav_eval.py
  scripts/gz_data_drive.py
  DESTINATION share/${PROJECT_NAME}/scripts
)

ament_package()
