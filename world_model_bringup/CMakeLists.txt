cmake_minimum_required(VERSION 3.8)
project(world_model_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config demo
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/nav2_world_model_params.py
  scripts/smoke_nav2_loopback.py
  scripts/build_demo_bag.py
  DESTINATION share/${PROJECT_NAME}/scripts
)

ament_package()
