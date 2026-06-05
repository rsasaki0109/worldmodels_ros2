cmake_minimum_required(VERSION 3.8)
project(world_model_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LatentState.msg"
  "msg/Observation.msg"
  "msg/ActionCondition.msg"
  "msg/FutureState.msg"
  "msg/FutureOccupancy.msg"
  "msg/RiskScore.msg"
  "msg/Rollout.msg"
  "action/PredictFuture.action"
  DEPENDENCIES std_msgs sensor_msgs nav_msgs
)

ament_package()
