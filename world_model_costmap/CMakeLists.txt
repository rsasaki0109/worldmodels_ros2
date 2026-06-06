cmake_minimum_required(VERSION 3.8)
project(world_model_costmap)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(world_model_msgs REQUIRED)

add_library(world_model_layer SHARED src/world_model_layer.cpp)
target_compile_features(world_model_layer PUBLIC cxx_std_17)
target_include_directories(world_model_layer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(world_model_layer
  rclcpp
  rclcpp_lifecycle
  nav2_costmap_2d
  pluginlib
  nav_msgs
  world_model_msgs)

pluginlib_export_plugin_description_file(nav2_costmap_2d world_model_costmap_plugins.xml)

install(TARGETS world_model_layer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_libraries(world_model_layer)
ament_export_dependencies(nav2_costmap_2d pluginlib world_model_msgs)

ament_package()
