cmake_minimum_required(VERSION 3.8)
project(world_model_dwb_critics)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(dwb_core REQUIRED)
find_package(dwb_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(pluginlib REQUIRED)
find_package(world_model_msgs REQUIRED)

add_library(world_model_occupancy_critic SHARED src/world_model_occupancy_critic.cpp)
target_compile_features(world_model_occupancy_critic PUBLIC cxx_std_17)
target_include_directories(world_model_occupancy_critic PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(world_model_occupancy_critic
  rclcpp dwb_core dwb_msgs geometry_msgs nav2_costmap_2d nav2_util pluginlib world_model_msgs)

pluginlib_export_plugin_description_file(dwb_core world_model_dwb_critics_plugins.xml)

install(TARGETS world_model_occupancy_critic
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_libraries(world_model_occupancy_critic)
ament_export_dependencies(dwb_core pluginlib world_model_msgs)

ament_package()
