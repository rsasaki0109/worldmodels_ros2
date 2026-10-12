cmake_minimum_required(VERSION 3.8)
project(world_model_mppi_critics)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav2_mppi_controller REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(world_model_msgs REQUIRED)

add_library(world_model_time_critic SHARED src/world_model_time_critic.cpp)
target_compile_features(world_model_time_critic PUBLIC cxx_std_17)
target_include_directories(world_model_time_critic PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(world_model_time_critic
  rclcpp nav2_mppi_controller nav2_costmap_2d nav_msgs geometry_msgs pluginlib tf2 tf2_ros
  world_model_msgs)

pluginlib_export_plugin_description_file(nav2_mppi_controller world_model_mppi_critics_plugins.xml)

install(TARGETS world_model_time_critic
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_world_model_time_critic test/test_world_model_time_critic.cpp)
  target_link_libraries(test_world_model_time_critic world_model_time_critic)
endif()

ament_export_include_directories(include)
ament_export_libraries(world_model_time_critic)
ament_export_dependencies(nav2_mppi_controller pluginlib world_model_msgs)

ament_package()
