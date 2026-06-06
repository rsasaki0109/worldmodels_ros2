#include "world_model_costmap/world_model_layer.hpp"

#include <algorithm>

#include "pluginlib/class_list_macros.hpp"

namespace world_model_costmap
{

using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::NO_INFORMATION;

void WorldModelLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("WorldModelLayer: unable to lock node");
  }

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("topic", rclcpp::ParameterValue(std::string("/world_model_runtime/future_occupancy")));
  declareParameter("occupied_threshold", rclcpp::ParameterValue(50));

  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".topic", topic_);
  node->get_parameter(name_ + ".occupied_threshold", occupied_threshold_);

  rolling_window_ = layered_costmap_->isRolling();
  default_value_ = NO_INFORMATION;
  WorldModelLayer::matchSize();
  current_ = true;

  sub_ = node->create_subscription<world_model_msgs::msg::FutureOccupancy>(
    topic_, rclcpp::QoS(1),
    std::bind(&WorldModelLayer::occupancyCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    node->get_logger(),
    "WorldModelLayer '%s' subscribed to '%s' (occupied_threshold=%d)",
    name_.c_str(), topic_.c_str(), occupied_threshold_);
}

void WorldModelLayer::matchSize()
{
  nav2_costmap_2d::CostmapLayer::matchSize();
}

void WorldModelLayer::occupancyCallback(world_model_msgs::msg::FutureOccupancy::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  latest_ = msg;
}

void WorldModelLayer::reset()
{
  resetMap(0, 0, getSizeInCellsX(), getSizeInCellsY());
  std::lock_guard<std::mutex> lock(mutex_);
  latest_.reset();
  current_ = false;
}

void WorldModelLayer::updateBounds(
  double robot_x, double robot_y, double /*robot_yaw*/,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) {
    return;
  }

  if (rolling_window_) {
    updateOrigin(robot_x - getSizeInMetersX() / 2, robot_y - getSizeInMetersY() / 2);
  }

  // clear our buffer each cycle so stale predictions do not linger
  resetMap(0, 0, getSizeInCellsX(), getSizeInCellsY());

  world_model_msgs::msg::FutureOccupancy::SharedPtr msg;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    msg = latest_;
  }
  if (!msg) {
    return;
  }

  // Union of predicted occupancy over the whole horizon.
  for (const auto & grid : msg->grids) {
    const double res = grid.info.resolution;
    if (res <= 0.0) {continue;}
    const double ox = grid.info.origin.position.x;
    const double oy = grid.info.origin.position.y;
    const unsigned int w = grid.info.width;
    const unsigned int h = grid.info.height;

    for (unsigned int gj = 0; gj < h; ++gj) {
      for (unsigned int gi = 0; gi < w; ++gi) {
        const int8_t v = grid.data[gj * w + gi];
        if (v < occupied_threshold_) {continue;}
        const double wx = ox + (gi + 0.5) * res;
        const double wy = oy + (gj + 0.5) * res;
        unsigned int mx, my;
        if (worldToMap(wx, wy, mx, my)) {
          setCost(mx, my, LETHAL_OBSTACLE);
          touch(wx, wy, min_x, min_y, max_x, max_y);
        }
      }
    }
  }
}

void WorldModelLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {
    return;
  }
  updateWithMax(master_grid, min_i, min_j, max_i, max_j);
}

}  // namespace world_model_costmap

PLUGINLIB_EXPORT_CLASS(world_model_costmap::WorldModelLayer, nav2_costmap_2d::Layer)
