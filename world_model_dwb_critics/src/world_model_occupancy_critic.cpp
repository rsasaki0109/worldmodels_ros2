#include "world_model_dwb_critics/world_model_occupancy_critic.hpp"

#include <cmath>

#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace world_model_dwb_critics
{

void WorldModelOccupancyCritic::onInit()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("WorldModelOccupancyCritic: unable to lock node");
  }

  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".topic", rclcpp::ParameterValue(std::string("/world_model_runtime/future_occupancy")));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".occupied_threshold", rclcpp::ParameterValue(50));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".collision_radius", rclcpp::ParameterValue(0.18));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".collision_penalty", rclcpp::ParameterValue(10.0));

  node->get_parameter(name_ + ".topic", topic_);
  node->get_parameter(name_ + ".occupied_threshold", occupied_threshold_);
  node->get_parameter(name_ + ".collision_radius", collision_radius_);
  node->get_parameter(name_ + ".collision_penalty", collision_penalty_);

  sub_ = node->create_subscription<world_model_msgs::msg::FutureOccupancy>(
    topic_, rclcpp::QoS(1),
    std::bind(&WorldModelOccupancyCritic::occupancyCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    node->get_logger(),
    "WorldModelOccupancyCritic '%s' subscribed to '%s' (threshold=%d, radius=%.2f)",
    name_.c_str(), topic_.c_str(), occupied_threshold_, collision_radius_);
}

void WorldModelOccupancyCritic::occupancyCallback(
  world_model_msgs::msg::FutureOccupancy::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  latest_ = msg;
}

void WorldModelOccupancyCritic::rebuildLethalCells()
{
  lethal_cells_.clear();
  if (!latest_) {
    return;
  }
  for (const auto & grid : latest_->grids) {
    const double res = grid.info.resolution;
    if (res <= 0.0) {
      continue;
    }
    const double ox = grid.info.origin.position.x;
    const double oy = grid.info.origin.position.y;
    const unsigned int w = grid.info.width;
    const unsigned int h = grid.info.height;
    for (unsigned int j = 0; j < h; ++j) {
      for (unsigned int i = 0; i < w; ++i) {
        const int8_t v = grid.data[j * w + i];
        if (v < occupied_threshold_) {
          continue;
        }
        lethal_cells_.emplace_back(ox + (i + 0.5) * res, oy + (j + 0.5) * res);
      }
    }
  }
}

bool WorldModelOccupancyCritic::prepare(
  const geometry_msgs::msg::Pose2D & /*pose*/,
  const nav_2d_msgs::msg::Twist2D & /*vel*/,
  const geometry_msgs::msg::Pose2D & /*goal*/,
  const nav_2d_msgs::msg::Path2D & /*global_plan*/)
{
  std::lock_guard<std::mutex> lock(mutex_);
  rebuildLethalCells();
  return true;
}

double WorldModelOccupancyCritic::scoreTrajectory(const dwb_msgs::msg::Trajectory2D & traj)
{
  if (lethal_cells_.empty() || traj.poses.empty()) {
    return 0.0;
  }
  const double r2 = collision_radius_ * collision_radius_;
  double score = 0.0;
  for (const auto & pose : traj.poses) {
    for (const auto & cell : lethal_cells_) {
      const double dx = pose.x - cell.first;
      const double dy = pose.y - cell.second;
      if (dx * dx + dy * dy <= r2) {
        score += collision_penalty_;
        break;
      }
    }
  }
  return score;
}

}  // namespace world_model_dwb_critics

PLUGINLIB_EXPORT_CLASS(
  world_model_dwb_critics::WorldModelOccupancyCritic,
  dwb_core::TrajectoryCritic)
