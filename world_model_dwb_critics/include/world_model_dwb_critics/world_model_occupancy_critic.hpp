#ifndef WORLD_MODEL_DWB_CRITICS__WORLD_MODEL_OCCUPANCY_CRITIC_HPP_
#define WORLD_MODEL_DWB_CRITICS__WORLD_MODEL_OCCUPANCY_CRITIC_HPP_

#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "dwb_core/trajectory_critic.hpp"
#include "rclcpp/rclcpp.hpp"
#include "world_model_msgs/msg/future_occupancy.hpp"

namespace world_model_dwb_critics
{

class WorldModelOccupancyCritic : public dwb_core::TrajectoryCritic
{
public:
  void onInit() override;
  bool prepare(
    const geometry_msgs::msg::Pose2D & pose,
    const nav_2d_msgs::msg::Twist2D & vel,
    const geometry_msgs::msg::Pose2D & goal,
    const nav_2d_msgs::msg::Path2D & global_plan) override;
  double scoreTrajectory(const dwb_msgs::msg::Trajectory2D & traj) override;

private:
  void occupancyCallback(world_model_msgs::msg::FutureOccupancy::SharedPtr msg);
  void rebuildLethalCells();

  rclcpp::Subscription<world_model_msgs::msg::FutureOccupancy>::SharedPtr sub_;
  world_model_msgs::msg::FutureOccupancy::SharedPtr latest_;
  std::mutex mutex_;
  std::vector<std::pair<double, double>> lethal_cells_;

  std::string topic_;
  int occupied_threshold_{50};
  double collision_radius_{0.18};
  double collision_penalty_{10.0};
};

}  // namespace world_model_dwb_critics

#endif  // WORLD_MODEL_DWB_CRITICS__WORLD_MODEL_OCCUPANCY_CRITIC_HPP_
