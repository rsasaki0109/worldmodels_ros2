#ifndef WORLD_MODEL_COSTMAP__WORLD_MODEL_LAYER_HPP_
#define WORLD_MODEL_COSTMAP__WORLD_MODEL_LAYER_HPP_

#include <mutex>
#include <string>

#include "nav2_costmap_2d/costmap_layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "rclcpp/rclcpp.hpp"
#include "world_model_msgs/msg/future_occupancy.hpp"

namespace world_model_costmap
{

/**
 * A Nav2 costmap layer that stamps a World Model's predicted FutureOccupancy
 * into the costmap, so any planner/controller avoids where the model thinks
 * obstacles will be over the prediction horizon (union of all future steps).
 *
 * Assumption: the FutureOccupancy grids are expressed in the costmap's global
 * frame (the dummy/ijepa runtime publishes grids with an explicit origin). TF
 * reprojection of a differing frame is left as future work.
 */
class WorldModelLayer : public nav2_costmap_2d::CostmapLayer
{
public:
  WorldModelLayer() = default;

  void onInitialize() override;
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;
  void reset() override;
  bool isClearable() override {return true;}
  void matchSize() override;

private:
  void occupancyCallback(world_model_msgs::msg::FutureOccupancy::SharedPtr msg);

  rclcpp::Subscription<world_model_msgs::msg::FutureOccupancy>::SharedPtr sub_;
  world_model_msgs::msg::FutureOccupancy::SharedPtr latest_;
  std::mutex mutex_;

  std::string topic_;
  int occupied_threshold_{50};
  bool rolling_window_{false};
};

}  // namespace world_model_costmap

#endif  // WORLD_MODEL_COSTMAP__WORLD_MODEL_LAYER_HPP_
