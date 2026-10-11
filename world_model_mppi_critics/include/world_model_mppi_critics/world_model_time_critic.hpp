#ifndef WORLD_MODEL_MPPI_CRITICS__WORLD_MODEL_TIME_CRITIC_HPP_
#define WORLD_MODEL_MPPI_CRITICS__WORLD_MODEL_TIME_CRITIC_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "builtin_interfaces/msg/time.hpp"
#include "nav2_mppi_controller/critic_function.hpp"
#include "rclcpp/rclcpp.hpp"
#include "world_model_msgs/msg/future_occupancy.hpp"

namespace world_model_mppi_critics
{

/**
 * MPPI critic that scores each sampled trajectory against a World Model's
 * FutureOccupancy *in time*: the trajectory point reached at time t is checked
 * against the predicted grid for t (prediction age included), not against a
 * union of the whole horizon. A trajectory that would be where a pedestrian is
 * predicted to be at that moment is penalised; one that waits, or passes after
 * the pedestrian has gone, is not.
 *
 * Grids are moved from their header frame (e.g. base_link at the prediction
 * stamp) into the costmap's global frame through TF. Each step's occupied
 * cells are dilated by `radius` (robot + obstacle clearance) once per message.
 */
class WorldModelTimeCritic : public mppi::critics::CriticFunction
{
public:
  void initialize() override;
  void score(mppi::CriticData & data) override;

  /** Occupied cells of one prediction step, dilated, in the grid frame. */
  struct Step
  {
    unsigned int width{0}, height{0};
    double resolution{0.0}, origin_x{0.0}, origin_y{0.0};
    std::vector<unsigned char> occupied;   // row-major, 1 = within `radius`
  };

  /** One prepared prediction. */
  struct Prediction
  {
    std::string frame_id;
    builtin_interfaces::msg::Time stamp;
    double dt{0.5};
    std::vector<Step> steps;               // steps[k] is t = stamp + (k + 1) * dt
  };

  /** Dilate one grid's cells >= threshold by `radius` (m). Exposed for tests. */
  static Step prepareStep(const nav_msgs::msg::OccupancyGrid & grid, int threshold, double radius);

  /**
   * Index of the prediction step for a point `tau` seconds after the
   * prediction stamp, or -1 when the horizon does not cover it. Steps are at
   * (k + 1) * dt; a point is matched to the nearest one, the first step also
   * covering [0, dt) and the last one up to half a step past the horizon.
   */
  static int stepIndex(double tau, double dt, int horizon);

protected:
  void occupancyCallback(world_model_msgs::msg::FutureOccupancy::SharedPtr msg);

  rclcpp::Subscription<world_model_msgs::msg::FutureOccupancy>::SharedPtr sub_;
  std::shared_ptr<const Prediction> prediction_;
  std::mutex mutex_;

  std::string topic_;
  int occupied_threshold_{50};
  double radius_{0.55};
  double cost_weight_{60.0};
  int cost_power_{1};
  double max_age_{1.0};
  double transform_tolerance_{0.5};
};

}  // namespace world_model_mppi_critics

#endif  // WORLD_MODEL_MPPI_CRITICS__WORLD_MODEL_TIME_CRITIC_HPP_
