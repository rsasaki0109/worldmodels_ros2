#include "world_model_costmap/world_model_layer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "pluginlib/class_list_macros.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"

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
  declareParameter("transform_tolerance", rclcpp::ParameterValue(0.5));
  // cost per horizon step (step k uses entry min(k, size - 1)); empty = every
  // step LETHAL. Graded, non-lethal costs keep the controller able to rank
  // trajectories when a prediction reaches the robot.
  declareParameter("step_costs", rclcpp::ParameterValue(std::vector<int64_t>{}));
  // never mark predictions within this radius (m) of the robot
  declareParameter("robot_clear_radius", rclcpp::ParameterValue(0.0));

  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".topic", topic_);
  node->get_parameter(name_ + ".occupied_threshold", occupied_threshold_);
  node->get_parameter(name_ + ".transform_tolerance", transform_tolerance_);
  std::vector<int64_t> step_costs;
  node->get_parameter(name_ + ".step_costs", step_costs);
  step_costs_.clear();
  for (const auto c : step_costs) {
    step_costs_.push_back(static_cast<unsigned char>(std::clamp<int64_t>(c, 1, LETHAL_OBSTACLE)));
  }
  node->get_parameter(name_ + ".robot_clear_radius", robot_clear_radius_);

  rolling_window_ = layered_costmap_->isRolling();
  default_value_ = NO_INFORMATION;
  WorldModelLayer::matchSize();
  current_ = true;

  sub_ = node->create_subscription<world_model_msgs::msg::FutureOccupancy>(
    topic_, rclcpp::QoS(1),
    std::bind(&WorldModelLayer::occupancyCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    node->get_logger(),
    "WorldModelLayer '%s' subscribed to '%s' (occupied_threshold=%d, %s, robot_clear_radius=%.2f)",
    name_.c_str(), topic_.c_str(), occupied_threshold_,
    step_costs_.empty() ? "all steps lethal" : "graded step costs", robot_clear_radius_);
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

  // ...and make the master reset (and so clear) what we marked last cycle:
  // it only resets inside reported bounds.
  if (has_last_bounds_) {
    touch(last_min_x_, last_min_y_, min_x, min_y, max_x, max_y);
    touch(last_max_x_, last_max_y_, min_x, min_y, max_x, max_y);
    has_last_bounds_ = false;
  }

  world_model_msgs::msg::FutureOccupancy::SharedPtr msg;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    msg = latest_;
  }
  if (!msg) {
    return;
  }

  double mark_min_x = std::numeric_limits<double>::max();
  double mark_min_y = std::numeric_limits<double>::max();
  double mark_max_x = std::numeric_limits<double>::lowest();
  double mark_max_y = std::numeric_limits<double>::lowest();

  const std::string global_frame = layered_costmap_->getGlobalFrameID();

  // Union of predicted occupancy over the whole horizon (max cost per cell).
  const double clear2 = robot_clear_radius_ * robot_clear_radius_;
  for (size_t k = 0; k < msg->grids.size(); ++k) {
    const auto & grid = msg->grids[k];
    const unsigned char cost = step_costs_.empty() ?
      LETHAL_OBSTACLE : step_costs_[std::min(k, step_costs_.size() - 1)];
    const double res = grid.info.resolution;
    if (res <= 0.0) {continue;}
    const double ox = grid.info.origin.position.x;
    const double oy = grid.info.origin.position.y;
    const unsigned int w = grid.info.width;
    const unsigned int h = grid.info.height;

    // grid frame -> costmap global frame (2D rigid transform)
    double tx = 0.0, ty = 0.0, c = 1.0, s = 0.0;
    const std::string & frame = grid.header.frame_id;
    if (!frame.empty() && frame != global_frame) {
      geometry_msgs::msg::TransformStamped tf;
      try {
        const tf2::TimePoint stamp = tf2_ros::fromMsg(grid.header.stamp);
        if (tf_->canTransform(global_frame, frame, stamp)) {
          tf = tf_->lookupTransform(global_frame, frame, stamp);
        } else {
          tf = tf_->lookupTransform(global_frame, frame, tf2::TimePointZero);
          const double age = std::abs(
            tf2::durationToSec(tf2_ros::fromMsg(tf.header.stamp) - stamp));
          if (stamp != tf2::TimePointZero && age > transform_tolerance_) {
            continue;
          }
        }
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(
          logger_, *clock_, 2000, "WorldModelLayer: no transform %s -> %s: %s",
          frame.c_str(), global_frame.c_str(), ex.what());
        continue;
      }
      tx = tf.transform.translation.x;
      ty = tf.transform.translation.y;
      const auto & q = tf.transform.rotation;
      const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
      c = std::cos(yaw);
      s = std::sin(yaw);
    }

    // cover each grid cell's whole footprint when the costmap is finer than
    // the grid (sub-sample at below the costmap resolution, so no holes even
    // when the grid is rotated)
    const int sub = std::max(1, static_cast<int>(std::ceil(res / (0.7 * getResolution()))));
    for (unsigned int gj = 0; gj < h; ++gj) {
      for (unsigned int gi = 0; gi < w; ++gi) {
        const int8_t v = grid.data[gj * w + gi];
        if (v < occupied_threshold_) {continue;}
        for (int sj = 0; sj < sub; ++sj) {
          for (int si = 0; si < sub; ++si) {
            const double gx = ox + (gi + (si + 0.5) / sub) * res;
            const double gy = oy + (gj + (sj + 0.5) / sub) * res;
            const double wx = tx + c * gx - s * gy;
            const double wy = ty + s * gx + c * gy;
            if (clear2 > 0.0 &&
              (wx - robot_x) * (wx - robot_x) + (wy - robot_y) * (wy - robot_y) < clear2)
            {
              continue;
            }
            unsigned int mx, my;
            if (worldToMap(wx, wy, mx, my)) {
              const unsigned char cur = getCost(mx, my);
              if (cur == NO_INFORMATION || cur < cost) {
                setCost(mx, my, cost);
              }
              touch(wx, wy, min_x, min_y, max_x, max_y);
              touch(wx, wy, &mark_min_x, &mark_min_y, &mark_max_x, &mark_max_y);
            }
          }
        }
      }
    }
  }

  if (mark_min_x <= mark_max_x) {
    has_last_bounds_ = true;
    last_min_x_ = mark_min_x;
    last_min_y_ = mark_min_y;
    last_max_x_ = mark_max_x;
    last_max_y_ = mark_max_y;
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
