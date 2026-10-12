#include "world_model_mppi_critics/world_model_time_critic.hpp"

#include <algorithm>
#include <cmath>

#include "pluginlib/class_list_macros.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"

namespace world_model_mppi_critics
{

void WorldModelTimeCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(topic_, "topic", std::string("/world_model_runtime/future_occupancy"));
  getParam(occupied_threshold_, "occupied_threshold", 50);
  getParam(radius_, "radius", 0.55);
  getParam(cost_weight_, "cost_weight", 60.0);
  getParam(cost_power_, "cost_power", 1);
  getParam(max_age_, "max_age", 1.0);
  getParam(transform_tolerance_, "transform_tolerance", 0.5);

  auto node = parent_.lock();
  if (!node) {
    throw std::runtime_error("WorldModelTimeCritic: unable to lock node");
  }
  sub_ = node->create_subscription<world_model_msgs::msg::FutureOccupancy>(
    topic_, rclcpp::QoS(1),
    std::bind(&WorldModelTimeCritic::occupancyCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    logger_, "WorldModelTimeCritic '%s' subscribed to '%s' (threshold=%d, radius=%.2f, weight=%.1f)",
    name_.c_str(), topic_.c_str(), occupied_threshold_, radius_, cost_weight_);
}

WorldModelTimeCritic::Step WorldModelTimeCritic::prepareStep(
  const nav_msgs::msg::OccupancyGrid & grid, int threshold, double radius)
{
  Step st;
  st.width = grid.info.width;
  st.height = grid.info.height;
  st.resolution = grid.info.resolution;
  st.origin_x = grid.info.origin.position.x;
  st.origin_y = grid.info.origin.position.y;
  st.occupied.assign(static_cast<size_t>(st.width) * st.height, 0);
  if (st.resolution <= 0.0 || grid.data.size() < st.occupied.size()) {
    return st;
  }
  const int r = static_cast<int>(std::ceil(radius / st.resolution));
  const double r2 = radius * radius;
  const int w = static_cast<int>(st.width), h = static_cast<int>(st.height);
  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < w; ++i) {
      if (grid.data[j * w + i] < threshold) {continue;}
      for (int dj = -r; dj <= r; ++dj) {
        const int jj = j + dj;
        if (jj < 0 || jj >= h) {continue;}
        for (int di = -r; di <= r; ++di) {
          const int ii = i + di;
          if (ii < 0 || ii >= w) {continue;}
          // cell-centre distance, so a radius below one cell still marks the cell itself
          const double dx = di * st.resolution, dy = dj * st.resolution;
          if (dx * dx + dy * dy <= r2 + 1e-9) {
            st.occupied[jj * w + ii] = 1;
          }
        }
      }
    }
  }
  return st;
}

int WorldModelTimeCritic::stepIndex(double tau, double dt, int horizon)
{
  if (dt <= 0.0 || horizon <= 0 || tau < 0.0 || tau > (horizon + 0.5) * dt) {
    return -1;
  }
  const int k = static_cast<int>(std::lround(tau / dt)) - 1;
  return std::clamp(k, 0, horizon - 1);
}

void WorldModelTimeCritic::occupancyCallback(
  world_model_msgs::msg::FutureOccupancy::SharedPtr msg)
{
  if (msg->grids.empty()) {
    return;
  }
  auto p = std::make_shared<Prediction>();
  p->frame_id = msg->grids[0].header.frame_id.empty() ?
    msg->header.frame_id : msg->grids[0].header.frame_id;
  p->stamp = msg->header.stamp;
  p->dt = msg->dt > 0.0 ? msg->dt : 0.5;
  for (const auto & g : msg->grids) {
    p->steps.push_back(prepareStep(g, occupied_threshold_, radius_));
  }
  std::lock_guard<std::mutex> lock(mutex_);
  prediction_ = p;
}

void WorldModelTimeCritic::score(mppi::CriticData & data)
{
  if (!enabled_) {
    return;
  }
  std::shared_ptr<const Prediction> p;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    p = prediction_;
  }
  auto node = parent_.lock();
  if (!p || !node) {
    return;
  }
  const rclcpp::Time stamp(p->stamp, node->get_clock()->get_clock_type());
  const double age = (node->get_clock()->now() - stamp).seconds();
  if (age > max_age_ || age < -transform_tolerance_) {
    return;   // stale (or from the future): do not steer on it
  }

  // costmap global frame <- grid frame
  const std::string global = costmap_ros_->getGlobalFrameID();
  double tx = 0.0, ty = 0.0, c = 1.0, s = 0.0;
  if (!p->frame_id.empty() && p->frame_id != global) {
    geometry_msgs::msg::TransformStamped tf;
    auto buffer = costmap_ros_->getTfBuffer();
    try {
      const tf2::TimePoint t = tf2_ros::fromMsg(p->stamp);
      if (buffer->canTransform(global, p->frame_id, t)) {
        tf = buffer->lookupTransform(global, p->frame_id, t);
      } else {
        tf = buffer->lookupTransform(global, p->frame_id, tf2::TimePointZero);
        if (std::abs(tf2::durationToSec(tf2_ros::fromMsg(tf.header.stamp) - t)) > transform_tolerance_) {
          return;
        }
      }
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        logger_, *node->get_clock(), 2000, "WorldModelTimeCritic: no transform %s -> %s: %s",
        p->frame_id.c_str(), global.c_str(), ex.what());
      return;
    }
    tx = tf.transform.translation.x;
    ty = tf.transform.translation.y;
    const auto & q = tf.transform.rotation;
    const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    c = std::cos(yaw);
    s = std::sin(yaw);
  }

  const auto & X = data.trajectories.x;
  const auto & Y = data.trajectories.y;
  const size_t batch = X.shape(0), steps = X.shape(1);
  const int horizon = static_cast<int>(p->steps.size());
  const double dt = data.model_dt;
  for (size_t b = 0; b < batch; ++b) {
    double overlap = 0.0;   // seconds this trajectory spends in predicted occupancy
    for (size_t j = 0; j < steps; ++j) {
      const double tau = age + (j + 1) * dt;
      const int k = stepIndex(tau, p->dt, horizon);
      if (k < 0) {
        if (tau > 0.0) {break;}   // past the horizon
        continue;
      }
      const Step & st = p->steps[k];
      if (st.resolution <= 0.0) {continue;}
      // global -> grid frame
      const double wx = X(b, j) - tx, wy = Y(b, j) - ty;
      const double gx = c * wx + s * wy, gy = -s * wx + c * wy;
      const double fi = (gx - st.origin_x) / st.resolution, fj = (gy - st.origin_y) / st.resolution;
      if (fi < 0.0 || fj < 0.0 || fi >= st.width || fj >= st.height) {continue;}
      if (st.occupied[static_cast<size_t>(fj) * st.width + static_cast<size_t>(fi)]) {
        overlap += dt;
      }
    }
    if (overlap > 0.0) {
      data.costs(b) += static_cast<float>(cost_weight_ * std::pow(overlap, cost_power_));
    }
  }
}

}  // namespace world_model_mppi_critics

PLUGINLIB_EXPORT_CLASS(world_model_mppi_critics::WorldModelTimeCritic, mppi::critics::CriticFunction)
