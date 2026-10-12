#include <gtest/gtest.h>

#include "world_model_mppi_critics/world_model_time_critic.hpp"

using world_model_mppi_critics::WorldModelTimeCritic;

TEST(WorldModelTimeCritic, StepIndexMatchesNearestPredictionStep)
{
  // steps at 0.5, 1.0, 1.5, 2.0 s
  EXPECT_EQ(WorldModelTimeCritic::stepIndex(0.05, 0.5, 4), 0);   // before the first step
  EXPECT_EQ(WorldModelTimeCritic::stepIndex(0.5, 0.5, 4), 0);
  EXPECT_EQ(WorldModelTimeCritic::stepIndex(0.8, 0.5, 4), 1);
  EXPECT_EQ(WorldModelTimeCritic::stepIndex(1.6, 0.5, 4), 2);
  EXPECT_EQ(WorldModelTimeCritic::stepIndex(2.0, 0.5, 4), 3);
  EXPECT_EQ(WorldModelTimeCritic::stepIndex(2.2, 0.5, 4), 3);     // within half a step
  EXPECT_EQ(WorldModelTimeCritic::stepIndex(2.3, 0.5, 4), -1);    // past the horizon
  EXPECT_EQ(WorldModelTimeCritic::stepIndex(-0.1, 0.5, 4), -1);
  EXPECT_EQ(WorldModelTimeCritic::stepIndex(1.0, 0.5, 0), -1);
}

TEST(WorldModelTimeCritic, PrepareStepDilatesOccupiedCellsByRadius)
{
  nav_msgs::msg::OccupancyGrid g;
  g.info.width = 11;
  g.info.height = 11;
  g.info.resolution = 0.25;
  g.info.origin.position.x = -1.0;
  g.info.origin.position.y = 2.0;
  g.data.assign(121, 0);
  g.data[5 * 11 + 5] = 80;     // occupied centre
  g.data[0] = -1;              // unknown is free
  g.data[1] = 49;              // below threshold

  const auto st = WorldModelTimeCritic::prepareStep(g, 50, 0.5);
  EXPECT_EQ(st.width, 11u);
  EXPECT_DOUBLE_EQ(st.origin_x, -1.0);
  EXPECT_DOUBLE_EQ(st.origin_y, 2.0);
  auto at = [&](int i, int j) {return st.occupied[j * 11 + i];};
  EXPECT_EQ(at(5, 5), 1);
  EXPECT_EQ(at(7, 5), 1);      // 0.5 m away: inside
  EXPECT_EQ(at(5, 3), 1);
  EXPECT_EQ(at(7, 7), 0);      // 0.71 m away: outside
  EXPECT_EQ(at(8, 5), 0);      // 0.75 m
  EXPECT_EQ(at(0, 0), 0);
  EXPECT_EQ(at(1, 0), 0);
  int n = 0;
  for (auto v : st.occupied) {n += v;}
  EXPECT_EQ(n, 13);            // disk of radius 2 cells
}

TEST(WorldModelTimeCritic, PrepareStepToleratesBadGrids)
{
  nav_msgs::msg::OccupancyGrid g;
  g.info.width = 4;
  g.info.height = 4;
  g.info.resolution = 0.0;
  g.data.assign(16, 100);
  auto st = WorldModelTimeCritic::prepareStep(g, 50, 0.5);
  for (auto v : st.occupied) {EXPECT_EQ(v, 0);}
  g.info.resolution = 0.25;
  g.data.resize(3);            // truncated data
  st = WorldModelTimeCritic::prepareStep(g, 50, 0.5);
  for (auto v : st.occupied) {EXPECT_EQ(v, 0);}
}
