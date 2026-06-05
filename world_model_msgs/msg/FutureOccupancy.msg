# Predicted occupancy over a future horizon, one grid per step.
# Consumed by Nav2 costmap layers / Autoware evaluators / RViz.

std_msgs/Header header

float32 dt                       # seconds between predicted grids
nav_msgs/OccupancyGrid[] grids   # predicted occupancy per step (index 0 = t+dt)
