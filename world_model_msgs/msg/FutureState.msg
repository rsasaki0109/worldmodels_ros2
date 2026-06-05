# The World Model's prediction of how the world evolves.
# Latents are always present; decoded frames are optional and only
# populated by backends that can render pixels.

std_msgs/Header header

float32 dt                    # seconds between predicted steps
LatentState[] latents         # predicted latent per future step (index 0 = t+dt)
sensor_msgs/Image[] frames    # optional decoded future frames (may be empty)
