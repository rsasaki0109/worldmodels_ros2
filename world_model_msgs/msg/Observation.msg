# A bundle of the current robot/world observation fed into a World Model.
# All fields are optional; an adapter consumes only what it needs.
# This keeps the contract usable for camera-only, state-only, or
# multimodal models without versioning the message per backend.

std_msgs/Header header

sensor_msgs/Image image       # primary camera frame (optional; empty -> unused)
float32[] ego_state           # generic ego state vector (pose/vel/joint/...)
float32[] action_history      # flattened recent actions (most-recent last)
uint32 action_dim             # dimension of one action in action_history (0 -> unknown)
string instruction            # natural-language instruction/goal (optional)
