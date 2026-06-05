# A candidate action sequence to condition a World Model rollout on.
# "What would happen if the robot executed these actions?"

std_msgs/Header header

float32[] action      # flattened action sequence (horizon * action_dim, row-major)
uint32 action_dim     # dimension of a single action step
uint32 horizon        # number of future action steps
float32 dt            # seconds between steps (0 -> backend default)
