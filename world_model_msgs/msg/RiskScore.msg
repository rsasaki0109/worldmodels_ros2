# A model-based risk/surprise score for a predicted future.
# Deliberately simple so it can gate VLA actions or walking commands.

std_msgs/Header header

float32 score         # normalized risk in [0, 1] (higher = riskier)
float32 confidence    # adapter confidence in the score, [0, 1]
string label          # what the risk is about, e.g. "collision", "fall", "surprise"
