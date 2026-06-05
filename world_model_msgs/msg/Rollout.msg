# One action-conditioned rollout: the action that was imagined, the
# predicted future it produced, and the risk of that future.
# This is the unit a trajectory scorer iterates over.

std_msgs/Header header

ActionCondition action    # the conditioning action sequence
FutureState future        # the predicted evolution
RiskScore risk            # risk of this rollout
