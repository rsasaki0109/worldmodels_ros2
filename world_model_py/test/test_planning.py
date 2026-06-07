"""Latent-space planner + retrieval dynamics. No ROS, no GPU, no torch.

We build a toy 2-D "latent" world where moving is literally adding the action to
the state, record a memory of real transitions, and check that:
  * RetrievalDynamics recovers that dynamics from memory alone (no training),
  * plan_to_goal finds an action sequence whose imagined future reaches a goal,
  * decode_trajectory maps imagined latents back to the nearest real frame.
"""
import numpy as np

from world_model_py.planning import (
    RetrievalDynamics,
    cosine_distance,
    decode_trajectory,
    plan_to_goal,
)


def _toy_memory(n=2000, seed=0):
    """A linear system: next = state + action, over a 2-D box. Returns the
    transition memory plus a tag frame per entry (here: the next state itself)."""
    rng = np.random.default_rng(seed)
    states = rng.uniform(-3, 3, size=(n, 2)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    nexts = states + actions
    frames = nexts.copy()  # stand-in for "the image observed at next state"
    return states, actions, nexts, frames


def test_cosine_distance_basics():
    a = np.array([1.0, 0.0])
    assert cosine_distance(a, a) < 1e-5
    assert cosine_distance(a, np.array([0.0, 1.0])) == 1.0          # orthogonal
    assert cosine_distance(a, np.array([-1.0, 0.0])) > 1.9          # opposite


def test_retrieval_dynamics_recovers_linear_system():
    s, a, n, _ = _toy_memory()
    dyn = RetrievalDynamics(s, a, n, k=8)
    # a fresh (state, action) not in memory: prediction ~= state + action.
    pred = dyn.step(np.array([0.5, -0.5]), np.array([0.3, 0.2]))
    assert np.allclose(pred, [0.8, -0.3], atol=0.15)


def test_step_with_index_points_into_memory():
    s, a, n, _ = _toy_memory()
    dyn = RetrievalDynamics(s, a, n, k=8)
    _, idx = dyn.step_with_index(s[7], a[7])
    assert 0 <= idx < len(s)


def test_plan_reaches_goal_latent():
    s, a, n, _ = _toy_memory()
    dyn = RetrievalDynamics(s, a, n, k=8)
    start = np.array([-2.0, -2.0], dtype=np.float32)
    goal = np.array([2.0, 2.0], dtype=np.float32)          # cosine goal: up-right
    res = plan_to_goal(
        dyn, start, goal, action_dim=2, horizon=10,
        samples=256, iterations=5, action_low=-1.0, action_high=1.0, seed=1,
    )
    assert res.horizon == 10
    # imagined endpoint should point toward the goal direction.
    assert cosine_distance(res.latents[-1], goal) < 0.1
    # CEM should not get worse over iterations.
    assert res.costs[-1] <= res.costs[0] + 1e-6


def test_plan_is_better_than_no_motion():
    s, a, n, _ = _toy_memory()
    dyn = RetrievalDynamics(s, a, n, k=8)
    start = np.array([-2.0, 0.1], dtype=np.float32)
    goal = np.array([2.0, 0.1], dtype=np.float32)
    res = plan_to_goal(dyn, start, goal, action_dim=2, horizon=8, samples=200,
                       iterations=4, seed=2)
    assert res.cost < cosine_distance(start, goal)


def test_decode_trajectory_retrieves_nearest_frame():
    s, a, n, frames = _toy_memory()
    # imagine a single latent right next to a known next-state; decode it.
    target = n[123]
    out = decode_trajectory([target], n, frames)
    assert np.allclose(out[0], frames[123], atol=1e-4)
