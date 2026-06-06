<!-- Thanks for the PR! Keep it focused; see CONTRIBUTING.md. -->

## What & why

<!-- What does this change and what problem does it solve? -->

## How verified

<!-- Commands run + result. This project values evidence over vibes. -->

- [ ] `colcon test` passes
- [ ] ROS-free unit tests pass (`world_model_py` pytest)
- [ ] Touched the `ijepa`/GPU path? ran `test/gpu_verify_*.py` locally
- [ ] Updated README / CHANGELOG if behavior or interfaces changed

## Design rules

- [ ] Adapters still take numpy/dataclasses (no ROS messages); heavy deps stay lazy-imported
- [ ] Mocks / structural-only verification are labeled honestly
