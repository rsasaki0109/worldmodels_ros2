"""Lifecycle runtime node: the ROS 2 surface of a World Model adapter.

Subscribes to ``~/observation`` and, for each observation, asks the
configured adapter to imagine the future, then publishes the predicted
future state, the per-step occupancy and a risk score.

    ros2 run world_model_py runtime_node --ros-args -p adapter:=dummy

Parameters
----------
adapter : str   registry name (``dummy``, ``remote``, ...). default ``dummy``.
remote_url : str  url for the ``remote`` adapter.
horizon : int   default rollout length when no action is supplied.
"""
from __future__ import annotations

import rclpy
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn

from world_model_msgs.msg import (
    FutureOccupancy as FutureOccupancyMsg,
    FutureState as FutureStateMsg,
    Observation as ObservationMsg,
    RiskScore as RiskScoreMsg,
)

from . import conversions as conv
from .registry import load_model


class WorldModelRuntime(LifecycleNode):
    def __init__(self, **kwargs):
        super().__init__("world_model_runtime", **kwargs)
        self.declare_parameter("adapter", "dummy")
        self.declare_parameter("remote_url", "http://localhost:8080/predict_future")
        self.declare_parameter("model_id", "facebook/ijepa_vith14_1k")
        self.declare_parameter("horizon", 8)
        # autostart: self-transition through configure+activate on boot, so the
        # plain `ros2 launch` / `ros2 run` path "just works" without a separate
        # lifecycle manager. Set false to drive transitions manually.
        self.declare_parameter("autostart", True)

        self._adapter = None
        self._sub = None
        self._pub_future = None
        self._pub_risk = None
        self._pub_occ = None
        self._autostart_timer = None

        if self.get_parameter("autostart").get_parameter_value().bool_value:
            self._autostart_timer = self.create_timer(0.2, self._autostart_once)

    def _autostart_once(self) -> None:
        self._autostart_timer.cancel()
        self.get_logger().info("autostart: configure + activate")
        self.trigger_configure()
        self.trigger_activate()

    # -- lifecycle ---------------------------------------------------------
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        name = self.get_parameter("adapter").get_parameter_value().string_value
        kwargs = {}
        if name == "remote":
            kwargs["url"] = self.get_parameter("remote_url").get_parameter_value().string_value
        elif name == "ijepa":
            kwargs["model_id"] = self.get_parameter("model_id").get_parameter_value().string_value
        try:
            self._adapter = load_model(name, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"failed to load adapter '{name}': {exc}")
            return TransitionCallbackReturn.FAILURE

        self._pub_future = self.create_lifecycle_publisher(FutureStateMsg, "~/future_state", 10)
        self._pub_risk = self.create_lifecycle_publisher(RiskScoreMsg, "~/risk_score", 10)
        self._pub_occ = self.create_lifecycle_publisher(FutureOccupancyMsg, "~/future_occupancy", 10)
        self._sub = self.create_subscription(
            ObservationMsg, "~/observation", self._on_observation, 10
        )
        self.get_logger().info(f"configured world model adapter '{name}': {self._adapter.info()}")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("world model runtime activated")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("world model runtime deactivated")
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self._teardown()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self._teardown()
        return TransitionCallbackReturn.SUCCESS

    def _teardown(self) -> None:
        for attr in ("_sub", "_pub_future", "_pub_risk", "_pub_occ"):
            obj = getattr(self, attr)
            if obj is not None:
                self.destroy_subscription(obj) if attr == "_sub" else self.destroy_publisher(obj)
                setattr(self, attr, None)
        self._adapter = None

    # -- work --------------------------------------------------------------
    def _on_observation(self, msg: ObservationMsg) -> None:
        if self._adapter is None:
            return
        horizon = int(self.get_parameter("horizon").get_parameter_value().integer_value)
        obs = conv.observation_from_msg(msg)
        try:
            pred = self._adapter.predict_future(obs, horizon=horizon)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"predict_future failed: {exc}")
            return

        header = msg.header
        self._pub_future.publish(conv.future_to_msg(pred, header))
        self._pub_risk.publish(conv.risk_to_msg(pred, header))

        occ = FutureOccupancyMsg()
        occ.header = header
        occ.dt = float(pred.dt)
        occ.grids = conv.occupancy_to_grids(pred, header)
        self._pub_occ.publish(occ)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WorldModelRuntime()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
