#!/usr/bin/env python3
"""Upstream's arm_ik_pose_node with two latency fixes (same solver and params).

A traced run (2026-10-08, trace_report.py) had the IK at 18 Hz with 164 gaps
over 100 ms and 340 ms of delay, though replaying the same hand targets
offline solves in 18 ms (p95 33 ms). Upstream's node:
  - queues 10 hand poses, so a slow solve leaves it solving stale ones;
  - resets the solver's start and smoothing reference to the arm's measured
    joints on every feedback message (200 Hz). The arm lags the IK, so each
    solve restarts from the lagging arm and, with the smoothing term, only
    creeps toward the hand: the arm moved in bursts.
Here the pose queue holds 1 (latest only), and feedback syncs the solver only
while no hand target has arrived for SYNC_IDLE_S (before you engage, after B)
or right after A (/quest/engage from quest_delta.py: re-anchored at the arm)
or /quest/reanchor (ik_guard, the arm held at a limit: re-anchored there too).
"""
from __future__ import annotations

import os
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
from rclpy.executors import ExternalShutdownException

sys.path.insert(0, os.path.expanduser("~/QuestArmTeleop/install/oculus_reader/lib/oculus_reader"))
from arm_ik_pose_node import ArmIKPoseNode  # noqa: E402

SYNC_IDLE_S = 0.3


class LatestPoseIKNode(ArmIKPoseNode):
    def __init__(self):
        self._last_pose_t = 0.0
        self._force_sync = False
        super().__init__()
        self.create_subscription(Empty, "/quest/engage", self._on_engage, 1)
        self.create_subscription(Empty, "/quest/reanchor", self._on_engage, 1)  # ik_guard at a limit

    def _on_engage(self, msg) -> None:
        self._force_sync = True

    def create_subscription(self, msg_type, topic, callback, qos_profile, **kw):
        if msg_type is PoseStamped:
            qos_profile = 1
        return super().create_subscription(msg_type, topic, callback, qos_profile, **kw)

    def pose_stamped_callback(self, msg) -> None:
        self._last_pose_t = time.monotonic()
        super().pose_stamped_callback(msg)

    def feedback_joint_callback(self, msg) -> None:
        if self._force_sync or time.monotonic() - self._last_pose_t > SYNC_IDLE_S:
            self._force_sync = False
            super().feedback_joint_callback(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LatestPoseIKNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
