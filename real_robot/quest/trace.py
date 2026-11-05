#!/usr/bin/env python3
"""Passive recorder for Quest teleop latency: subscribes only, never publishes.

Logs, with arrival times (time.monotonic):
  /delta_pose            hand target pose for the IK (pub_delta_pose, 30 Hz)
  /ik/joint_states       IK solution (before ik_guard)
  /control/joint_states  commands to the driver (ik_guard's arm joints; gripper
                         messages from pub_delta_pose are skipped)
  /feedback/joint_states arm joints from the driver (200 Hz)
  /feedback/tcp_pose     arm TCP pose
and writes them to an .npz on exit (Ctrl-C / launch shutdown). Started by
teleop_quest_piper.launch.py with trace:=<file>.npz; analyze with
real_robot/quest/trace_report.py.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINTS = [f"joint{i}" for i in range(1, 7)]


def _joints(msg: JointState):
    pos = dict(zip(msg.name, msg.position))
    return [pos[j] for j in JOINTS] if all(j in pos for j in JOINTS) else None


def _pose(msg: PoseStamped):
    p, o = msg.pose.position, msg.pose.orientation
    return [p.x, p.y, p.z, o.x, o.y, o.z, o.w]


class Trace(Node):
    def __init__(self):
        super().__init__("quest_trace")
        self.declare_parameter("out", "quest_trace.npz")
        self.out = self.get_parameter("out").value
        self.rows = {k: [] for k in ("delta_pose", "ik", "cmd", "feedback", "tcp")}
        for topic, typ, key, conv in (
                ("/delta_pose", PoseStamped, "delta_pose", _pose),
                ("/ik/joint_states", JointState, "ik", _joints),
                ("/control/joint_states", JointState, "cmd", _joints),
                ("/feedback/joint_states", JointState, "feedback", _joints),
                ("/feedback/tcp_pose", PoseStamped, "tcp", _pose)):
            self.create_subscription(typ, topic, self._cb(key, conv), 50)
        self.get_logger().info(f"quest_trace: recording to {self.out}")

    def _cb(self, key, conv):
        def cb(msg):
            v = conv(msg)
            if v is not None:
                self.rows[key].append([time.monotonic(), *v])
        return cb

    def save(self):
        np.savez(self.out, **{k: np.asarray(v, dtype=np.float64) for k, v in self.rows.items()})
        print(f"quest_trace: wrote {self.out} " + ", ".join(f"{k} {len(v)}" for k, v in self.rows.items()),
              file=sys.stderr)


def main(args=None):
    rclpy.init(args=args)
    node = Trace()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
