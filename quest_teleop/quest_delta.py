#!/usr/bin/env python3
"""Controller buttons and motion -> IK target, replacing upstream's pub_delta_pose.

mode "room" (default since 2026-10-15): 1:1 in the room. The hand's
displacement since engaging, in the room (the Quest's tracking frame, z up),
moves the gripper the same distance (x scale) the same way in the robot's
base frame, and the hand's rotation about room axes turns the gripper about
the same axes (room_target). Your frame is calibrated once per session with
two strokes of the right controller, each with the right grip held: straight
forward ~15 cm, then straight up ~15 cm (CALIB_MIN_M). Two strokes, not one,
because the Quest reports poses relative to the headset, which lies on the
table at any angle (2026-10-15: forward strokes read < 8 cm "horizontal").
Forward maps to forward_yaw_deg in the base frame: by default the direction
the arm reaches at home (facing the table).
mode "controller": upstream's motion, the controller's motion in its own
frame applied to the gripper in its own frame (target = anchor_tcp @
inv(anchor_controller) @ controller), so "forward" follows how the
controller and the gripper point. Differences from upstream:
  - A re-anchors both sides, the controller and the arm's current TCP
    (upstream re-anchored the arm only on B, so A mid-session pulled the arm
    back toward an old pose); A also publishes /quest/engage, which clears an
    ik_guard pause (table floor, angle limits) and resyncs the IK to the arm.
    While A is held the anchor keeps following, so nothing moves until release.
  - B stops publishing targets (the arm holds).
  - /quest/reanchor (from ik_guard, while the arm is held at a soft limit or
    the table floor and the hand pushes on) re-anchors like A, without the
    engage: the hand's overshoot is dropped, so moving back moves the arm.
  - The trigger goes to /quest/gripper (0 = closed, full pull = gripper_max_range m)
    for ik_guard, the driver's only sender.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty

CALIB_MIN_M = 0.08  # each calibration stroke at least this long
CALIB_MIN_ANGLE_DEG = 45.0  # forward and up strokes at least this far from parallel


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


FOLLOW_MIN_RADIUS_M = 0.05  # closer to the base axis than this, the heading doesn't follow


def room_target(anchor_tcp, anchor_hand, hand, r_base_room, scale: float = 1.0,
                follow_base: bool = True) -> np.ndarray:
    """Gripper target (4x4, base frame): the hand's room-frame motion since the
    anchor, rotated into the base frame (r_base_room), applied to anchor_tcp.

    follow_base: the gripper's heading also turns about the base axis by the
    angle the target position swept around it, as a forearm does reaching
    sideways. Holding the heading fixed in the room instead makes the wrist
    counter-twist the whole way: offline, a 20 cm sideways sweep then missed
    the hand by 53 mm median (172 max), or swung J4/J6 by 140 deg with 30 deg
    steps when position was weighted up; following, 4.6 mm (9 max) with 9 deg
    steps (2026-10-15, IK w_pos 20, w_smooth 0.3).
    """
    out = np.eye(4)
    out[:3, 3] = anchor_tcp[:3, 3] + scale * r_base_room @ (hand[:3, 3] - anchor_hand[:3, 3])
    turn = hand[:3, :3] @ anchor_hand[:3, :3].T  # about room axes
    rot = r_base_room @ turn @ r_base_room.T @ anchor_tcp[:3, :3]
    p0, p1 = anchor_tcp[:2, 3], out[:2, 3]
    if follow_base and min(np.hypot(*p0), np.hypot(*p1)) > FOLLOW_MIN_RADIUS_M:
        swept = np.arctan2(p1[1], p1[0]) - np.arctan2(p0[1], p0[0])
        rot = rot_z((swept + np.pi) % (2 * np.pi) - np.pi) @ rot
    out[:3, :3] = rot
    return out


def user_from_strokes(forward, up) -> np.ndarray | None:
    """Rotation room -> user frame (x forward, y left, z up) from a forward and an
    up stroke (room-frame displacements); None if either is short or they are
    too close to parallel."""
    f, u = np.asarray(forward, float), np.asarray(up, float)
    if np.linalg.norm(f) < CALIB_MIN_M or np.linalg.norm(u) < CALIB_MIN_M:
        return None
    f = f / np.linalg.norm(f)
    u = u - (u @ f) * f  # up, made perpendicular to forward
    if np.linalg.norm(u) < np.sin(np.radians(CALIB_MIN_ANGLE_DEG)) * np.linalg.norm(up):
        return None
    u = u / np.linalg.norm(u)
    return np.stack([f, np.cross(u, f), u])  # rows: forward, left, up

sys.path.insert(0, os.path.expanduser("~/QuestArmTeleop/install/oculus_reader/lib/oculus_reader"))
from oculus_reader import OculusReader  # noqa: E402


def pose_to_mat(p) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat([p.orientation.x, p.orientation.y, p.orientation.z,
                                    p.orientation.w]).as_matrix()
    m[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return m


class QuestDelta(Node):
    def __init__(self):
        super().__init__("quest_delta")
        self.declare_parameter("handle_pose_topic", "/right_handle_pose")
        self.declare_parameter("feedback_tcp_pose_topic", "/feedback/tcp_pose")
        self.declare_parameter("delta_pose_topic", "/delta_pose")
        self.declare_parameter("gripper_topic", "/quest/gripper")
        self.declare_parameter("engage_topic", "/quest/engage")
        self.declare_parameter("reanchor_topic", "/quest/reanchor")
        self.declare_parameter("start_button", "A")
        self.declare_parameter("stop_button", "B")
        self.declare_parameter("trigger_axis", "rightTrig")
        self.declare_parameter("gripper_max_range", 0.1)
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("mode", "room")
        self.declare_parameter("forward_yaw_deg", -93.3)  # base-frame yaw of "forward": home's base angle
        self.declare_parameter("scale", 1.0)
        self.declare_parameter("follow_base", True)
        self.declare_parameter("calib_button", "RG")  # right grip
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.start_button, self.stop_button = g("start_button"), g("stop_button")
        self.trigger_axis, self.grip_max = g("trigger_axis"), float(g("gripper_max_range"))
        self.pub_target = self.create_publisher(PoseStamped, g("delta_pose_topic"), 1)
        self.pub_grip = self.create_publisher(JointState, g("gripper_topic"), 1)
        self.pub_engage = self.create_publisher(Empty, g("engage_topic"), 1)
        self.hand = self.tcp = None
        self.create_subscription(PoseStamped, g("handle_pose_topic"), self._on_hand, 1)
        self.create_subscription(PoseStamped, g("feedback_tcp_pose_topic"), self._on_tcp, 1)
        self.create_subscription(Empty, g("reanchor_topic"), self._on_reanchor, 1)
        self.reader = OculusReader()
        time.sleep(0.5)
        self.engaged = self.a_was_down = False
        self.anchor_hand = self.anchor_tcp = None
        self.mode, self.scale = str(g("mode")), float(g("scale"))
        self.follow_base = bool(g("follow_base"))
        self.forward_yaw = np.radians(float(g("forward_yaw_deg")))
        self.calib_button, self.calib_start, self.r_base_room = g("calib_button"), None, None
        self.calib_forward = None  # first stroke, waiting for the up stroke
        self.create_timer(1.0 / float(g("rate_hz")), self._tick)
        self.get_logger().info(f"quest_delta ready ({self.mode} mode): press+release {self.start_button} "
                               f"to engage (re-anchors at the arm's pose), {self.stop_button} to stop")
        if self.mode == "room":
            self.get_logger().warn("CALIBRATE FIRST, right grip held for each stroke: move the "
                                   "controller straight FORWARD ~15 cm, release; then straight UP "
                                   "~15 cm, release. Then A to engage")

    def _on_hand(self, msg):
        self.hand = pose_to_mat(msg.pose)

    def _on_tcp(self, msg):
        self.tcp = pose_to_mat(msg.pose)

    def _on_reanchor(self, msg):
        if self.engaged and self.hand is not None and self.tcp is not None:
            self.anchor_hand, self.anchor_tcp = self.hand.copy(), self.tcp.copy()

    def _tick(self):
        _, buttons = self.reader.get_transformations_and_buttons()
        trig = buttons.get(self.trigger_axis, (0.0,))
        trig = float(trig[0] if isinstance(trig, (list, tuple)) and trig else trig or 0.0)
        grip = JointState()
        grip.header.stamp = self.get_clock().now().to_msg()
        grip.name, grip.position = ["gripper"], [float(np.clip(trig, 0.0, 1.0)) * self.grip_max]
        self.pub_grip.publish(grip)
        if self.hand is None or self.tcp is None:
            return
        if self.mode == "room":
            self._calibrate(bool(buttons.get(self.calib_button, False)))
        a_down = bool(buttons.get(self.start_button, False))
        if a_down and self.mode == "room" and self.r_base_room is None:
            if not self.a_was_down:
                self.get_logger().warn("not calibrated: right grip held, FORWARD ~15 cm, release; "
                                       "then right grip held, UP ~15 cm, release; then A")
            self.a_was_down = a_down
            return
        if a_down:  # anchor both sides; nothing moves while A is held
            self.anchor_hand, self.anchor_tcp = self.hand.copy(), self.tcp.copy()
            if not self.a_was_down:
                self.pub_engage.publish(Empty())
                self.get_logger().info(f"{self.start_button}: engaged, anchored at the arm's pose")
            self.engaged = True
        self.a_was_down = a_down
        if buttons.get(self.stop_button, False) and self.engaged:
            self.engaged = False
            self.get_logger().info(f"{self.stop_button}: stopped")
        if not self.engaged:
            return
        if self.mode == "room":
            target = room_target(self.anchor_tcp, self.anchor_hand, self.hand, self.r_base_room, self.scale,
                                 self.follow_base)
        else:
            target = self.anchor_tcp @ np.linalg.inv(self.anchor_hand) @ self.hand
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = target[:3, 3].tolist()
        q = Rotation.from_matrix(target[:3, :3]).as_quat()
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = q.tolist()
        self.pub_target.publish(msg)

    def _calibrate(self, held: bool) -> None:
        """Grip strokes: the first is forward, the second up; then the frame is set."""
        if held and self.calib_start is None:
            self.calib_start = self.hand[:3, 3].copy()
            if self.engaged:
                self.engaged = False  # don't drive the arm with the calibration stroke
                self.get_logger().info("calibrating: following stopped")
        elif not held and self.calib_start is not None:
            stroke = self.hand[:3, 3] - self.calib_start
            self.calib_start = None
            n = np.linalg.norm(stroke) * 100
            if n < CALIB_MIN_M * 100:
                self.get_logger().warn(f"stroke only {n:.0f} cm (need {CALIB_MIN_M * 100:.0f}); "
                                       f"{'UP' if self.calib_forward is not None else 'FORWARD'} again")
                return
            if self.calib_forward is None:
                self.calib_forward = stroke
                self.get_logger().info(f"forward stroke {n:.0f} cm recorded; now right grip held, "
                                       f"straight UP ~15 cm, release")
                return
            r_user_room = user_from_strokes(self.calib_forward, stroke)
            self.calib_forward = None
            if r_user_room is None:
                self.get_logger().warn("the up stroke was too close to the forward one; start over: "
                                       "FORWARD, then UP")
                return
            self.r_base_room = rot_z(self.forward_yaw) @ r_user_room
            self.get_logger().info(f"calibrated (forward -> base yaw {np.degrees(self.forward_yaw):.0f} "
                                   f"deg). Press A to engage")

    def destroy_node(self):
        self.reader.stop()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = QuestDelta()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
