#!/usr/bin/env python3
"""Rate limiter and stop checks between the Quest IK and the Piper driver.

Upstream's arm_ik_pose_node publishes its joint solution straight to the
driver. Its IK can switch arm configuration between frames (tested offline:
18-25 deg jumps per 30 Hz frame, the base swinging ~60 deg for a 3 cm hand
move), and its targets jitter frame to frame, so here the IK publishes to
/ik/joint_states and this node forwards it:

  - commands stream at stream_hz (100) through the critically damped filter of
    real_robot/smoothing.py (cutoff smooth_hz, default 2 Hz: ~0.16 s of lag).
    Sending each IK target as its own move_j made the arm shake: the firmware
    slows down as it reaches each target, as with the policy runs;
  - each command moves at most the config's max_step_delta per control step
    (scaled to stream_hz) from the last one; firmware speed cap on top;
  - an IK target that jumps more than GLITCH_RAD between frames, or a target
    more than LAG_ABORT_RAD from where the arm is, pauses forwarding: the arm
    holds its last command until the targets stop (press B), and the next A
    re-anchors the controller at the arm's pose, so following resumes without
    a jump. (Until 2026-10-13 a stop needed a restart; in practice the IK's
    configuration flips tripped it within seconds.)
  - nothing is sent while IK targets are stale (B pressed, IK failing); when
    they resume, the filter restarts at rest from the arm's feedback.

Workspace limits (2026-10-13; the gripper had touched the table) stop the
arm at the boundary without pausing teleop (since 2026-10-15; before, a limit
paused until A, and pushing into one meant pressing A again and again):
  - joints: the IK target is clipped to soft_lower/soft_upper (inside the
    hardware limits: J5 once overshot its +-70 deg limit by 5 deg when
    commanded right at it), so a joint stops at its limit and the rest follow;
  - table floor: forward kinematics (the IK's URDF, which matches the firmware
    to 0.1 mm) checks the fingertips, gripper base and wrist stay floor_z
    above the table plane (z = 0 in the arm's base frame). A step that would
    go lower is applied joint by joint, keeping only the joints that don't,
    so the gripper slides along the floor and lifts off it freely;
  - no windup: while the arm sits settled at a limit and the hand pushes on,
    this node publishes reanchor_topic (at most REANCHOR_EVERY_S): quest_delta
    re-anchors the controller at the arm's pose and the IK resyncs, so moving
    the hand back moves the arm back at once.

This node is the only sender to the driver's control topic. The driver keeps
just the newest control message (queue depth 1), so when pub_delta_pose sent
its 30 Hz gripper messages there too, each sender overwrote the other's: arm
commands were lost (the arm stalled ~17% of the time while its command moved,
2026-10-13) and most gripper commands too (the gripper felt slow). Now
pub_delta_pose publishes the gripper to gripper_topic, and every command here
carries it; when the arm isn't being driven, gripper-only messages go out at
stream_hz / GRIPPER_ONLY_EVERY.
Runs in the QuestArmTeleop conda env (rclpy); imports only real_robot.config.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from real_robot.config import load_config  # noqa: E402
from real_robot.smoothing import SmoothStreamer  # noqa: E402
from std_msgs.msg import Empty  # noqa: E402

JOINTS = [f"joint{i}" for i in range(1, 7)]
GLITCH_RAD = 0.5     # IK target jump between frames treated as a bad solution (teleop: LEADER_GLITCH_RAD)
LAG_ABORT_RAD = 1.0  # arm this far behind the IK target on any joint: stop (teleop: FOLLOW_LAG_ABORT)
STALE_S = 0.2        # no IK target this recent: send nothing
# Soft joint limits, deg: just inside the hardware's (J1 +-150, J2 0..180,
# J3 -170..0, J4 +-100, J5 +-70, J6 +-120), so they only keep commands off the
# hard stops; the table floor is the workspace limit. J5 keeps 3 deg: commanded
# right at 70 it once overshot by 5. (Until 2026-10-15 they were tighter, J1
# -150..-35 around facing the table, and the operator kept hitting J1 turning
# left, J2 110, J4 +-90 and J5 +-65.)
SOFT_LOWER_DEG = [-148.0, 0.0, -168.0, -98.0, -67.0, -118.0]
SOFT_UPPER_DEG = [148.0, 178.0, 0.0, 98.0, 67.0, 118.0]
# m above z = 0 of the base frame for every checked point. The table is ~15 mm
# below that plane (the mounting plate): at 40 mm the fingertips stopped ~55 mm
# above the table (2026-10-15), so 0 leaves them ~15 mm above it.
FLOOR_Z = 0.0
TCP_OFFSET = 0.13  # m along link6 z: fingertips (as the launch file's tcp_offset)
FLOOR_FRAMES = ["link4", "link5", "link6", "gripper_base", "gripper_link1", "gripper_link2"]
GRIPPER_ONLY_EVERY = 3  # gripper-only messages at stream_hz / 3 (~33 Hz) while the arm is idle
LIMIT_PUSH_RAD = np.radians(2.0)   # IK target this far past a soft limit: the hand is pushing on
SETTLED_RAD = np.radians(1.5)      # arm within this of the last command: it has stopped there
REANCHOR_EVERY_S = 0.25


def _by_name(msg: JointState) -> np.ndarray | None:
    pos = dict(zip(msg.name, msg.position))
    if not all(j in pos for j in JOINTS):
        return None
    q = np.array([pos[j] for j in JOINTS], dtype=np.float64)
    return q if np.all(np.isfinite(q)) else None


class IkGuard(Node):
    def __init__(self):
        super().__init__("ik_guard")
        self.declare_parameter("config", "real_robot/configs/piper_1.json")
        self.declare_parameter("ik_topic", "/ik/joint_states")
        self.declare_parameter("feedback_topic", "/feedback/joint_states")
        self.declare_parameter("control_topic", "/control/joint_states")
        self.declare_parameter("gripper_topic", "/quest/gripper")
        self.declare_parameter("engage_topic", "/quest/engage")
        self.declare_parameter("reanchor_topic", "/quest/reanchor")
        self.declare_parameter("soft_lower_deg", SOFT_LOWER_DEG)
        self.declare_parameter("soft_upper_deg", SOFT_UPPER_DEG)
        self.declare_parameter("floor_z", FLOOR_Z)
        self.declare_parameter("smooth_hz", 2.0)
        self.declare_parameter("stream_hz", 100.0)
        cfg = load_config(self.get_parameter("config").value)
        smooth_hz = float(self.get_parameter("smooth_hz").value)
        self.stream_hz = float(self.get_parameter("stream_hz").value)
        self.lo = np.asarray(cfg.joint_lower[:6])
        self.hi = np.asarray(cfg.joint_upper[:6])
        self.soft_lo = np.radians(self.get_parameter("soft_lower_deg").value)
        self.soft_hi = np.radians(self.get_parameter("soft_upper_deg").value)
        self.floor_z = float(self.get_parameter("floor_z").value)
        self._init_fk()
        self.max_delta = np.asarray(cfg.max_step_delta[:6]) * cfg.control_hz / self.stream_hz
        self.filter = SmoothStreamer(send=None, lower=self.lo, upper=self.hi,
                                     cutoff_hz=smooth_hz, rate_hz=self.stream_hz)
        self.target = self.prev_target = self.cur = self.last_cmd = None
        self.target_t = 0.0
        self.grip_lo, self.grip_hi = cfg.joint_lower[6], cfg.joint_upper[6]
        self.gripper = None
        self.ticks = 0
        self.paused = None
        self.steps = 0
        self.raw_target = None
        self.limited, self.limited_t = None, 0.0  # last boundary logged, and when
        self.floor_blocked = False
        self.last_reanchor = 0.0
        self.clip_lo = np.maximum(self.lo, self.soft_lo)
        self.clip_hi = np.minimum(self.hi, self.soft_hi)
        self.pub_reanchor = self.create_publisher(Empty, self.get_parameter("reanchor_topic").value, 1)
        self.pub = self.create_publisher(JointState, self.get_parameter("control_topic").value, 1)
        self.create_subscription(JointState, self.get_parameter("ik_topic").value, self._on_ik, 1)
        self.create_subscription(JointState, self.get_parameter("feedback_topic").value,
                                 self._on_feedback, 1)
        self.create_subscription(JointState, self.get_parameter("gripper_topic").value,
                                 self._on_gripper, 1)
        self.create_subscription(Empty, self.get_parameter("engage_topic").value, self._on_engage, 1)
        self.create_timer(1.0 / self.stream_hz, self._step)
        self.get_logger().info(
            f"ik_guard: {self.stream_hz:.0f} Hz through a {smooth_hz:g} Hz filter, max deg/s "
            f"{np.degrees(self.max_delta * self.stream_hz).round(0)}, "
            f"pause on IK jump > {np.degrees(GLITCH_RAD):.0f} deg or lag > {np.degrees(LAG_ABORT_RAD):.0f} deg; "
            f"stops (no pause) at soft limits deg {np.degrees(self.soft_lo).round(0)}.."
            f"{np.degrees(self.soft_hi).round(0)} and floor {self.floor_z * 1000:.0f} mm")

    def _init_fk(self) -> None:
        import pinocchio as pin
        from ament_index_python.packages import get_package_share_directory
        pkg = get_package_share_directory("agx_arm_description")
        self.pin = pin
        self.model = pin.buildModelFromUrdf(
            os.path.join(pkg, "agx_arm_urdf/piper/urdf/piper_with_gripper_description.urdf"))
        self.data = self.model.createData()
        self.fk_frames = [self.model.getFrameId(n) for n in FLOOR_FRAMES]
        self.link6 = self.model.getFrameId("link6")

    def lowest_point(self, q) -> tuple[float, str]:
        """Lowest checked point (z in m, name) with the arm joints at q."""
        full = np.zeros(self.model.nq)
        full[:6] = q
        self.pin.framesForwardKinematics(self.model, self.data, full)
        pts = [(self.data.oMf[f].translation[2], n) for f, n in zip(self.fk_frames, FLOOR_FRAMES)]
        pts.append((self.data.oMf[self.link6].act(np.array([0.0, 0.0, TCP_OFFSET]))[2], "fingertips"))
        return min(pts)

    def below_floor(self, q, ref) -> bool:
        """q puts a checked point under the floor, lower than at ref (an arm already
        under it may always move up)."""
        z, _ = self.lowest_point(q)
        return z < self.floor_z and z < self.lowest_point(ref)[0] - 1e-4

    def floor_step(self, cmd, ref) -> np.ndarray:
        """The step ref -> cmd without what would go under the floor: the whole step if
        allowed, else each joint's part in turn (largest first), kept if allowed."""
        if not self.below_floor(cmd, ref):
            return cmd
        out = ref.copy()
        for j in np.argsort(-np.abs(cmd - ref)):
            trial = out.copy()
            trial[j] = cmd[j]
            if not self.below_floor(trial, out):
                out = trial
        return out

    def _on_engage(self, msg) -> None:
        self.last_cmd = None  # restart the filter from the arm's feedback
        self.prev_target = None
        self.limited = None
        self.target = None  # wait for a fresh IK target: the stored one is from before A
        if self.paused:
            self.get_logger().info(f"A pressed: pause cleared ({self.paused})")
            self.paused = None

    def _pause(self, why: str) -> None:
        if self.paused is None:
            self.paused = why
            self.get_logger().error(f"PAUSED: {why}. The arm holds; press A to "
                                    f"follow again from the arm's pose")

    def _on_feedback(self, msg: JointState) -> None:
        q = _by_name(msg)
        if q is not None:
            self.cur = q

    def _on_gripper(self, msg: JointState) -> None:
        pos = dict(zip(msg.name, msg.position))
        if "gripper" in pos and np.isfinite(pos["gripper"]):
            self.gripper = float(np.clip(pos["gripper"], self.grip_lo, self.grip_hi))

    def _send(self, arm) -> None:
        names, pos = [], []
        if arm is not None:
            names, pos = list(JOINTS), arm.tolist()
        if self.gripper is not None:
            names.append("gripper")
            pos.append(self.gripper)
        if names:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name, msg.position = names, pos
            self.pub.publish(msg)

    def _on_ik(self, msg: JointState) -> None:
        q = _by_name(msg)
        if q is None:
            return
        if self.prev_target is not None and time.monotonic() - self.target_t < STALE_S:
            jump = np.abs(q - self.prev_target)
            if np.max(jump) > GLITCH_RAD:
                self._pause(f"IK target for joint {int(np.argmax(jump)) + 1} jumped "
                            f"{np.degrees(np.max(jump)):.0f} deg in one frame")
        self.prev_target = q
        self.raw_target = q
        self.target, self.target_t = np.clip(q, self.clip_lo, self.clip_hi), time.monotonic()

    def _step(self) -> None:
        self.ticks += 1
        if not self._step_arm() and self.ticks % GRIPPER_ONLY_EVERY == 0:
            self._send(None)

    def _step_arm(self) -> bool:
        """Send one arm command (with the gripper); False when the arm isn't driven."""
        if self.cur is None or self.target is None:
            return False
        if time.monotonic() - self.target_t > STALE_S:
            self.last_cmd = None  # resume from the arm's feedback, at rest
            self.prev_target = None
            if self.paused:
                self.paused = None
                self.get_logger().info("targets stopped (B): pause cleared; press A to follow again")
            return False
        if self.paused:
            return False
        if self.last_cmd is None:
            self.filter.reset(self.cur)
            self.last_cmd = self.cur.copy()
        # Blocked arm: what we last sent vs where it is (a target held off by a
        # limit or the floor is expected to be far from the arm).
        gaps = np.abs(self.last_cmd - self.cur)
        if np.max(gaps) > LAG_ABORT_RAD:
            self._pause(f"arm joint {int(np.argmax(gaps)) + 1} is {np.max(gaps):.2f} rad from its command")
            return False
        self.filter.set_target(self.target)
        smooth = self.filter.step()
        cmd = self.last_cmd + np.clip(smooth - self.last_cmd, -self.max_delta, self.max_delta)
        floored = self.floor_step(cmd, self.last_cmd)
        self.floor_blocked = not np.array_equal(floored, cmd)
        cmd = floored
        self._note_limits()
        if not np.array_equal(cmd, smooth):  # rate-limited or floored: keep the filter from winding up
            with self.filter.lock:
                self.filter.y = cmd.copy()
                self.filter.v = (cmd - self.last_cmd) * self.stream_hz
        self.last_cmd = cmd
        self._send(cmd)
        self.steps += 1
        if self.steps % int(self.stream_hz * 5) == 0:
            self.get_logger().info(f"target deg {np.degrees(self.target).round(0)} "
                                   f"lag {np.degrees(np.max(gaps)):.1f} deg")
        return True

    def _note_limits(self) -> None:
        """Log entering/leaving a boundary; re-anchor the hand while it pushes past one."""
        past = np.maximum(self.clip_lo - self.raw_target, self.raw_target - self.clip_hi)
        why = None
        if np.max(past) > LIMIT_PUSH_RAD:
            j = int(np.argmax(past))
            lim = self.clip_lo[j] if self.raw_target[j] < self.clip_lo[j] else self.clip_hi[j]
            why = f"joint {j + 1} at its limit ({np.degrees(lim):.0f} deg)"
        elif self.floor_blocked:
            why = f"at the table floor ({self.floor_z * 1000:.0f} mm)"
        now = time.monotonic()
        if why and (why != self.limited or now - self.limited_t > 3.0):  # re-anchoring toggles it
            self.get_logger().warn(f"holding: {why}; the other directions still follow")
            self.limited, self.limited_t = why, now
        settled = np.max(np.abs(self.last_cmd - self.cur)) < SETTLED_RAD
        if why and settled and now - self.last_reanchor > REANCHOR_EVERY_S:
            self.last_reanchor = now
            self.prev_target = None  # the IK resyncs to the arm: its next target may jump back
            self.pub_reanchor.publish(Empty())


def main(args=None):
    rclpy.init(args=args)
    node = IkGuard()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
