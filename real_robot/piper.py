"""AgileX Piper over CAN (piper_sdk): the follower backend and the teaching-mode leader."""
from __future__ import annotations

import time

import numpy as np

from .backends import RobotBackend
from .cameras import Cameras
from .config import PER_ARM, RobotConfig

MDEG_PER_RAD = 1000.0 * 180.0 / np.pi  # piper_sdk joints are 0.001 deg
UM_PER_M = 1e6  # piper_sdk gripper stroke is 0.001 mm


def piper_sdk_interface(can_name: str):
    from piper_sdk import C_PiperInterface_V2  # lazy: robot PC only

    return C_PiperInterface_V2(can_name)


class PiperBackend(RobotBackend):
    """AgileX Piper arm(s) over CAN via piper_sdk (C_PiperInterface_V2).

    One entry in cfg.arms per arm; a single Piper robot is one arm.

    Real units: joints in rad, gripper stroke in m. API per piper_sdk
    demo/V2 (piper_ctrl_joint.py, piper_read_joint_state.py); verify against
    the installed version (`pip show piper_sdk`) before a live run.
    """

    def __init__(self, cfg: RobotConfig, sdk_factory=None, cameras=None,
                 passive: bool = False):
        # passive: read-only (demo recording while the arms are hand-guided
        # in teach mode); never enables motors and refuses commands.
        self.cfg = cfg
        self.passive = passive
        self.sdk_factory = sdk_factory or piper_sdk_interface
        self.cameras = cameras
        self.arms: dict = {}

    def connect(self, enable_timeout_s: float = 5.0) -> None:
        for side, can in self.cfg.arms.items():
            arm = self.sdk_factory(can)
            arm.ConnectPort()
            self.arms[side] = arm
            if self.passive:
                continue
            deadline = time.monotonic() + enable_timeout_s
            while not arm.EnablePiper():
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{side} Piper did not enable on "
                                       f"{can}")
                time.sleep(0.01)
        if self.cameras is None:
            self.cameras = Cameras(self.cfg.cameras, self.cfg.camera_names)

    def _read_arm(self, arm) -> list[float]:
        js = arm.GetArmJointMsgs().joint_state
        joints = [getattr(js, f"joint_{i}") / MDEG_PER_RAD for i in range(1, 7)]
        grip = arm.GetArmGripperMsgs().gripper_state.grippers_angle / UM_PER_M
        return joints + [grip]

    def get_joint_state(self) -> np.ndarray:
        return np.asarray([v for arm in self.arms.values() for v in self._read_arm(arm)],
                          dtype=np.float64)

    def get_images(self) -> dict[str, np.ndarray]:
        return self.cameras.read()

    def get_end_pose(self) -> np.ndarray:
        """(6 * n_arms,) [x, y, z m, roll, pitch, yaw rad] per arm, from the
        firmware's forward kinematics (piper_sdk: 0.001 mm, 0.001 deg)."""
        out = []
        for arm in self.arms.values():
            e = arm.GetArmEndPoseMsgs().end_pose
            out += [e.X_axis / 1e6, e.Y_axis / 1e6, e.Z_axis / 1e6,
                    *np.radians([e.RX_axis / 1e3, e.RY_axis / 1e3, e.RZ_axis / 1e3])]
        return np.asarray(out, dtype=np.float64)

    def _send_arm(self, arm, q7) -> None:
        mdeg = [int(round(v * MDEG_PER_RAD)) for v in q7[:6]]
        arm.MotionCtrl_2(0x01, 0x01, int(self.cfg.arm_speed_percent), 0x00)
        arm.JointCtrl(*mdeg)
        arm.GripperCtrl(abs(int(round(q7[6] * UM_PER_M))), 1000, 0x01, 0)

    def command_joints(self, q: np.ndarray) -> None:
        if self.passive:
            raise RuntimeError("PiperBackend is passive (recording); refusing to command")
        q = np.asarray(q, dtype=np.float64)
        for k, arm in enumerate(self.arms.values()):
            self._send_arm(arm, q[PER_ARM * k:PER_ARM * (k + 1)])

    def hold(self) -> None:
        # Re-command the measured pose; the arms stay enabled and stiff.
        if self.passive:
            return
        for arm in self.arms.values():
            try:
                self._send_arm(arm, self._read_arm(arm))
            except Exception:  # never let hold() raise during cleanup
                pass

    def close(self) -> None:
        if self.cameras:
            self.cameras.close()


class PiperLeader:
    """A Piper in teaching mode (MasterSlaveConfig 0xFA): limp, moved by hand,
    broadcasting its joint targets on the bus."""

    def __init__(self, can: str, sdk_factory=None):
        self.arm = (sdk_factory or piper_sdk_interface)(can)

    def connect(self) -> None:
        self.arm.ConnectPort()
        self.arm.MasterSlaveConfig(0xFA, 0, 0, 0)  # teaching input arm
        time.sleep(0.3)

    def wait_for_broadcast(self, timeout_s: float = 30.0, settle_s: float = 0.2,
                           tol_rad: float = 0.02, sleep=time.sleep) -> None:
        """Block until the leader's broadcast is complete and steady.

        A teaching arm broadcasts only while it is being moved, and its six
        joints arrive in three CAN frames (0x155-0x157): until each has been
        seen the SDK reports 0 for those joints. So wait for the first frame,
        then for readings that stay within tol_rad for settle_s.
        """
        deadline = time.monotonic() + timeout_s
        ts0 = self.arm.GetArmJointCtrl().time_stamp
        while self.arm.GetArmJointCtrl().time_stamp == ts0:
            if time.monotonic() > deadline:
                raise RuntimeError("leader sent nothing: move it a little by hand "
                                   "(is it in teaching mode and powered?)")
            sleep(0.02)
        ref, steady_since = self.read(), time.monotonic()
        while time.monotonic() - steady_since < settle_s:
            if time.monotonic() > deadline:
                raise RuntimeError("leader readings never settled; hold it still for a moment")
            sleep(0.02)
            q = self.read()
            if np.max(np.abs(q[:6] - ref[:6])) > tol_rad:
                ref, steady_since = q, time.monotonic()

    def restore_follower_mode(self) -> None:
        self.arm.MasterSlaveConfig(0xFC, 0, 0, 0)

    def read(self) -> np.ndarray:
        """(7,) joint targets the leader broadcasts: 6 joints rad + gripper m."""
        j = self.arm.GetArmJointCtrl().joint_ctrl
        g = self.arm.GetArmGripperCtrl().gripper_ctrl
        return np.asarray([getattr(j, f"joint_{i}") / MDEG_PER_RAD for i in range(1, 7)]
                          + [abs(g.grippers_angle) / UM_PER_M], dtype=np.float64)

    def read_feedback(self) -> np.ndarray:
        """(7,) measured joints (rad) + gripper (m), for comparison."""
        j = self.arm.GetArmJointMsgs().joint_state
        g = self.arm.GetArmGripperMsgs().gripper_state
        return np.asarray([getattr(j, f"joint_{i}") / MDEG_PER_RAD for i in range(1, 7)]
                          + [g.grippers_angle / UM_PER_M], dtype=np.float64)
