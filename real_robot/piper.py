"""AgileX Piper over CAN (piper_sdk): the follower backend and the teaching-mode leader."""
from __future__ import annotations

import time

import numpy as np

from .backends import RobotBackend, ease_to
from .cameras import Cameras
from .config import PER_ARM, RobotConfig
from .safety import SafetyFilter

MDEG_PER_RAD = 1000.0 * 180.0 / np.pi  # piper_sdk joints are 0.001 deg
UM_PER_M = 1e6  # piper_sdk gripper stroke is 0.001 mm

# The folded pose every home() and park() ends in: J2/J3 at the firmware's
# lowest accepted values, J5 at 36.8 deg. This is how the arms park; keep it.
PARK_J2_TO_J6 = np.array([0.0, 0.0, 0.0, 0.642, 0.0])


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

    def home_pose(self) -> np.ndarray:
        """Folded rest pose: J2-J6 at PARK_J2_TO_J6, base at cfg.home_base
        (or where it is), gripper unchanged."""
        q = self.get_joint_state()
        target = q.copy()
        for k in range(len(self.arms)):
            target[PER_ARM * k + 1:PER_ARM * k + 6] = PARK_J2_TO_J6
            if self.cfg.home_base is not None:
                target[PER_ARM * k] = self.cfg.home_base[k]
        return np.clip(target, self.cfg.joint_lower, self.cfg.joint_upper)

    def home(self, hz: float = 25.0, vel: float = 0.2, sleep=time.sleep, log=print) -> None:
        """Fold (base kept), then turn the base to cfg.home_base. Folding
        first keeps an extended arm from sweeping across the table. Leaves
        the motors on, holding."""
        if self.passive:
            raise RuntimeError("PiperBackend is passive; refusing to move")
        safety = SafetyFilter(self.cfg)
        target = self.home_pose()
        folded = target.copy()
        q = self.get_joint_state()
        for k in range(len(self.arms)):
            folded[PER_ARM * k] = q[PER_ARM * k]
        log(f"folding to deg {np.degrees(folded[:6]).round(0)}")
        ease_to(self, safety, folded, hz, vel=vel, sleep=sleep)
        if np.max(np.abs(target - folded)) > np.radians(0.5):
            log(f"turning base to deg {np.degrees(target[:6]).round(0)}")
            ease_to(self, safety, target, hz, vel=vel, sleep=sleep)

    def park(self, hz: float = 25.0, settle_s: float = 3.0, sleep=time.sleep, log=print) -> None:
        """home(), then power down gently instead of dropping.

        Disabling the motors at the fold drops each arm ~1 inch onto its base
        and the wrist to ~37 deg. So after home(): trigger the firmware's
        damped emergency stop (EmergencyStop 0x01: joints sink slowly), wait settle_s,
        disable, then clear the e-stop (0x02) so the next EnablePiper works.
        """
        self.home(hz=hz, sleep=sleep, log=log)
        log("damped stop, then disable")
        for arm in self.arms.values():
            arm.EmergencyStop(0x01)
        sleep(settle_s)
        for arm in self.arms.values():
            arm.DisablePiper()
            arm.EmergencyStop(0x02)

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
