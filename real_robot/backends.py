"""Hardware backends. All values here are in REAL robot units."""
from __future__ import annotations

import abc

import numpy as np

from .config import PER_ARM, SIM_IMAGE_HW, RobotConfig


class RobotBackend(abc.ABC):
    """Minimal interface the runner needs. Implement one per robot."""

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def get_joint_state(self) -> np.ndarray:
        """(7 * n_arms,) [arm 6, grip] per arm in cfg.arms order, real units."""

    @abc.abstractmethod
    def get_images(self) -> dict[str, np.ndarray]:
        """cfg.camera_names -> HxWx3 uint8 RGB."""

    @abc.abstractmethod
    def command_joints(self, q: np.ndarray) -> None:
        """Send one position target (already safety-filtered)."""

    @abc.abstractmethod
    def hold(self) -> None:
        """Freeze at the current position. Must be safe to call any time."""

    def close(self) -> None:
        pass


class MockBackend(RobotBackend):
    """Software robot: tracks commands with first-order lag, random images."""

    def __init__(self, cfg: RobotConfig, seed: int = 0, lag: float = 0.5):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.lag = lag
        self.q = np.zeros(cfg.state_dim)
        self.commands: list[np.ndarray] = []
        self.held = 0

    def connect(self) -> None:
        pass

    def get_joint_state(self) -> np.ndarray:
        return self.q.copy()

    def get_images(self) -> dict[str, np.ndarray]:
        return {c: self.rng.integers(0, 256, (*SIM_IMAGE_HW, 3), dtype=np.uint8)
                for c in self.cfg.camera_names}

    def command_joints(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        self.commands.append(q.copy())
        self.q = self.q + self.lag * (q - self.q)

    def hold(self) -> None:
        self.held += 1


class OpenCVCameras:
    """Reads the policy cameras via OpenCV; resizes to sim resolution.

    One thread per camera drains the driver's frame queue so read() always
    returns the newest frame. Reading on demand instead returns frames
    queued while the caller was busy (e.g. during inference): on the Piper
    rig, a frame read after a 17 deg base turn still showed the old pose.
    """

    max_age_s = 0.5  # read() fails rather than hand the policy an old frame

    def __init__(self, devices: dict[str, str | int], names):
        missing = [c for c in names if c not in devices]
        if missing:
            raise ValueError(f"config.cameras missing {missing}")
        import threading

        import cv2  # lazy: only the hardware box needs it

        self.cv2 = cv2
        self.caps = {}
        self.latest: dict[str, tuple[float, np.ndarray]] = {}
        self.frame_counts = dict.fromkeys(names, 0)  # frames delivered per camera
        self.lock = threading.Lock()
        self.stop = threading.Event()
        for name in names:
            cap = cv2.VideoCapture(devices[name])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, SIM_IMAGE_HW[1])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SIM_IMAGE_HW[0])
            if not cap.isOpened():
                raise RuntimeError(f"cannot open camera {name} ({devices[name]!r})")
            self.caps[name] = cap
        self.threads = [threading.Thread(target=self._drain, args=(n, c), daemon=True)
                        for n, c in self.caps.items()]
        for t in self.threads:
            t.start()

    def _drain(self, name, cap) -> None:
        import time

        while not self.stop.is_set():
            ok, bgr = cap.read()
            if ok:
                with self.lock:
                    self.latest[name] = (time.monotonic(), bgr)
                    self.frame_counts[name] += 1
            else:
                time.sleep(0.01)

    def read(self) -> dict[str, np.ndarray]:
        import time

        deadline = time.monotonic() + 2.0
        while True:
            with self.lock:
                frames = dict(self.latest)
            now = time.monotonic()
            stale = [n for n in self.caps
                     if n not in frames or now - frames[n][0] > self.max_age_s]
            if not stale:
                break
            if now > deadline:
                raise RuntimeError(f"camera(s) {stale} read failed (no frame newer "
                                   f"than {self.max_age_s} s)")
            time.sleep(0.005)
        out = {}
        for name in self.caps:
            bgr = frames[name][1]
            if bgr.shape[:2] != SIM_IMAGE_HW:
                bgr = self.cv2.resize(bgr, (SIM_IMAGE_HW[1], SIM_IMAGE_HW[0]))
            out[name] = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        return out

    def close(self) -> None:
        self.stop.set()
        for t in self.threads:
            t.join(timeout=1.0)
        for cap in self.caps.values():
            cap.release()


ORBBEC_PREFIX = "orbbec"  # config.cameras value: "orbbec" or "orbbec:<serial>"


def _is_orbbec(device) -> bool:
    return isinstance(device, str) and (device == ORBBEC_PREFIX
                                        or device.startswith(ORBBEC_PREFIX + ":"))


class OrbbecCameras:
    """Color (+ depth when the device offers it) via pyorbbecsdk (SDK v2).

    Same read()/close() contract as OpenCVCameras: read() returns RGB uint8
    at sim resolution. The latest depth frame per camera (float32 mm, native
    resolution) is kept in self.depth for recording; the policy never sees it.
    Needs the Orbbec udev rules (ROBOT_PC_SETUP.md), else opening fails.
    """

    def __init__(self, devices: dict[str, str], names, sdk=None,
                 timeout_ms: int = 1000):
        if sdk is None:
            import pyorbbecsdk as sdk  # lazy: robot PC only
        import cv2

        self.sdk, self.cv2, self.timeout_ms = sdk, cv2, timeout_ms
        found = sdk.Context().query_devices()
        by_serial = {}
        for i in range(found.get_count()):
            dev = found.get_device_by_index(i)
            by_serial[dev.get_device_info().get_serial_number()] = dev
        self.pipes = {}
        self.depth: dict[str, np.ndarray] = {}
        for name in names:
            serial = devices[name].partition(":")[2]
            if not serial:
                if len(by_serial) != 1:
                    raise RuntimeError(f"camera {name}: {len(by_serial)} Orbbec devices, "
                                       f"set 'orbbec:<serial>' (serials: {sorted(by_serial)})")
                serial = next(iter(by_serial))
            if serial not in by_serial:
                raise RuntimeError(f"camera {name}: no Orbbec with serial {serial!r} "
                                   f"(found {sorted(by_serial)})")
            self.pipes[name] = self._start(by_serial[serial])

    def _start(self, dev):
        sdk = self.sdk
        pipe = sdk.Pipeline(dev)
        config = sdk.Config()
        for sensor, required in ((sdk.OBSensorType.COLOR_SENSOR, True),
                                 (sdk.OBSensorType.DEPTH_SENSOR, False)):
            try:
                profiles = pipe.get_stream_profile_list(sensor)
                config.enable_stream(profiles.get_default_video_stream_profile())
            except Exception:
                if required:
                    raise
        pipe.start(config)
        return pipe

    def _to_bgr(self, frame) -> np.ndarray:
        fmt, F = frame.get_format(), self.sdk.OBFormat
        h, w = frame.get_height(), frame.get_width()
        data = np.asanyarray(frame.get_data())
        if fmt == F.MJPG:
            return self.cv2.imdecode(data, self.cv2.IMREAD_COLOR)
        if fmt == F.RGB:
            return self.cv2.cvtColor(data.reshape(h, w, 3), self.cv2.COLOR_RGB2BGR)
        if fmt == F.BGR:
            return data.reshape(h, w, 3)
        if fmt == F.YUYV:
            return self.cv2.cvtColor(data.reshape(h, w, 2), self.cv2.COLOR_YUV2BGR_YUYV)
        raise RuntimeError(f"unsupported Orbbec color format {fmt}")

    def read(self) -> dict[str, np.ndarray]:
        out = {}
        for name, pipe in self.pipes.items():
            frames = pipe.wait_for_frames(self.timeout_ms)
            color = frames.get_color_frame() if frames is not None else None
            if color is None:
                raise RuntimeError(f"camera {name} read failed")
            bgr = self._to_bgr(color)
            if bgr.shape[:2] != SIM_IMAGE_HW:
                bgr = self.cv2.resize(bgr, (SIM_IMAGE_HW[1], SIM_IMAGE_HW[0]))
            out[name] = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
            depth = frames.get_depth_frame()
            if depth is not None:
                raw = np.frombuffer(depth.get_data(), dtype=np.uint16)
                self.depth[name] = (raw.reshape(depth.get_height(), depth.get_width())
                                    .astype(np.float32) * depth.get_depth_scale())
        return out

    def close(self) -> None:
        for pipe in self.pipes.values():
            pipe.stop()


class Cameras:
    """Routes each policy camera to OrbbecCameras or OpenCVCameras by device."""

    def __init__(self, devices: dict[str, str | int], names):
        missing = [c for c in names if c not in devices]
        if missing:
            raise ValueError(f"config.cameras missing {missing}")
        orbbec = [c for c in names if _is_orbbec(devices[c])]
        other = [c for c in names if c not in orbbec]
        self.readers = []
        if orbbec:
            self.readers.append(OrbbecCameras(devices, orbbec))
        if other:
            self.readers.append(OpenCVCameras(devices, other))

    @property
    def frame_counts(self) -> dict[str, int]:
        return {k: v for r in self.readers for k, v in getattr(r, "frame_counts", {}).items()}

    def read(self) -> dict[str, np.ndarray]:
        return {k: v for r in self.readers for k, v in r.read().items()}

    def close(self) -> None:
        for r in self.readers:
            r.close()


class ArxX5Backend(RobotBackend):
    """Two ARX X5 arms. SDK calls are left unimplemented on purpose.

    Fill the four TODOs from the ARX SDK installed on the robot PC (versions
    differ; do not guess the API). Everything else in the pipeline is tested.
    """

    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self.cameras: Cameras | None = None
        self.left = self.right = None

    def connect(self) -> None:
        # TODO(sdk): open both arms (e.g. CAN interfaces for left/right),
        # enable motors in position mode, store handles in self.left/right,
        # then: self.cameras = Cameras(self.cfg.cameras, self.cfg.camera_names)
        raise NotImplementedError("ArxX5Backend.connect: wire up the ARX SDK")

    def get_joint_state(self) -> np.ndarray:
        # TODO(sdk): read 6 joint positions + gripper per arm, return 14-D.
        raise NotImplementedError

    def get_images(self) -> dict[str, np.ndarray]:
        assert self.cameras is not None
        return self.cameras.read()

    def command_joints(self, q: np.ndarray) -> None:
        # TODO(sdk): send q[0:6], q[6] to left; q[7:13], q[13] to right.
        raise NotImplementedError

    def hold(self) -> None:
        # TODO(sdk): command current position (or SDK hold/damping mode).
        raise NotImplementedError

    def close(self) -> None:
        if self.cameras:
            self.cameras.close()


_MDEG_PER_RAD = 1000.0 * 180.0 / np.pi  # piper_sdk joints are 0.001 deg
_UM_PER_M = 1e6  # piper_sdk gripper stroke is 0.001 mm


def _default_piper_factory(can_name: str):
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
        self.sdk_factory = sdk_factory or _default_piper_factory
        self.cameras = cameras
        self.arms: dict = {}

    def connect(self, enable_timeout_s: float = 5.0) -> None:
        import time

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
        joints = [getattr(js, f"joint_{i}") / _MDEG_PER_RAD for i in range(1, 7)]
        grip = arm.GetArmGripperMsgs().gripper_state.grippers_angle / _UM_PER_M
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
        mdeg = [int(round(v * _MDEG_PER_RAD)) for v in q7[:6]]
        arm.MotionCtrl_2(0x01, 0x01, int(self.cfg.arm_speed_percent), 0x00)
        arm.JointCtrl(*mdeg)
        arm.GripperCtrl(abs(int(round(q7[6] * _UM_PER_M))), 1000, 0x01, 0)

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


def make_backend(name: str, cfg: RobotConfig) -> RobotBackend:
    if name == "mock":
        return MockBackend(cfg)
    if name == "arx_x5":
        return ArxX5Backend(cfg)
    if name == "piper":
        return PiperBackend(cfg)
    raise ValueError(f"unknown backend {name!r}")
