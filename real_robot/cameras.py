"""Policy cameras: OpenCV (UVC) and Orbbec SDK readers, RGB uint8 at sim resolution."""
from __future__ import annotations

import numpy as np

from .config import SIM_IMAGE_HW


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


def is_orbbec(device) -> bool:
    return isinstance(device, str) and (device == ORBBEC_PREFIX
                                        or device.startswith(ORBBEC_PREFIX + ":"))


class OrbbecCameras:
    """Color (+ depth when the device offers it) via pyorbbecsdk (SDK v2).

    Same read()/close() contract as OpenCVCameras: read() returns RGB uint8
    at sim resolution. The latest depth frame per camera (float32 mm, native
    resolution) is kept in self.depth for recording; the policy never sees it.
    Needs the Orbbec udev rules (docs/robot.md), else opening fails.
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
        orbbec = [c for c in names if is_orbbec(devices[c])]
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
