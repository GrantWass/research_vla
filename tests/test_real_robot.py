"""Real-robot bridge tests: mock robot + fake policy, no GPU/server/hardware."""
import json
import os
import time
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

if np is not None:
    from real_robot.backends import MockBackend
    from real_robot.piper import PiperBackend
    from real_robot.config import RobotConfig, load_config
    from real_robot.mapping import real_to_sim, sim_to_real
    from real_robot.runner import build_obs, chunk_to_array, preflight, run
    from real_robot.safety import SafetyAbort, SafetyFilter

REPO = os.path.join(os.path.dirname(__file__), "..")


class FakePolicy:
    """Mimics WsModelClient.call for reset/update_obs/get_action."""

    def __init__(self, chunk_fn, chunk_len=10):
        self.chunk_fn = chunk_fn
        self.chunk_len = chunk_len
        self.obs = []
        self.resets = 0

    def call(self, func_name=None, obs=None):
        if func_name == "reset":
            self.resets += 1
        elif func_name == "update_obs":
            self.obs.append(obs)
        elif func_name == "get_action":
            state = np.asarray(self.obs[-1]["state"], dtype=np.float64)
            return self.chunk_fn(state, self.chunk_len)
        return None


def drift(state, n, step=0.01):
    return [state + step * (i + 1) for i in range(n)]


@unittest.skipIf(np is None, "numpy not installed")
class TestRealRobot(unittest.TestCase):
    def setUp(self):
        self.cfg = RobotConfig(verified=True)

    def test_config_extends_base(self):
        cfg = load_config(os.path.join(REPO, "real_robot/configs/piper_1_banana.json"))
        self.assertEqual(cfg.arms, {"arm": "can_piper1"})  # from piper_1.json
        self.assertEqual(cfg.joint_upper[6], 0.07)  # from piper.json
        self.assertEqual(cfg.control_hz, 30.0)  # its own override

    def test_bad_config_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"joint_lower": [0.0] * 3}, f)
        with self.assertRaises(ValueError):
            load_config(f.name)

    def test_mapping_roundtrip(self):
        cfg = RobotConfig(joint_sign=[-1.0] + [1.0] * 6, joint_offset=[0.2] * 7,
                          gripper_sim_range=[0, 1], gripper_real_range=[0, 0.08])
        q = np.linspace(-1, 1, 7)
        q[6] = 0.04
        sim = real_to_sim(q, cfg)
        self.assertAlmostEqual(sim[6], 0.5)
        self.assertAlmostEqual(sim[0], -q[0] + 0.2)
        np.testing.assert_allclose(sim_to_real(sim, cfg), q)

    def test_safety_rate_limits_and_clips(self):
        cfg = RobotConfig(max_step_delta=[0.1] * 7, joint_upper=[0.15] * 7,
                          joint_lower=[-1.0] * 7, abort_delta=1.0)
        sf = SafetyFilter(cfg)
        cmd = sf.filter(np.full(7, 0.5), np.zeros(7))
        np.testing.assert_allclose(cmd, 0.1)
        cmd = sf.filter(np.full(7, 0.5), cmd)
        np.testing.assert_allclose(cmd, 0.15)  # limit clip
        self.assertGreaterEqual(sf.stats.rate_clips, 1)
        self.assertGreaterEqual(sf.stats.limit_clips, 1)

    def test_safety_aborts_on_jump_and_nan(self):
        sf = SafetyFilter(RobotConfig(abort_delta=0.5))
        with self.assertRaises(SafetyAbort):
            sf.filter(np.full(7, 0.6), np.zeros(7))
        bad = np.zeros(7)
        bad[3] = np.nan
        with self.assertRaises(SafetyAbort):
            sf.filter(bad, np.zeros(7))

    def test_obs_format_matches_pi05_encode(self):
        robot = MockBackend(self.cfg)
        obs = build_obs(robot.get_images(), np.zeros(7), "stack the bowls")
        self.assertEqual(set(obs["images"]), {"cam_high", "cam_wrist"})
        self.assertEqual(obs["state"].dtype, np.float32)
        self.assertEqual(obs["images"]["cam_high"].shape, (480, 640, 3))
        with self.assertRaises(ValueError):
            build_obs({**robot.get_images(), "cam_high": np.zeros((4, 4))},
                      np.zeros(7), "x")

    def test_chunk_dict_form(self):
        row = {"left_arm_joint_state": [1] * 6, "left_ee_joint_state": [2],
               "right_arm_joint_state": [3] * 6, "right_ee_joint_state": [4]}
        arr = chunk_to_array([row, row], 14)
        self.assertEqual(arr.shape, (2, 14))
        self.assertEqual(arr[0, 6], 2)
        self.assertEqual(arr[0, 13], 4)

    def test_chunk_padding_trimmed(self):
        arr = chunk_to_array(np.zeros((50, 32)), 7)  # openpi pads to 32
        self.assertEqual(arr.shape, (50, 7))
        with self.assertRaises(ValueError):
            chunk_to_array(np.zeros((50, 6)), 7)

    def test_dry_run_never_commands(self):
        robot = MockBackend(self.cfg)
        with tempfile.TemporaryDirectory() as d:
            s = run(robot, FakePolicy(drift), self.cfg, instruction="x",
                    live=False, max_steps=25, log_dir=d)
            self.assertTrue(os.path.exists(os.path.join(d, "trajectory.npz")))
            self.assertTrue(os.path.exists(os.path.join(d, "summary.json")))
        self.assertEqual(s["status"], "ok")
        self.assertEqual(s["steps"], 25)
        self.assertEqual(robot.commands, [])
        self.assertEqual(robot.held, 0)

    def test_live_run_moves_and_holds(self):
        robot = MockBackend(self.cfg, lag=1.0)
        s = run(robot, FakePolicy(drift, chunk_len=5), self.cfg, instruction="x",
                live=True, speed=1.0, max_steps=12, execute_steps=3)
        self.assertEqual(s["status"], "ok")
        self.assertEqual(len(robot.commands), 12)
        self.assertEqual(s["chunks"], 4)
        self.assertEqual(robot.held, 1)
        self.assertGreater(robot.q[0], 0)

    def test_live_run_aborts_and_holds_on_bad_policy(self):
        robot = MockBackend(self.cfg)
        jump = lambda st, n: [st + 2.0] * n  # noqa: E731
        s = run(robot, FakePolicy(jump), self.cfg, instruction="x",
                live=True, speed=1.0, max_steps=10)
        self.assertTrue(s["status"].startswith("safety_abort"))
        self.assertEqual(robot.commands, [])
        self.assertEqual(robot.held, 1)

    def test_async_runs_without_pausing_and_skips_elapsed_actions(self):
        cfg = RobotConfig(verified=True, control_hz=100.0)  # 10 ms steps
        infer_s = 0.05  # 5 steps of latency

        class Slow(FakePolicy):
            def call(self, func_name=None, obs=None):
                if func_name == "get_action":
                    time.sleep(infer_s)
                return super().call(func_name, obs)

        robot = MockBackend(cfg, lag=1.0)
        policy = Slow(lambda st, n: drift(st, n, step=0.001), chunk_len=30)
        t0 = time.perf_counter()
        s = run(robot, policy, cfg, instruction="x", live=True, speed=1.0,
                max_steps=60, execute_steps=15, async_infer=True)
        wall = time.perf_counter() - t0
        self.assertEqual(s["status"], "ok")
        self.assertEqual(len(robot.commands), 60)
        self.assertEqual(robot.held, 1)
        self.assertEqual(s["starved_steps"], 0)
        self.assertGreater(s["chunks"], 2)
        # each switch skips the actions that elapsed while the chunk was inferred
        self.assertGreaterEqual(s["skipped_actions"], 3 * (s["chunks"] - 1))
        # async: one up-front inference + 60 steps (0.65 s); sync would add a
        # pause per chunk (0.6 + 4 * 0.05 = 0.8 s)
        self.assertLess(wall, 0.6 + infer_s + 0.07)

    def test_async_inference_error_holds(self):
        class FailsSecond(FakePolicy):
            def call(self, func_name=None, obs=None):
                if func_name == "get_action" and len(self.obs) > 1:
                    raise TimeoutError("server gone")
                return super().call(func_name, obs)

        robot = MockBackend(self.cfg)
        s = run(robot, FailsSecond(drift, chunk_len=10), self.cfg, instruction="x",
                live=True, speed=1.0, max_steps=50, execute_steps=5, async_infer=True)
        self.assertIn("TimeoutError", s["status"])
        self.assertEqual(robot.held, 1)

    def test_server_error_holds(self):
        class Dead:
            def call(self, func_name=None, obs=None):
                if func_name == "get_action":
                    raise TimeoutError("server gone")

        robot = MockBackend(self.cfg)
        s = run(robot, Dead(), self.cfg, instruction="x", live=True, speed=1.0)
        self.assertIn("TimeoutError", s["status"])
        self.assertEqual(robot.held, 1)

    def test_preflight_flags_unverified_config(self):
        cfg = RobotConfig()
        probs = preflight(MockBackend(cfg), cfg, SafetyFilter(cfg))
        self.assertTrue(any("verified" in p for p in probs))
        self.assertEqual(preflight(MockBackend(self.cfg), self.cfg,
                                   SafetyFilter(self.cfg)), [])


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakePiper:
    """Records calls; feedback mirrors the last command (piper_sdk units)."""

    def __init__(self, can_name):
        self.can_name = can_name
        self.joints = [0] * 6  # 0.001 deg
        self.grip = 0  # 0.001 mm
        self.calls = []
        self.enable_tries = 0

    def ConnectPort(self):
        self.calls.append("ConnectPort")

    def EnablePiper(self):
        self.enable_tries += 1
        return self.enable_tries >= 2

    def GetArmJointMsgs(self):
        return _NS(joint_state=_NS(**{f"joint_{i + 1}": v for i, v in enumerate(self.joints)}))

    def GetArmGripperMsgs(self):
        return _NS(gripper_state=_NS(grippers_angle=self.grip))

    def MotionCtrl_2(self, *a):
        self.calls.append(("MotionCtrl_2", a))

    def JointCtrl(self, *j):
        self.joints = list(j)

    def GripperCtrl(self, pos, *a):
        self.grip = pos

    def EmergencyStop(self, code):
        self.calls.append(("EmergencyStop", code))

    def DisablePiper(self):
        self.calls.append("DisablePiper")


class FakeCams:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last = None

    def read(self):
        self.last = MockBackend(self.cfg).get_images()
        return self.last

    def close(self):
        pass


@unittest.skipIf(np is None, "numpy not installed")
class TestPiper(unittest.TestCase):
    def make(self, n=1, **kw):
        cfg = load_config(os.path.join(REPO, f"real_robot/configs/piper_{n}.json"))
        made = {}

        def factory(can):
            made[can] = FakePiper(can)
            return made[can]
        b = PiperBackend(cfg, sdk_factory=factory, cameras=FakeCams(cfg), **kw)
        b.connect()
        return b, made

    def test_park_folds_then_stops_damped_and_disables(self):
        b, made = self.make()
        arm = made["can_piper1"]
        arm.joints = [-90000, 10000, -10000, 0, 0, 5000]  # 0.001 deg
        b.park(sleep=lambda s: None, log=lambda *a: None)
        deg = np.asarray(arm.joints) / 1000
        np.testing.assert_allclose(deg, [-90, 0, 0, 0, 36.8, 0], atol=0.1)  # base kept
        tail = [c for c in arm.calls if c == "DisablePiper" or c[0] == "EmergencyStop"]
        self.assertEqual(tail, [("EmergencyStop", 1), "DisablePiper", ("EmergencyStop", 2)])

    def test_single_arm_config(self):
        b, made = self.make()
        self.assertEqual(b.cfg.state_dim, 7)
        self.assertEqual(b.cfg.gripper_idx, [6])
        self.assertEqual(set(made), {"can_piper1"})
        b2, made2 = self.make(2)
        self.assertEqual(set(made2), {"can_piper2"})

    def test_units_roundtrip(self):
        b, made = self.make()
        q = np.array([0.1, 1.0, -0.5, 0.2, -0.3, 0.4, 0.035])
        b.command_joints(q)
        self.assertEqual(made["can_piper1"].joints[0], round(0.1 * 180e3 / np.pi))
        self.assertEqual(made["can_piper1"].grip, 35000)  # 0.035 m -> 0.001 mm
        np.testing.assert_allclose(b.get_joint_state(), q, atol=1e-5)
        self.assertEqual(made["can_piper1"].calls[1], ("MotionCtrl_2", (1, 1, 30, 0)))

    def test_two_arm_layout(self):
        cfg = RobotConfig(arms={"a": "canA", "b": "canB"})
        made = {}

        def factory(can):
            made[can] = FakePiper(can)
            return made[can]
        b = PiperBackend(cfg, sdk_factory=factory, cameras=FakeCams(cfg))
        b.connect()
        q = np.zeros(14)
        q[7] = 0.2
        b.command_joints(q)
        self.assertEqual(made["canA"].joints[0], 0)
        self.assertNotEqual(made["canB"].joints[0], 0)

    def test_passive_never_enables_or_commands(self):
        b, made = self.make(passive=True)
        self.assertEqual(made["can_piper1"].enable_tries, 0)
        with self.assertRaises(RuntimeError):
            b.command_joints(np.zeros(7))
        b.hold()  # no-op, must not raise

    def test_runner_dry_run_on_piper(self):
        b, made = self.make()
        b.cfg.verified = True
        s = run(b, FakePolicy(drift), b.cfg, instruction="x", live=False, max_steps=5)
        self.assertEqual(s["status"], "ok")
        self.assertEqual(made["can_piper1"].joints, [0] * 6)  # nothing sent
        self.assertEqual(set(b.cameras.last), {"cam_high", "cam_wrist"})


class FakeOrbbecSDK:
    """Minimal pyorbbecsdk v2 surface: devices by serial, RGB color + depth."""

    class OBSensorType:
        COLOR_SENSOR, DEPTH_SENSOR = "color", "depth"

    class OBFormat:
        RGB, BGR, YUYV, MJPG = "rgb", "bgr", "yuyv", "mjpg"

    def __init__(self, serials=("AB1",), depth=True):
        self.serials, self.has_depth = list(serials), depth
        self.started, self.stopped = [], []
        sdk = self

        class Frame:
            def __init__(self, fmt, h, w, data):
                self.fmt, self.h, self.w, self.data = fmt, h, w, data

            def get_format(self): return self.fmt
            def get_height(self): return self.h
            def get_width(self): return self.w
            def get_data(self): return self.data
            def get_depth_scale(self): return 1.0

        class Frames:
            def get_color_frame(self):
                rgb = np.zeros((48, 64, 3), np.uint8)
                rgb[..., 0] = 200  # pure red, checks RGB order end to end
                return Frame(sdk.OBFormat.RGB, 48, 64, rgb.reshape(-1))

            def get_depth_frame(self):
                if not sdk.has_depth:
                    return None
                d = np.full((4, 5), 700, np.uint16)
                return Frame(None, 4, 5, d.tobytes())

        class Profiles:
            def __init__(self, sensor): self.sensor = sensor
            def get_default_video_stream_profile(self): return self.sensor

        class Pipeline:
            def __init__(self, dev): self.dev = dev

            def get_stream_profile_list(self, sensor):
                if sensor == sdk.OBSensorType.DEPTH_SENSOR and not sdk.has_depth:
                    raise RuntimeError("no depth sensor")
                return Profiles(sensor)

            def start(self, config): sdk.started.append((self.dev, config.streams))
            def stop(self): sdk.stopped.append(self.dev)
            def wait_for_frames(self, timeout_ms): return Frames()

        class Config:
            def __init__(self): self.streams = []
            def enable_stream(self, p): self.streams.append(p)

        class Devices:
            def get_count(self): return len(sdk.serials)

            def get_device_by_index(self, i):
                info = _NS(get_serial_number=lambda: sdk.serials[i])
                return _NS(serial=sdk.serials[i], get_device_info=lambda: info)

        self.Pipeline, self.Config = Pipeline, Config
        self.Context = lambda: _NS(query_devices=Devices)


@unittest.skipIf(np is None, "numpy not installed")
class TestOrbbec(unittest.TestCase):
    def setUp(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("opencv not installed")
        from real_robot.cameras import OrbbecCameras
        self.OrbbecCameras = OrbbecCameras

    def test_reads_rgb_at_sim_size_and_depth(self):
        from real_robot.config import IMAGE_HW
        sdk = FakeOrbbecSDK()
        cams = self.OrbbecCameras({"cam_high": "orbbec"}, ["cam_high"], sdk=sdk)
        self.assertEqual(sdk.started[0][1], ["color", "depth"])
        img = cams.read()["cam_high"]
        self.assertEqual(img.shape, (*IMAGE_HW, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(tuple(img[0, 0]), (200, 0, 0))
        self.assertEqual(cams.depth["cam_high"].shape, (4, 5))
        self.assertEqual(float(cams.depth["cam_high"][0, 0]), 700.0)
        cams.close()
        self.assertEqual(len(sdk.stopped), 1)

    def test_color_only_device(self):
        sdk = FakeOrbbecSDK(depth=False)
        cams = self.OrbbecCameras({"c": "orbbec"}, ["c"], sdk=sdk)
        self.assertEqual(sdk.started[0][1], ["color"])
        cams.read()
        self.assertEqual(cams.depth, {})

    def test_serial_selection(self):
        sdk = FakeOrbbecSDK(serials=["AB1", "CD2"])
        with self.assertRaisesRegex(RuntimeError, "orbbec:<serial>"):
            self.OrbbecCameras({"c": "orbbec"}, ["c"], sdk=sdk)
        with self.assertRaisesRegex(RuntimeError, "no Orbbec"):
            self.OrbbecCameras({"c": "orbbec:ZZ"}, ["c"], sdk=sdk)
        self.OrbbecCameras({"c": "orbbec:CD2"}, ["c"], sdk=sdk)
        self.assertEqual(sdk.started[-1][0].serial, "CD2")

    def test_router_splits_by_device(self):
        from real_robot.cameras import is_orbbec
        self.assertTrue(is_orbbec("orbbec"))
        self.assertTrue(is_orbbec("orbbec:AB1"))
        self.assertFalse(is_orbbec("/dev/v4l/by-id/usb-orbbec"))
        self.assertFalse(is_orbbec(0))


if __name__ == "__main__":
    unittest.main()
