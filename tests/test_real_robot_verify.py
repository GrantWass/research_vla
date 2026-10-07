"""verify.py stages against a mock robot and a fake openpi policy."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

REPO = os.path.join(os.path.dirname(__file__), "..")

if np is not None:
    from real_robot.backends import MockBackend
    from real_robot.config import load_config
    from real_robot.verify import (Report, from_droid_actions, stage_policy,
                                   stage_robot, to_droid_obs)


class FakeDroidPolicy:
    """Mimics openpi Policy: _input_transform like DroidInputs + resize, infer -> (10, 8)."""

    def __init__(self):
        self.last = None

    def _input_transform(self, obs):
        base = obs["observation/exterior_image_1_left"][:224, :224]
        wrist = obs["observation/wrist_image_left"][:224, :224]
        state = np.concatenate([obs["observation/joint_position"],
                                obs["observation/gripper_position"]])
        return {"state": state,
                "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist,
                          "right_wrist_0_rgb": np.zeros_like(base)},
                "image_mask": {"base_0_rgb": True, "left_wrist_0_rgb": True,
                               "right_wrist_0_rgb": False}}

    def infer(self, obs):
        self.last = obs
        return {"actions": np.arange(80, dtype=float).reshape(10, 8)}


@unittest.skipIf(np is None, "numpy not installed")
class TestVerify(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(os.path.join(REPO, "real_robot/configs/piper_1.json"))

    def test_droid_mapping(self):
        state = np.array([1, 2, 3, 4, 5, 6, 0.035])
        imgs = MockBackend(self.cfg).get_images()
        obs = to_droid_obs(state, imgs, self.cfg, "x")
        np.testing.assert_array_equal(obs["observation/joint_position"], [1, 2, 3, 4, 5, 6, 0])
        self.assertAlmostEqual(float(obs["observation/gripper_position"][0]), 0.5)
        self.assertIs(obs["observation/wrist_image_left"], imgs["cam_wrist"])
        ours = from_droid_actions(np.arange(16).reshape(2, 8), self.cfg)
        np.testing.assert_array_equal(ours[0], [0, 1, 2, 3, 4, 5, 7])

    def test_robot_then_policy_stage(self):
        with tempfile.TemporaryDirectory() as d:
            rep = Report(d)
            robot = MockBackend(self.cfg)
            snap = stage_robot(rep, robot, self.cfg, frames=3)
            self.assertFalse(rep.failed("robot"))
            self.assertTrue(os.path.exists(os.path.join(d, "snapshot.npz")))
            pol = FakeDroidPolicy()
            stage_policy(rep, pol, self.cfg, snap["state"], snap["images"], "x", n_infer=2)
            self.assertFalse(rep.failed("policy"))
            self.assertEqual(rep.data["policy"]["model -> our action"]["ok"], True)
            files = os.listdir(d)
            self.assertIn("model_sees_base_0_rgb.png", files)
            self.assertIn("raw_cam_wrist.png", files)
            rep.save()
            self.assertTrue(os.path.exists(os.path.join(d, "report.json")))
            self.assertEqual(robot.commands, [])

    def test_wiggle_declined_sends_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            robot = MockBackend(self.cfg)
            stage_robot(Report(d), robot, self.cfg, wiggle=True, frames=1,
                        confirm=lambda _: "no")
            self.assertEqual(robot.commands, [])

    def test_out_of_limits_fails(self):
        with tempfile.TemporaryDirectory() as d:
            rep = Report(d)
            robot = MockBackend(self.cfg)
            robot.q[0] = 3.0  # beyond J1 limit 2.618
            stage_robot(rep, robot, self.cfg, frames=1)
            self.assertTrue(rep.failed("robot"))


if __name__ == "__main__":
    unittest.main()
