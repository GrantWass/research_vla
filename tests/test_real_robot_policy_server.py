"""Policy server protocol over a Unix socket, with a fake policy (no torch)."""
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np

    from real_robot import lerobot_policy as lp
except ImportError:  # pragma: no cover
    np = None


class FakePolicy:
    def __init__(self):
        self.obs = None
        self.resets = 0

    def call(self, func_name, **kw):
        if func_name == "update_obs":
            self.obs = kw["obs"]
        elif func_name == "reset":
            self.resets += 1
        elif func_name == "get_action":
            if self.obs["instruction"] == "boom":
                raise ValueError("bad obs")
            return np.tile(np.r_[self.obs["state"], self.obs["extra_state"]], (50, 1))


@unittest.skipIf(np is None, "numpy not installed")
class TestPolicyServer(unittest.TestCase):
    def setUp(self):
        from multiprocessing.connection import Listener

        self.dir = tempfile.TemporaryDirectory()
        self.addr = os.path.join(self.dir.name, "p.sock")
        self.policy = FakePolicy()
        self.info = {"path": "ckpt", "camera_map": {"a": "b"}, "base_offset": 1.5,
                     "compile_mode": None, "num_steps": None, "noise_seed": 0}
        self.listener = Listener(self.addr, family="AF_UNIX", authkey=lp._AUTHKEY)
        self.served = []

        def loop():
            while True:
                with self.listener.accept() as conn:
                    if lp._handle(conn, self.policy, self.info):
                        return
                    self.served.append(1)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def tearDown(self):
        c = lp.PolicyClient(self.addr)
        c.call("shutdown")
        self.thread.join(2)
        self.listener.close()
        self.dir.cleanup()

    def test_runs_reuse_one_loaded_policy(self):
        for run in range(2):  # two runs, one server
            c = lp.PolicyClient(self.addr, extra_state=lambda: [7.0, 8.0],
                                expect={"path": "ckpt", "base_offset": 1.5, "noise_seed": 0})
            c.call("reset")
            c.call("update_obs", obs={"state": np.array([1.0, 2.0]), "instruction": "x"})
            a = c.call("get_action")
            np.testing.assert_allclose(a[0], [1, 2, 7, 8])  # extra_state read on the client
            c.close()
        self.assertEqual(self.policy.resets, 2)

    def test_mismatched_settings_refused(self):
        with self.assertRaises(ValueError):
            lp.PolicyClient(self.addr, expect={"base_offset": 0.0})
        with self.assertRaises(ValueError):
            lp.PolicyClient(self.addr, expect={"path": "other"})

    def test_policy_error_reaches_client_and_server_survives(self):
        c = lp.PolicyClient(self.addr, extra_state=lambda: [0.0])
        c.call("update_obs", obs={"state": np.zeros(2), "instruction": "boom"})
        with self.assertRaises(RuntimeError):
            c.call("get_action")
        c.call("update_obs", obs={"state": np.zeros(2), "instruction": "ok"})
        self.assertEqual(c.call("get_action").shape, (50, 3))
        c.close()

    def test_no_server(self):
        with self.assertRaises(RuntimeError):
            lp.PolicyClient(os.path.join(self.dir.name, "none.sock"))


if __name__ == "__main__":
    unittest.main()
