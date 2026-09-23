"""GPU-box tooling: setup_policy.sh, lowvram.sh, patches/, smoke configs.

Mac-safe (stdlib only). The patch checks run `git apply --check` against the
pinned upstream checkouts, so an upstream bump that breaks a patch fails here
instead of on someone's GPU box.
"""

import os
import re
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBODOJO = os.path.join(ROOT, "RoboDojo")
XPL = os.path.join(ROBODOJO, "XPolicyLab")
PATCHES = os.path.join(ROOT, "patches")
ADAPTER_SRC = os.path.join(ROOT, "adapters", "turbovla_robodojo")

TURBOVLA = os.path.join(ROOT, "turbovla")

# patch file -> repo it applies to. Every .patch must be listed here
# (test_every_patch_has_a_target); a target whose checkout is absent on this
# machine is skipped rather than failed, since the turbovla training repo is
# only cloned on the GPU box.
PATCH_TARGETS = {
    "robodojo_lowvram_sim.patch": ROBODOJO,
    "xpolicylab_openvla_oft_lowvram.patch": XPL,
    "xpolicylab_pi05_mem_fraction.patch": XPL,
    "turbovla_full_ckpt_init.patch": TURBOVLA,
    "turbovla_lerobot_video_index.patch": TURBOVLA,
}


def run(cmd, **kw):
    kw.setdefault("cwd", ROOT)
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, check=False, **kw
    )


class TestScripts(unittest.TestCase):
    def test_scripts_parse(self):
        for script in ["scripts/setup_policy.sh", "scripts/lowvram.sh"]:
            with self.subTest(script=script):
                r = run(["bash", "-n", script])
                self.assertEqual(r.returncode, 0, r.stderr)

    def test_setup_policy_help_and_bad_name(self):
        r = run(["bash", "scripts/setup_policy.sh", "--help"])
        self.assertEqual(r.returncode, 0, r.stderr)
        for name in ["openvla", "pi05", "turbovla", "demo"]:
            self.assertIn(name, r.stdout)
        r = run(["bash", "scripts/setup_policy.sh", "not_a_policy"])
        self.assertEqual(r.returncode, 2)

    def test_lowvram_patch_list_matches_files(self):
        with open(os.path.join(ROOT, "scripts", "lowvram.sh")) as f:
            src = f.read()
        for name in re.findall(r"patches/([\w.]+\.patch)", src):
            self.assertTrue(os.path.isfile(os.path.join(PATCHES, name)), name)


@unittest.skipUnless(os.path.isdir(XPL), "upstream checkouts missing (run setup.sh)")
class TestPatchesApply(unittest.TestCase):
    def test_every_patch_has_a_target(self):
        self.assertEqual(
            sorted(p for p in os.listdir(PATCHES) if p.endswith(".patch")),
            sorted(PATCH_TARGETS),
        )

    def test_patches_apply_or_are_applied(self):
        for name, repo in PATCH_TARGETS.items():
            with self.subTest(patch=name):
                if not os.path.isdir(repo):
                    self.skipTest(f"{os.path.basename(repo)} checkout missing")
                path = os.path.join(PATCHES, name)
                fwd = run(["git", "-C", repo, "apply", "--check", path])
                rev = run(["git", "-C", repo, "apply", "--check", "--reverse", path])
                self.assertTrue(
                    fwd.returncode == 0 or rev.returncode == 0,
                    f"{name} neither applies nor is applied: {fwd.stderr}",
                )

    def test_lowvram_status_runs(self):
        r = run(["bash", "scripts/lowvram.sh", "status"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("robodojo_lowvram_sim.patch", r.stdout)


class TestRunEvalLowvramEnv(unittest.TestCase):
    ENV_FILE = os.path.join(ROOT, ".lowvram.env")

    def test_env_file_exported_to_harness(self):
        if os.path.exists(self.ENV_FILE):
            self.skipTest(".lowvram.env already present (profile applied here)")
        with tempfile.TemporaryDirectory() as tmp:
            stub = os.path.join(tmp, "robodojo.sh")
            with open(stub, "w") as f:
                f.write('echo "FRAC=${XLA_PYTHON_CLIENT_MEM_FRACTION:-unset}"\n')
            with open(self.ENV_FILE, "w") as f:
                f.write("export XLA_PYTHON_CLIENT_MEM_FRACTION=0.42\n")
            try:
                env = {**os.environ, "ROBODOJO_SH": stub}
                env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
                r = run(
                    [
                        "bash",
                        "scripts/run_eval.sh",
                        "--policy",
                        "pi05",
                        "--task",
                        "stack_bowls",
                        "--mode",
                        "smoke",
                    ],
                    env=env,
                )
            finally:
                os.remove(self.ENV_FILE)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("FRAC=0.42", r.stdout)
        self.assertIn("low-VRAM profile active", r.stderr)


class TestTurboVLADeployConfigs(unittest.TestCase):
    @staticmethod
    def _keys(path):
        with open(path) as f:
            return {line.split(":", 1)[0] for line in f if re.match(r"^[a-z_]+:", line)}

    def test_robotwin_smoke_has_same_keys_as_adapter(self):
        base = self._keys(os.path.join(ADAPTER_SRC, "deploy.yml"))
        smoke = self._keys(os.path.join(ADAPTER_SRC, "deploy.robotwin_smoke.yml"))
        self.assertEqual(base - smoke, set(), "smoke config is missing adapter knobs")

    def test_robotwin_smoke_matches_ckpt_shape(self):
        with open(os.path.join(ADAPTER_SRC, "deploy.robotwin_smoke.yml")) as f:
            yml = f.read()
        for line in [
            "num_views: 3",
            "state_dim: 14",
            "action_dim: 14",
            "action_layout: arms_first",
            "dinov3-vitl16",
            "@SHARED@",
        ]:
            self.assertIn(line, yml)


if __name__ == "__main__":
    unittest.main()
