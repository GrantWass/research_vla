"""Policy-registry tests: swapping VLA models is one flag.

Covers policies/*.conf (parse + required fields), scripts/run_eval.sh
(resolves every registered policy to its XPolicyLab adapter), and
adapters/turbovla_robodojo/ (XPolicyLab contract: deploy.yml, model.py
interface, launchers parse).

Mac-safe: system Python only, no torch/Isaac/GPU/conda. The TurboVLA adapter
is installed into the (git-ignored) RoboDojo checkout by the test setup via
scripts/install_adapter.sh, which only copies files.
Run:  python3 -m unittest discover -s tests -v   (from research_vla root)
"""

import filecmp
import os
import py_compile
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICIES_DIR = os.path.join(ROOT, "policies")
ADAPTER_SRC = os.path.join(ROOT, "adapters", "turbovla_robodojo")
ROBODOJO = os.path.join(ROOT, "RoboDojo")

REQUIRED_CONF_KEYS = [
    "POLICY_NAME",
    "DESCRIPTION",
    "XPOLICYLAB_POLICY_DIR",
    "POLICY_ENV",
    "ACTION_TYPE",
    "ENV_CFG",
]

ADAPTER_FILES = [
    "deploy.yml",
    "model.py",
    "eval.sh",
    "setup_eval_policy_server.sh",
    "setup_eval_env_client.sh",
    "install.sh",
]


def run(cmd, cwd=ROOT, timeout=120):
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )


def ensure_adapter_installed():
    """Install the this-repo adapter (file copies only, Mac-safe).

    Re-syncs with --force when the installed copy drifted from
    adapters/turbovla_robodojo/ (the source of truth), so the
    installed-matches-source contract below holds either way.
    """
    r = run(["bash", "scripts/install_adapter.sh", "turbovla"])
    if r.returncode != 0:
        r = run(["bash", "scripts/install_adapter.sh", "turbovla", "--force"])
    assert r.returncode == 0, f"install_adapter failed:\n{r.stderr}"


def parse_conf(path):
    """Minimal KEY=value parse (values may be double-quoted). No yaml dep."""
    out = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            assert "=" in line, f"{path}:{lineno}: not KEY=value: {line!r}"
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            assert key and " " not in key, f"{path}:{lineno}: bad key {key!r}"
            if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            out[key] = value
    return out


class TestPolicyRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.confs = {}
        for fname in sorted(os.listdir(POLICIES_DIR)):
            if fname.endswith(".conf"):
                cls.confs[fname[: -len(".conf")]] = parse_conf(
                    os.path.join(POLICIES_DIR, fname)
                )

    def test_known_policies_registered(self):
        for name in ["openvla", "turbovla", "demo", "pi05"]:
            self.assertIn(name, self.confs, f"policies/{name}.conf missing")

    def test_names_match_filenames(self):
        for name, conf in self.confs.items():
            self.assertEqual(
                conf.get("POLICY_NAME"), name, f"POLICY_NAME mismatch in {name}.conf"
            )

    def test_required_fields_present(self):
        for name, conf in self.confs.items():
            for key in REQUIRED_CONF_KEYS:
                self.assertTrue(conf.get(key), f"{name}.conf missing {key}")

    def test_action_type_values_valid(self):
        # A conf typo here would only surface on the GPU box; fail here.
        for name, conf in self.confs.items():
            self.assertIn(conf.get("ACTION_TYPE"), ["joint", "ee"], f"{name}.conf")

    def test_policy_dir_shape(self):
        for name, conf in self.confs.items():
            d = conf["XPOLICYLAB_POLICY_DIR"]
            self.assertTrue(d.startswith("XPolicyLab/policy/"), f"{name}: {d}")
            self.assertEqual(len(d.split("/")), 3, f"{name}: {d}")

    def test_upstream_adapters_exist(self):
        # openvla/demo/pi05 adapters ship with XPolicyLab; they must be present.
        for name in ["openvla", "demo", "pi05"]:
            d = os.path.join(ROBODOJO, self.confs[name]["XPOLICYLAB_POLICY_DIR"])
            self.assertTrue(
                os.path.isfile(os.path.join(d, "eval.sh")), f"missing adapter: {d}"
            )

    def test_uv_policies_use_uv_env_marker(self):
        # uv-managed adapters (Pi_05) take "uv" where conda policies take an
        # env name; robodojo.sh accepts "conda env, uv, or env path" there.
        self.assertEqual(self.confs["pi05"]["POLICY_ENV"], "uv")
        for name in ["openvla", "turbovla"]:
            self.assertNotEqual(self.confs[name]["POLICY_ENV"], "uv", name)

    def test_turbovla_conf_points_at_owned_adapter(self):
        self.assertTrue(
            os.path.isdir(ADAPTER_SRC), "adapters/turbovla_robodojo/ missing"
        )
        for fname in ADAPTER_FILES:
            self.assertTrue(
                os.path.isfile(os.path.join(ADAPTER_SRC, fname)),
                f"adapter missing {fname}",
            )


class TestRunEvalWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Install the this-repo adapter (file copies only, Mac-safe) so the
        # turbovla dry-run below resolves like it does on a set-up machine.
        ensure_adapter_installed()

    def test_scripts_parse(self):
        for script in ["scripts/run_eval.sh", "scripts/install_adapter.sh"]:
            r = run(["bash", "-n", script])
            self.assertEqual(r.returncode, 0, f"{script}: {r.stderr}")

    def test_list_reports_all_policies(self):
        r = run(["bash", "scripts/run_eval.sh", "--list"])
        self.assertEqual(r.returncode, 0, r.stderr)
        for name in ["openvla", "turbovla", "demo", "pi05"]:
            self.assertIn(name, r.stdout)

    def test_dry_run_resolves_each_policy(self):
        expectations = {
            "openvla": "XPolicyLab/policy/OpenVLA_OFT",
            "turbovla": "XPolicyLab/policy/TurboVLA",
            "demo": "XPolicyLab/policy/demo_policy",
            "pi05": "XPolicyLab/policy/Pi_05",
        }
        for policy, adapter in expectations.items():
            with self.subTest(policy=policy):
                r = run(
                    [
                        "bash",
                        "scripts/run_eval.sh",
                        "--policy",
                        policy,
                        "--task",
                        "stack_bowls",
                        "--dry-run",
                    ]
                )
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn(adapter, r.stdout + r.stderr)
                self.assertIn("stack_bowls", r.stdout + r.stderr)

    def test_swap_changes_only_model_flags(self):
        # The whole point: openvla <-> turbovla <-> pi05 differ in
        # policy-dir/env/ckpt, never in task or harness invocation.
        outs = {}
        for policy in ["openvla", "turbovla", "pi05"]:
            r = run(
                [
                    "bash",
                    "scripts/run_eval.sh",
                    "--policy",
                    policy,
                    "--task",
                    "stack_bowls",
                    "--dry-run",
                ]
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            outs[policy] = r.stdout + r.stderr
        self.assertIn("openvla_oft", outs["openvla"])
        self.assertIn("turbovla-robodojo", outs["turbovla"])
        self.assertIn(" 0 uv ", outs["pi05"])  # uv env marker, not a conda env
        for out in outs.values():
            self.assertIn("stack_bowls", out)
            self.assertIn("run_policy_eval.sh", out)

    def test_cli_overrides_beat_conf(self):
        # Regression: sourcing the conf must not clobber CLI-passed
        # --env-cfg/--action-type (they used to be silently ignored).
        base = [
            "bash",
            "scripts/run_eval.sh",
            "--policy",
            "turbovla",
            "--task",
            "stack_bowls",
            "--dry-run",
        ]
        r = run(base)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("arx_x5 joint", r.stdout + r.stderr)  # conf defaults
        r = run(base + ["--env-cfg", "dual_x5", "--action-type", "ee"])
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout + r.stderr
        self.assertIn("dual_x5", out)
        self.assertIn(" ee ", out)

    def test_smoke_mode_forwards_conf_action_type(self):
        # Regression: smoke/benchmark dropped --env-cfg/--action-type, so
        # robodojo.sh fell back to `ee` and drove joint-space ckpts wrongly.
        # (--dry-run would switch run_eval.sh to dry-run mode, so stub the
        # harness instead and read the args it receives.)
        with tempfile.TemporaryDirectory() as tmp:
            stub = os.path.join(tmp, "robodojo.sh")
            with open(stub, "w") as f:
                f.write('echo "ARGS: $*"\n')
            for mode in ["smoke", "benchmark"]:
                with self.subTest(mode=mode):
                    r = subprocess.run(
                        [
                            "bash",
                            "scripts/run_eval.sh",
                            "--policy",
                            "openvla",
                            "--task",
                            "stack_bowls",
                            "--mode",
                            mode,
                        ],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        env={**os.environ, "ROBODOJO_SH": stub},
                        check=False,
                    )
                    self.assertEqual(r.returncode, 0, r.stderr)
                    self.assertIn(f"ARGS: {mode} ", r.stdout)
                    self.assertIn("--action-type joint", r.stdout)
                    self.assertIn("--env-cfg arx_x5", r.stdout)

    def test_unknown_policy_fails_with_hint(self):
        r = run(
            [
                "bash",
                "scripts/run_eval.sh",
                "--policy",
                "nope",
                "--task",
                "stack_bowls",
                "--dry-run",
            ]
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Unknown policy", r.stderr)

    def test_install_adapter_noop_for_upstream_policies(
        self,
    ):  # Upstream adapters (openvla/demo/pi05) ship with XPolicyLab:
        # install must succeed as a no-op, so `make install-adapter`
        # never blocks the one-flag swap story.
        for policy in ["openvla", "demo", "pi05"]:
            with self.subTest(policy=policy):
                r = run(["bash", "scripts/install_adapter.sh", policy])
                self.assertEqual(r.returncode, 0, f"{policy}: {r.stderr}")
                self.assertIn("nothing to install", r.stdout)


class TestTurboVLAAdapter(unittest.TestCase):
    INSTALLED = os.path.join(ROBODOJO, "XPolicyLab", "policy", "TurboVLA")

    @classmethod
    def setUpClass(cls):
        ensure_adapter_installed()

    def test_installed_matches_source(self):
        for fname in ADAPTER_FILES:
            with self.subTest(file=fname):
                self.assertTrue(
                    filecmp.cmp(
                        os.path.join(ADAPTER_SRC, fname),
                        os.path.join(self.INSTALLED, fname),
                        shallow=False,
                    ),
                    f"{fname}: installed copy differs from adapters/ source",
                )

    def test_launchers_parse(self):
        for fname in [
            "eval.sh",
            "setup_eval_policy_server.sh",
            "setup_eval_env_client.sh",
            "install.sh",
        ]:
            for base in [ADAPTER_SRC, self.INSTALLED]:
                r = run(["bash", "-n", os.path.join(base, fname)])
                self.assertEqual(r.returncode, 0, f"{base}/{fname}: {r.stderr}")

    def test_deploy_yml_contract(self):
        with open(os.path.join(ADAPTER_SRC, "deploy.yml")) as f:
            yml = f.read()
        for line in ["policy_name: TurboVLA", "protocol: ws"]:
            self.assertIn(line, yml)

    def test_model_adapter_interface(self):
        path = os.path.join(ADAPTER_SRC, "model.py")
        py_compile.compile(path, doraise=True)
        with open(path) as f:
            src = f.read()
        for method in ["update_obs", "get_action", "reset"]:
            self.assertIn(f"def {method}", src)
        self.assertIn("ModelTemplate", src)
        self.assertIn("unpack_robot_state", src)


class TestTurboVLACheckout(unittest.TestCase):
    def test_setup_pins_turbovla(self):
        with open(os.path.join(ROOT, "setup.sh")) as f:
            src = f.read()
        self.assertIn("TURBOVLA_PIN=", src)
        self.assertIn("https://github.com/H-EmbodVis/TurboVLA.git", src)

    @unittest.skipUnless(
        os.path.isdir(os.path.join(ROOT, "turbovla", ".git")),
        "turbovla/ not cloned (run setup.sh with network)",
    )
    def test_turbovla_layout(self):
        for rel in [
            "turbovla/pyproject.toml",
            "turbovla/turbovla/models/turbovla.py",
            "turbovla/turbovla/models/configuration.py",
            "turbovla/turbovla/evaluation/policy.py",
        ]:
            self.assertTrue(os.path.exists(os.path.join(ROOT, rel)), f"missing: {rel}")


if __name__ == "__main__":
    unittest.main()
