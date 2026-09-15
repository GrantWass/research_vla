"""Eval-wiring tests: scripts parse, dry-run resolves, contracts intact.

No Isaac, no policy server, no GPU. The `--dry-run` paths print the resolved
command without launching anything, so they are safe on Apple Silicon.
Run:  python3 -m unittest discover -s tests -v   (from research_vla root)
"""

import os
import py_compile
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBODOJO = os.path.join(ROOT, "RoboDojo")
XPL_POLICY = os.path.join(ROBODOJO, "XPolicyLab", "policy")


def run(cmd, cwd, timeout=120):
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )


def has_mapfile():
    # smoke_all_tasks.sh needs bash 4+ (mapfile); stock macOS ships bash 3.2.
    # `brew install bash` provides it; the Linux GPU box is unaffected.
    r = subprocess.run(["bash", "-c", "type mapfile"], capture_output=True, check=False)
    return r.returncode == 0


def deploy_yml_value(policy, key):
    # Minimal "key: value" parse; avoids a pyyaml dependency.
    with open(os.path.join(XPL_POLICY, policy, "deploy.yml")) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip()
    return None


class TestShellSyntax(unittest.TestCase):
    def test_robodojo_scripts_parse(self):
        for script in ["scripts/robodojo.sh", "scripts/eval_policy.sh"]:
            r = run(["bash", "-n", script], cwd=ROBODOJO)
            self.assertEqual(r.returncode, 0, f"{script}: {r.stderr}")

    def test_policy_scripts_parse(self):
        for policy in ["demo_policy", "OpenVLA_OFT"]:
            for script in [
                "eval.sh",
                "setup_eval_policy_server.sh",
                "setup_eval_env_client.sh",
            ]:
                path = os.path.join("XPolicyLab", "policy", policy, script)
                if not os.path.isfile(os.path.join(ROBODOJO, path)):
                    continue
                r = run(["bash", "-n", path], cwd=ROBODOJO)
                self.assertEqual(r.returncode, 0, f"{path}: {r.stderr}")


class TestDryRunEval(unittest.TestCase):
    def test_eval_dry_run_resolves(self):
        r = run(
            [
                "bash",
                "scripts/robodojo.sh",
                "eval",
                "--policy-dir",
                "XPolicyLab/policy/demo_policy",
                "--task",
                "stack_bowls",
                "--ckpt",
                "demo",
                "--policy-env",
                "demo-env",
                "--dry-run",
            ],
            cwd=ROBODOJO,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("stack_bowls", r.stdout + r.stderr)

    @unittest.skipUnless(
        has_mapfile(), "needs bash 4+ (mapfile); stock macOS bash is 3.2"
    )
    def test_smoke_dry_run_resolves(self):
        r = run(
            [
                "bash",
                "scripts/robodojo.sh",
                "smoke",
                "--policy-dir",
                "XPolicyLab/policy/demo_policy",
                "--ckpt",
                "demo",
                "--policy-env",
                "demo-env",
                "--only",
                "stack_bowls,push_T",
                "--dry-run",
            ],
            cwd=ROBODOJO,
        )
        self.assertEqual(r.returncode, 0, r.stderr)


class TestPolicyContract(unittest.TestCase):
    def test_websocket_protocol(self):
        # The eval client only speaks ws by default; a tcp-only deploy.yml
        # would fail at connection time on the GPU box, not here.
        for policy in ["demo_policy", "OpenVLA_OFT"]:
            self.assertEqual(deploy_yml_value(policy, "protocol"), "ws", policy)

    def test_adapter_entry_points(self):
        for policy in ["demo_policy", "OpenVLA_OFT"]:
            for fname in ["eval.sh", "deploy.yml"]:
                self.assertTrue(
                    os.path.isfile(os.path.join(XPL_POLICY, policy, fname)),
                    f"{policy}/{fname}",
                )

    def test_eval_scripts_have_bash_shebang(self):
        for policy in ["demo_policy", "OpenVLA_OFT"]:
            with open(os.path.join(XPL_POLICY, policy, "eval.sh")) as f:
                self.assertTrue(f.readline().startswith("#!/bin/bash"), policy)


class TestOpenVLAScripts(unittest.TestCase):
    def test_key_scripts_compile(self):
        for rel in [
            "vla-scripts/deploy.py",
            "vla-scripts/finetune.py",
            "experiments/robot/libero/run_libero_eval.py",
        ]:
            path = os.path.join(ROOT, "openvla", rel)
            py_compile.compile(path, doraise=True)


if __name__ == "__main__":
    unittest.main()
