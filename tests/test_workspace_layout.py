"""Workspace layout smoke tests: repos cloned, key paths present.

Runs on any machine with system Python. No GPU, Isaac, torch, or conda needed.
Run:  python3 -m unittest discover -s tests -v   (from research_vla root)
"""
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def p(*parts):
    return os.path.join(ROOT, *parts)


class TestWorkspaceLayout(unittest.TestCase):
    def assertPathExists(self, path):  # noqa: N802
        self.assertTrue(os.path.exists(path), f"missing: {path}")

    def test_openvla_cloned(self):
        for rel in [
            "openvla/pyproject.toml",
            "openvla/vla-scripts/deploy.py",
            "openvla/vla-scripts/finetune.py",
            "openvla/vla-scripts/train.py",
            "openvla/requirements-min.txt",
            "openvla/experiments/robot/libero/run_libero_eval.py",
        ]:
            self.assertPathExists(p(rel))

    def test_openvla_is_code_repo(self):
        # openvla/ is the runnable codebase (the openvla.github.io website
        # clone was removed: 108 MB of static assets, zero Python files).
        self.assertPathExists(p("openvla/prismatic"))

    def test_robodojo_cloned(self):
        for rel in [
            "RoboDojo/scripts/robodojo.sh",
            "RoboDojo/scripts/eval_policy.sh",
            "RoboDojo/scripts/internal/task_inventory.py",
            "RoboDojo/task/RoboDojo",
            "RoboDojo/env",
            "RoboDojo/env_cfg",
            "RoboDojo/src/eval_client",
        ]:
            self.assertPathExists(p(rel))

    def test_xpolicylab_submodule_initialized(self):
        for rel in [
            "RoboDojo/XPolicyLab/policy/demo_policy/eval.sh",
            "RoboDojo/XPolicyLab/policy/demo_policy/deploy.yml",
            "RoboDojo/XPolicyLab/policy/demo_policy/model.py",
            "RoboDojo/XPolicyLab/policy/demo_policy/deploy.py",
            "RoboDojo/XPolicyLab/policy/OpenVLA_OFT/eval.sh",
            "RoboDojo/XPolicyLab/policy/OpenVLA_OFT/deploy.yml",
        ]:
            self.assertPathExists(p(rel))

    def test_setup_guide_exists(self):
        self.assertPathExists(p("OPENVLA_ROBODOJO_SETUP.md"))

    def test_readme_maps_workspace(self):
        # The root README is the entry point; it must exist and point at
        # every owned area so new teammates find tests/templates/docs.
        path = p("README.md")
        self.assertPathExists(path)
        with open(path) as f:
            src = f.read()
        for keyword in [
            "tests/",
            "templates/robodojo_task/",
            "templates/xpolicylab_policy/",
            "templates/openvla_finetune/",
            "OPENVLA_ROBODOJO_SETUP.md",
            "CONTRIBUTING.md",
            "setup.sh",
            "make test",
        ]:
            self.assertIn(keyword, src, f"README missing: {keyword}")


if __name__ == "__main__":
    unittest.main()
