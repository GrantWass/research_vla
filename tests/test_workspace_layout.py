"""Workspace layout: docs and README in place; upstream checkouts when cloned.

Runs on any machine with system Python. No GPU, Isaac, torch, or conda needed.
Run:  python3 -m unittest discover -s tests -v   (from research_vla root)
"""

import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def p(*parts):
    return os.path.join(ROOT, *parts)


needs_robodojo = unittest.skipUnless(
    os.path.isfile(p("RoboDojo", "scripts", "robodojo.sh")),
    "RoboDojo checkout missing (bash setup.sh)",
)


class TestWorkspaceLayout(unittest.TestCase):
    def assertPathExists(self, path):
        self.assertTrue(os.path.exists(path), f"missing: {path}")

    @needs_robodojo
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

    @needs_robodojo
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

    def test_docs_exist(self):
        for rel in ["docs/robot.md", "docs/sim.md", "docs/serving.md", "docs/findings.md"]:
            self.assertPathExists(p(rel))

    def test_readme_maps_workspace(self):
        # The root README is the entry point; it must exist and point at
        # every owned area so new teammates find the code and docs.
        path = p("README.md")
        self.assertPathExists(path)
        with open(path) as f:
            src = f.read()
        for keyword in [
            "real_robot/",
            "tests/",
            "templates/",
            "docs/robot.md",
            "docs/sim.md",
            "CONTRIBUTING.md",
            "setup.sh",
            "make test",
        ]:
            self.assertIn(keyword, src, f"README missing: {keyword}")


if __name__ == "__main__":
    unittest.main()
