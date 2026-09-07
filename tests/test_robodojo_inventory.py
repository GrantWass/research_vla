"""RoboDojo task-inventory tests: 54 runnable tasks, names consistent.

Parses `task_inventory.py --format json --check` output; never imports Isaac.
Run:  python3 -m unittest discover -s tests -v   (from research_vla root)
"""
import json
import os
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBODOJO = os.path.join(ROOT, "RoboDojo")


def run_inventory():
    out = subprocess.run(
        ["python3", "scripts/internal/task_inventory.py", "--format", "json", "--check"],
        cwd=ROBODOJO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, f"task_inventory failed:\n{out.stderr}"
    return json.loads(out.stdout)


class TestTaskInventory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inv = run_inventory()

    def test_counts(self):
        counts = self.inv["counts"]
        self.assertEqual(counts["runnable"], 54, f"counts: {counts}")
        self.assertEqual(counts["missing_class"], 0)
        self.assertEqual(counts["missing_config"], 0)
        self.assertEqual(counts["config_only"], 0)

    def test_runnable_tasks_have_matching_files(self):
        # Registry rule (RoboDojo/CLAUDE.md): YAML name == module basename ==
        # exported env class name == result path. Check the file side here.
        bad = []
        for t in self.inv["tasks"]:
            if not t.get("runnable"):
                continue
            cfg = os.path.join(ROBODOJO, t["config"])
            mod = os.path.join(ROBODOJO, t["module"])
            if not os.path.isfile(cfg):
                bad.append(f"{t['name']}: missing config {t['config']}")
            if not os.path.isfile(mod):
                bad.append(f"{t['name']}: missing module {t['module']}")
            if os.path.basename(cfg) != f"{t['name']}.yml":
                bad.append(f"{t['name']}: config basename mismatch")
            if os.path.basename(mod) != f"{t['name']}.py":
                bad.append(f"{t['name']}: module basename mismatch")
            if t.get("class_name") != t["name"]:
                bad.append(f"{t['name']}: class_name mismatch")
        self.assertEqual(bad, [])

    def test_runnable_tasks_define_real_reward(self):
        # Every task must implement run_reward() via reward_manager.check(...);
        # a trivially-True success check silently corrupts eval metrics.
        bad = []
        for t in self.inv["tasks"]:
            if not t.get("runnable"):
                continue
            with open(os.path.join(ROBODOJO, t["module"])) as f:
                src = f.read()
            if "def run_reward" not in src:
                bad.append(f"{t['name']}: no run_reward()")
            elif "reward_manager.check(" not in src and "reward_manager" not in src:
                bad.append(f"{t['name']}: run_reward ignores reward_manager")
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
