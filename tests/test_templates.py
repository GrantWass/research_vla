"""Template sanity tests: template shell parses, Python compiles, names match.

Guards the templates/ folder without needing Isaac or the XPolicyLab tree.
Run:  python3 -m unittest discover -s tests -v   (from research_vla root)
"""
import os
import py_compile
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK_TMPL = os.path.join(ROOT, "templates", "robodojo_task")
POL_TMPL = os.path.join(ROOT, "templates", "xpolicylab_policy", "my_policy")


class TestTaskTemplate(unittest.TestCase):
    def test_python_compiles_and_names_match(self):
        path = os.path.join(TASK_TMPL, "my_task.py")
        py_compile.compile(path, doraise=True)
        with open(path) as f:
            src = f.read()
        self.assertIn("class my_task(", src)  # exported class == module basename
        self.assertIn("def run_reward", src)
        self.assertIn("reward_manager.check(", src)

    def test_yaml_labels_match_python(self):
        # Minimal check: every label= string in the .py appears in the .yml.
        import re

        with open(os.path.join(TASK_TMPL, "my_task.py")) as f:
            py_src = f.read()
        with open(os.path.join(TASK_TMPL, "my_task.yml")) as f:
            yml_src = f.read()
        labels = set(re.findall(r'label="([^"]+)"', py_src))
        self.assertTrue(labels, "template should reference at least one label")
        for label in labels:
            self.assertIn(label, yml_src, f"label {label!r} missing from YAML")


class TestPolicyTemplate(unittest.TestCase):
    def test_shell_parses(self):
        for fname in [
            "eval.sh",
            "setup_eval_policy_server.sh",
            "setup_eval_env_client.sh",
            "install.sh",
        ]:
            r = subprocess.run(
                ["bash", "-n", os.path.join(POL_TMPL, fname)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(r.returncode, 0, f"{fname}: {r.stderr}")

    def test_python_compiles(self):
        for fname in ["model.py", "deploy.py"]:
            py_compile.compile(os.path.join(POL_TMPL, fname), doraise=True)

    def test_deploy_yml_contract(self):
        with open(os.path.join(POL_TMPL, "deploy.yml")) as f:
            yml = f.read()
        self.assertIn("policy_name: my_policy", yml)
        self.assertIn("protocol: ws", yml)

    def test_model_adapter_interface(self):
        with open(os.path.join(POL_TMPL, "model.py")) as f:
            src = f.read()
        for method in ["update_obs", "get_action", "reset"]:
            self.assertIn(f"def {method}", src)


if __name__ == "__main__":
    unittest.main()
