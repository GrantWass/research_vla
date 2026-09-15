"""TurboVLA-on-RoboDojo training tests: overlay parity + stats math.

The training recipe itself needs the GPU box (torch, lerobot, 4x GPU), but
everything this repo owns is verifiable on a Mac with system Python:
  - data_registry/data_config.py mirrors the upstream robotwin50 modality keys
    (same LeRobot feature schema on both sides), checked via AST (no imports).
  - configs/robodojo.yaml keeps the recipe's config shape (key parity with the
    paper's clean50.yaml) and the 14-D / 3-view / horizon-50 settings.
  - compute_stats.summarize_frames is pure numpy: exact mean/std/min/max on
    synthetic frames, JSON-serializable output.
  - train.sh + install_turbovla_training.sh parse; train.sh requires the
    overlay and the model-asset env vars.
Run:  python3 -m unittest discover -s tests -v   (from research_vla root)
"""
import ast
import json
import os
import py_compile
import subprocess
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMPL = os.path.join(ROOT, "templates", "turbovla_finetune")
OVERLAY = os.path.join(ROOT, "turbovla", "experiments", "robodojo")
UPSTREAM_DC = os.path.join(
    ROOT, "turbovla", "experiments", "robotwin", "data_registry", "data_config.py")
UPSTREAM_YAML = os.path.join(
    ROOT, "turbovla", "experiments", "robotwin", "configs", "clean50.yaml")

sys.path.insert(0, TMPL)


def run(cmd, cwd=ROOT, timeout=120):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def modality_key_lists(tree):
    """{class attr name -> list literal} for *_keys assignments."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name)
                        and target.id.endswith("_keys")
                        and isinstance(node.value, ast.List)):
                    out[target.id] = [e.value for e in node.value.elts
                                      if isinstance(e, ast.Constant)]
    return out


def top_level_keys(path):
    """Top-level `key:` names of a simple YAML (no pyyaml needed)."""
    keys = []
    with open(path) as f:
        for line in f:
            if line and not line[0].isspace() and ":" in line \
                and not line.startswith("#"):
                keys.append(line.split(":", 1)[0].strip())
    return keys


def yml_value(path, *key_path):
    """Fetch nested scalar/list from simple YAML by indentation (2-space)."""
    with open(path) as f:
        lines = f.readlines()
    parsed: dict = {}
    stack = [(parsed, -1)]
    for line in lines:
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = stripped.strip().partition(":")
        key, value = key.strip(), value.strip()
        while stack and indent <= stack[-1][1]:
            stack.pop()
        parent = stack[-1][0]
        if value == "":
            child = {}
            parent[key] = child
            stack.append((child, indent))
        else:
            try:
                parent[key] = json.loads(value)
            except ValueError:
                parent[key] = value.strip('"')
    node = parsed
    for key in key_path:
        node = node[key]
    return node


class TestTrainingOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r = run(["bash", "scripts/install_turbovla_training.sh"])
        assert r.returncode == 0, f"install_training failed:\n{r.stderr}"

    def test_scripts_parse(self):
        for script in ["scripts/install_turbovla_training.sh",
                       "templates/turbovla_finetune/train.sh"]:
            r = run(["bash", "-n", script])
            self.assertEqual(r.returncode, 0, f"{script}: {r.stderr}")

    def test_overlay_matches_source(self):
        import filecmp
        for rel in ["data_registry/data_config.py", "configs/robodojo.yaml"]:
            with self.subTest(file=rel):
                self.assertTrue(
                    filecmp.cmp(os.path.join(TMPL, rel),
                                os.path.join(OVERLAY, rel), shallow=False))

    def test_modality_keys_match_upstream(self):
        # The whole no-rewrite claim: RoboDojo data uses the same LeRobot
        # feature names the paper's robotwin50 config consumes.
        with open(os.path.join(TMPL, "data_registry", "data_config.py")) as f:
            ours = modality_key_lists(ast.parse(f.read()))
        with open(UPSTREAM_DC) as f:
            upstream = modality_key_lists(ast.parse(f.read()))
        for attr in ["video_keys", "state_keys", "action_keys", "language_keys"]:
            with self.subTest(attr=attr):
                self.assertTrue(ours.get(attr), f"missing {attr}")
                self.assertEqual(ours[attr], upstream[attr])

    def test_data_config_registries(self):
        with open(os.path.join(TMPL, "data_registry", "data_config.py")) as f:
            src = f.read()
        tree = ast.parse(src)
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        top = {t.id for t in tree.body
               if isinstance(t, ast.Assign) for t in t.targets
               if isinstance(t, ast.Name)}
        self.assertIn("RoboDojoArxX5DataConfig", names)
        for symbol in ["ROBOT_TYPE_CONFIG_MAP", "ROBOT_TYPE_TO_EMBODIMENT_TAG",
                       "DATASET_NAMED_MIXTURES"]:
            self.assertIn(symbol, top, f"missing registry {symbol}")
        self.assertIn("robodojo_arx_x5", src)
        self.assertIn("ROBODOJO_TASKS", src)

    def test_data_config_compiles(self):
        py_compile.compile(
            os.path.join(TMPL, "data_registry", "data_config.py"), doraise=True)

    def test_train_launcher_contract(self):
        with open(os.path.join(TMPL, "train.sh")) as f:
            src = f.read()
        for needle in ["ROBODOJO_DATA_ROOT", "BERT_MODEL_PATH",
                       "TURBOVLA_INIT_CKPT", "DINOV3_MODEL_PATH",
                       "install_turbovla_training.sh",
                       "train_robotwin_clean_act_pi05_recipe.py",
                       "meta/info.json", "ROBODOJO_TASKS"]:
            self.assertIn(needle, src, f"train.sh missing {needle}")


class TestRoboDojoYaml(unittest.TestCase):
    PATH = os.path.join(TMPL, "configs", "robodojo.yaml")

    def test_recipe_shape_parity(self):
        # Same top-level config shape the training recipe reads.
        self.assertEqual(top_level_keys(self.PATH), top_level_keys(UPSTREAM_YAML))

    def test_robodojo_dims(self):
        self.assertEqual(yml_value(self.PATH, "framework", "action", "action_dim"), 14)
        self.assertEqual(yml_value(self.PATH, "framework", "action", "state_dim"), 14)
        self.assertEqual(yml_value(self.PATH, "framework", "action", "horizon"), 50)
        self.assertEqual(yml_value(self.PATH, "framework", "vision", "num_views"), 3)
        self.assertEqual(yml_value(self.PATH, "framework", "vision", "image_size"), 224)
        self.assertEqual(
            yml_value(self.PATH, "datasets", "vla_data", "data_mix"), "robodojo_arx_x5")
        self.assertIn("ROBODOJO_DATA_ROOT",
                      yml_value(self.PATH, "datasets", "vla_data", "data_root_dir"))

    def test_deploy_robodojo_matches_training(self):
        from pathlib import Path as _P
        deploy = _P(TMPL) / "deploy.robodojo.yml"
        self.assertTrue(deploy.is_file())
        src = deploy.read_text()
        for line in ["policy_name: TurboVLA", "protocol: ws",
                     "num_views: 3", "image_size: 224", "chunk_size: 50",
                     "state_dim: 14", "action_dim: 14"]:
            self.assertIn(line, src)


class TestComputeStats(unittest.TestCase):
    def test_summarize_frames_exact(self):
        from compute_stats import summarize_frames
        rng = np.random.default_rng(0)
        states = rng.normal(size=(100, 14))
        actions = rng.uniform(-1, 1, size=(100, 14))
        out = summarize_frames(states, actions)
        np.testing.assert_allclose(
            out["proprio"]["mean"], states.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(
            out["proprio"]["std"], states.std(axis=0), rtol=1e-6)
        np.testing.assert_allclose(
            out["action"]["min"], actions.min(axis=0), rtol=1e-6)
        np.testing.assert_allclose(
            out["action"]["max"], actions.max(axis=0), rtol=1e-6)
        # Adapter-consumable: plain lists, JSON round-trips.
        json.dumps(out)

    def test_summarize_frames_rejects_bad_input(self):
        from compute_stats import summarize_frames
        with self.assertRaises(ValueError):
            summarize_frames(np.zeros((0, 14)), np.zeros((0, 14)))
        with self.assertRaises(ValueError):
            summarize_frames(np.zeros((10, 14)), np.zeros((8, 14)))

    def test_script_compiles(self):
        py_compile.compile(os.path.join(TMPL, "compute_stats.py"), doraise=True)


if __name__ == "__main__":
    unittest.main()
