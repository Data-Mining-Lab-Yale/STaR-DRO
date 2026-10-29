"""Requested hardness-scale choices and notebook-matched LoRA targets."""
import json
from pathlib import Path
import unittest
from star_dro import Config, StarDRO
from star_dro.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


class RequestedSettingsTests(unittest.TestCase):
    def test_uniform_and_count_signals(self):
        count = StarDRO(Config(3, hardness_scale="count"))
        uniform = StarDRO(Config(3, hardness_scale="uniform"))
        default = StarDRO(Config(3))
        for state in (count, uniform, default):
            state.update([1, 1, 7], [[0], [0], [1]])
        self.assertEqual(count.last_update["scale"], 3)
        self.assertEqual(uniform.last_update["scale"], 4)
        self.assertEqual(uniform.last_update["ascent_signal"], [.25, 1.75, 0])
        self.assertEqual(count.last_update["ascent_signal"], [1/3, 7/3, 0])
        self.assertNotEqual(count.q, uniform.q)
        self.assertEqual(count.q, default.q)

    def test_scaling_checkpoint_and_validation(self):
        state = StarDRO(Config(3, hardness_scale="uniform"))
        state.update([2, 5], [[0], [2]])
        restored = StarDRO.from_state_dict(json.loads(json.dumps(state.state_dict())))
        self.assertEqual(restored.config.hardness_scale, "uniform")
        self.assertEqual(state.update([3, 8], [[1], [2]], .5), restored.update([3, 8], [[1], [2]], .5))
        legacy = state.state_dict();legacy["config"].pop("hardness_scale")
        self.assertEqual(StarDRO.from_state_dict(legacy).config.hardness_scale, "count")
        with self.assertRaises(ValueError):
            Config(3, hardness_scale="invalid")

    def test_default_parameters_targets_and_uniform_preset(self):
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        for cfg in (load_config(), load_config(ROOT/"configs/reference.json")):
            self.assertEqual(cfg["lora"]["target_modules"], targets)
            self.assertEqual(cfg["star_dro"], {"alpha": 1.2, "eta": .0006, "rho_start": .3,
                                              "rho_end": .03, "upper": 15., "gamma": .75, "hardness_scale": "count"})
        self.assertEqual(load_config(ROOT/"configs/uniform_scaling.json")["star_dro"]["hardness_scale"], "uniform")
