"""Numerical notebook144 baseline and explicit overlapping-split smoke policy."""
import contextlib
import io
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from star_dro.standard_dro import StandardDRO
from star_dro.data import check_disjoint, split_overlap_report
from star_dro.cli import main
from star_dro.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


class StandardDROTests(unittest.TestCase):
    def test_hand_calculated_q_and_count_correction(self):
        dro = StandardDRO(2)
        weights = dro.update([2, 2, 4], [[0], [0], [1]])
        q0 = 1 / (1 + math.exp(.2))
        self.assertAlmostEqual(dro.q[0], q0)
        self.assertAlmostEqual(weights[0], q0 / (2/3))
        self.assertAlmostEqual(weights[1], (1-q0) / (1/3))
        self.assertAlmostEqual((2*weights[0] + weights[1]) / 3, 1)
        self.assertEqual(dro.ema, [None, None])

    def test_matches_notebook_full_simplex_update_over_200_steps(self):
        rng = random.Random(42)
        dro, q = StandardDRO(4), [0.25] * 4
        for step in range(200):
            groups = [rng.randrange(4) for _ in range(1 + step % 9)]
            losses = [rng.uniform(.01, 4) for _ in groups]
            counts = [groups.count(g) for g in range(4)]
            means = [sum(loss for loss, group in zip(losses, groups) if group == g) / counts[g] if counts[g] else 0 for g in range(4)]
            updated = [q[g] * math.exp(.1 * means[g]) for g in range(4)]
            q = [x / sum(updated) for x in updated]
            present_mass = sum(q[g] for g in range(4) if counts[g])
            actual = dro.update(losses, [[g] for g in groups])
            for g in range(4):
                self.assertAlmostEqual(dro.q[g], q[g], places=12)
                expected = q[g] / (counts[g]/len(groups)) / present_mass if counts[g] else 1
                self.assertAlmostEqual(actual[g], expected, places=11)

    def test_checkpoint_continuation_and_empty_step(self):
        state = StandardDRO(3)
        state.update([1, 2], [[0], [2]])
        saved = json.loads(json.dumps(state.state_dict()))
        restored = StandardDRO.from_state_dict(saved)
        self.assertEqual(state.update([7, 3], [[1], [2]]), restored.update([7, 3], [[1], [2]]))
        saved = state.state_dict()
        self.assertEqual(state.update([], []), [1, 1, 1])
        self.assertEqual(saved, state.state_dict())

    def test_small_q_can_recover_without_nonfinite_multipliers(self):
        state = StandardDRO(2)
        state.update([100000], [[0]])
        weights = state.update([100000], [[1]])
        self.assertTrue(all(math.isfinite(x) for x in weights))
        self.assertAlmostEqual(sum(state.q), 1)
        self.assertAlmostEqual(state.q[0], .5)

    def test_configuration_and_bad_inputs(self):
        cfg = load_config(ROOT / 'configs/standard_dro.json')
        self.assertEqual(cfg['training']['objective'], 'standard_dro')
        self.assertEqual(cfg['standard_dro']['eta'], .1)
        self.assertEqual(cfg['training']['dro_start_epoch'], 2)
        for eta in (0, -1, float('nan'), True):
            with self.assertRaises(ValueError): StandardDRO(2, eta)
        with self.assertRaises(ValueError): StandardDRO(2).update([1], [[0, 1]])
        with self.assertRaises(ValueError): StandardDRO(2).update([float('nan')], [[0]])


class SplitSmokeTests(unittest.TestCase):
    def rows(self):
        ann = [{"Code": "Topic", "Sub-code": "Schedule", "Span": "Office"}]
        return [{"id": "train-0", "prompt": "Office hours?", "annotations": ann}], [{"id": "valid-0", "prompt": "Office hours?", "annotations": ann}]

    def test_default_blocks_overlap_and_explicit_exception_reports_it(self):
        train, valid = self.rows()
        with self.assertRaises(ValueError): check_disjoint(train, valid)
        report = check_disjoint(train, valid, allow_overlap=True)
        self.assertEqual(report['0:1']['exact_inputs'], 1)
        self.assertEqual(split_overlap_report({'train': train, 'valid': valid})['train:valid']['exact_inputs'], 1)

    def test_cli_records_smoke_exception_without_changing_rows(self):
        train, valid = self.rows()
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(temporary)
            for name, rows in [('train', train), ('valid', valid)]:
                (root / (name+'.json')).write_text(json.dumps(rows))
            command = ['validate-data', '--train', str(root/'train.json'), '--valid', str(root/'valid.json')]
            with self.assertRaises(SystemExit): main(command)
            report = main(command + ['--allow-split-overlap'])
            self.assertEqual(report['evaluation_scope'], 'smoke_test_with_overlap_allowed')
            self.assertEqual(report['train']['records'], 1)
            self.assertEqual(report['split_overlap']['train:valid']['exact_inputs'], 1)
