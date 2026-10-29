import json
import math
import unittest
from star_dro import Config, StarDRO, backmap, schedule


class AlgorithmTests(unittest.TestCase):
    def test_ema_continues_across_epoch_boundary(self):
        s = StarDRO(Config(2, rho_start=0.5, rho_end=0.5))
        # Last robust minibatch of epoch 2.
        s.update([2, 4], [[0], [1]], progress=0.65)
        # First minibatch of epoch 3: retain history, rather than reinitialize.
        s.update([10, 8], [[0], [1]], progress=0.7)
        self.assertEqual(s.ema, [6, 6])
        self.assertNotEqual(s.ema, [10, 8])
        self.assertEqual(s.steps, 2)

    def test_rescaled_signal_is_count_weighted_and_scale_invariant(self):
        a, b = StarDRO(Config(3)), StarDRO(Config(3))
        memberships = [[0], [0], [1]]
        a.update([2, 2, 8], memberships)
        b.update([20, 20, 80], memberships)
        self.assertEqual(a.last_update["scale"], 4)
        self.assertEqual(a.last_update["ascent_signal"], [0.5, 2, 0])
        signal = a.last_update["ascent_signal"]
        self.assertAlmostEqual((2 * signal[0] + signal[1]) / 3, 1)
        for x, y in zip(a.q, b.q):
            self.assertAlmostEqual(x, y)

    def test_excess_only_multiplier_neutral_and_ceiling(self):
        s = StarDRO(Config(3, alpha=2, eta=10, upper=7, gamma=0.75))
        m = s.update([0, 0, 10], [[0], [1], [2]])
        self.assertEqual(s.q, [0, 0, 1])
        self.assertEqual(m, [1, 1, 7])
        self.assertEqual(s.last_update["multipliers"], m)

    def test_projection_known_sparsemax(self):
        self.assertEqual(backmap([2, 0, -1], 2), [1, 0, 0])
        q = backmap([0.7, 0.5, 0], 2)
        for a, b in zip(q, [0.6, 0.4, 0]):
            self.assertAlmostEqual(a, b)

    def test_projection_translation_and_inversion(self):
        for alpha in [1.1, 1.2, 1.3, 1.5, 2, 3]:
            original = [0.0, 0.1, 0.3, 0.6]
            u = [q ** (alpha-1) for q in original]
            for actual, expected in zip(backmap(u, alpha), original):
                self.assertAlmostEqual(actual, expected, places=11)
            shifted = backmap([x-7 for x in u], alpha)
            for actual, expected in zip(shifted, original):
                self.assertAlmostEqual(actual, expected, places=10)

    def test_negative_equal_dual_is_uniform(self):
        self.assertEqual(backmap([-3, -3], 1.2), [0.5, 0.5])

    def test_hand_calculated_update(self):
        s = StarDRO(Config(2, alpha=2, eta=0.1, upper=3, gamma=1))
        m = s.update([1, 3], [[0], [1]])
        self.assertAlmostEqual(s.q[0], 0.45)
        self.assertAlmostEqual(s.q[1], 0.55)
        self.assertAlmostEqual(m[0], 1)
        self.assertAlmostEqual(m[1], 1.2)

    def test_overlap_and_count_weighted_scale(self):
        s = StarDRO(Config(2, alpha=2, eta=0.1))
        s.update([2, 6], [[0, 1], [0]])
        self.assertAlmostEqual(s.ema[0], 14/3)
        self.assertAlmostEqual(s.ema[1], 2)
        scale = (2*(14/3)+2)/3
        # Euclidean projection subtracts common mass excess / G.
        expected0 = 0.5 + 0.1 * ((14/3)/scale - ((14/3)/scale+2/scale)/2)
        self.assertAlmostEqual(s.q[0], expected0)

    def test_persistence_absent_group_and_first_seen(self):
        s = StarDRO(Config(3, rho_start=0.5, rho_end=0.5))
        s.update([2, 4], [[0], [1]])
        old = s.ema[1]
        m = s.update([6, 9], [[0], [2]])
        self.assertEqual(s.ema, [4, old, 9])
        self.assertEqual(m[1], 1)

    def test_empty_update_no_state_change(self):
        s = StarDRO(Config(2))
        state = s.state_dict()
        self.assertEqual(s.update([], []), [1, 1])
        self.assertEqual(state, s.state_dict())

    def test_single_group_neutral(self):
        s = StarDRO(Config(1))
        self.assertEqual(s.update([3], [[0]]), [1])
        self.assertEqual(s.q, [1])

    def test_repeated_updates_bounds_and_sparse_support(self):
        s = StarDRO(Config(4, alpha=2, eta=0.3))
        for _ in range(100):
            m = s.update([1, 2, 3, 10], [[0], [1], [2], [3]])
            self.assertAlmostEqual(sum(s.q), 1)
            self.assertTrue(all(1 <= x <= 15 for x in m))
        self.assertEqual(s.q, [0, 0, 0, 1])

    def test_checkpoint_continuation(self):
        a = StarDRO(Config(3))
        a.update([1, 3], [[0], [2]])
        b = StarDRO.from_state_dict(json.loads(json.dumps(a.state_dict())))
        self.assertEqual(a.update([4, 2], [[0], [1]], 0.7), b.update([4, 2], [[0], [1]], 0.7))
        self.assertEqual(a.state_dict(), b.state_dict())

    def test_cosine(self):
        self.assertAlmostEqual(schedule(.3, .03, 0), .3)
        self.assertAlmostEqual(schedule(.3, .03, 1), .03)
        self.assertAlmostEqual(schedule(.3, .03, .5), .165)

    def test_bad_inputs(self):
        for kw in [{"alpha": 1}, {"eta": -1}, {"rho_end": 0}, {"gamma": 0}, {"upper": 1}]:
            with self.assertRaises(ValueError):
                Config(2, **kw)
        for losses, memberships in [([1], [[2]]), ([1], [[0, 0]]), ([float("nan")], [[0]]), ([1], [])]:
            with self.assertRaises(ValueError):
                StarDRO(Config(2)).update(losses, memberships)


try:
    import torch
    from star_dro.torch_loss import annotation_loss
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class TorchTests(unittest.TestCase):
    def test_gradient_weighting_and_neutral_tokens(self):
        logits = torch.zeros(1, 5, 3, requires_grad=True)
        labels = torch.tensor([[-100, 0, 1, 2, 0]])
        masks = torch.tensor([[[True, False, False, False], [False, True, False, False]]])
        gids = torch.tensor([[0, 1]])
        class FixedAdversary:
            def update(self, losses, memberships, progress):
                return [1, 3]
        loss, diag = annotation_loss(logits, labels, masks, gids, FixedAdversary())
        self.assertTrue(torch.equal(diag["token_weights"], torch.tensor([[1., 3., 1., 1.]])))
        loss.backward()
        self.assertAlmostEqual(float(logits.grad[0, 1].norm()/logits.grad[0, 0].norm()), 3)
        self.assertAlmostEqual(float(loss.detach()), math.log(3), places=6)

    def test_eval_neutral_and_padding(self):
        logits = torch.zeros(1, 4, 3, requires_grad=True)
        labels = torch.tensor([[-100, 1, 2, -100]])
        loss, _ = annotation_loss(logits, labels, torch.zeros(1, 0, 3, dtype=torch.bool), torch.empty(1, 0, dtype=torch.long))
        self.assertAlmostEqual(float(loss.detach()), math.log(3), places=6)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[0, 2], torch.zeros(3)))


if __name__ == "__main__":
    unittest.main()

