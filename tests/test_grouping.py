"""Scientific invariants of group construction and robust attribution."""
import unittest
from star_dro.grouping import GroupInventory, annotation_keys


def annotations():
    return [{"Code": "A", "Sub-code": "shared", "Span": "one"},
            {"Code": "B", "Sub-code": "shared", "Span": "two"},
            {"Code": "A", "Sub-code": "unique", "Span": "one"}]


class GroupingTests(unittest.TestCase):
    def test_five_constructions_do_not_conflate_group_identity(self):
        ann = annotations()
        expected = {"code": ["A", "B", "A"], "subcode": ["shared", "shared", "unique"],
                    "code_subcode": ['["A","shared"]', '["B","shared"]', '["A","unique"]'],
                    "num_annotations": ["NA_3"] * 3,
                    "code_subcode_num_annotations": ['["A","shared",3]', '["B","shared",3]', '["A","unique",3]']}
        for by, keys in expected.items():
            with self.subTest(groupby=by):
                self.assertEqual(annotation_keys(ann, by), keys)
                inv = GroupInventory.fit([{"annotations": ann}], by)
                ids, members = inv.encode(ann)
                self.assertEqual([inv.names[i] for i in ids], keys)
                self.assertEqual(members, sorted(set(ids)))
                self.assertNotIn("__OTHER__", inv.names)

    def test_pair_keys_do_not_create_cartesian_pairs_or_delimiter_collisions(self):
        ann = [{"Code": "A||B", "Sub-code": "C", "Span": "one"},
               {"Code": "A", "Sub-code": "B||C", "Span": "two"}]
        self.assertEqual(len(set(annotation_keys(ann, "code_subcode"))), 2)
        self.assertEqual(len(GroupInventory.fit([{"annotations": ann}], "code_subcode").names), 2)

    def test_count_aliases_count_objects_including_repeated_spans(self):
        self.assertEqual(annotation_keys(annotations(), "num_span"), ["NA_3"] * 3)
        self.assertEqual(annotation_keys(annotations(), "code_subcode_num_span"),
                         annotation_keys(annotations(), "code_subcode_num_annotations"))

    def test_inventory_is_training_only_and_pool_is_explicit(self):
        rows = [{"annotations": annotations()}, {"annotations": [annotations()[0]]}]
        inv = GroupInventory.fit(rows, "code", min_count=2)
        self.assertEqual(inv.names, ("A", "__OTHER__"))
        self.assertEqual(inv.rare_keys, ("B",))
        new = [{"Code": "C", "Sub-code": "shared", "Span": "new"}]
        self.assertEqual(inv.encode(new), ([1], [1]))
        with self.assertRaisesRegex(ValueError, "absent from the training inventory"):
            GroupInventory.fit(rows).encode(new)

    def test_empty_records_have_only_count_membership(self):
        inv = GroupInventory.fit([{"annotations": []}], "num_annotations")
        self.assertEqual(inv.encode([]), ([], [0]))
        self.assertEqual(GroupInventory.fit([{"annotations": annotations()}]).encode([]), ([], []))


try:
    import torch
    from star_dro.torch_loss import annotation_loss, sample_loss
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class AttributionTests(unittest.TestCase):
    def test_annotation_and_sample_modes_allocate_different_gradients(self):
        class Fixed:
            def update(self, losses, memberships, progress):
                self.losses, self.memberships = losses, memberships
                return [3, 1]
        logits = torch.zeros(2, 5, 3, requires_grad=True)
        labels = torch.tensor([[-100, 0, 1, 2, 0], [-100, 0, 1, -100, -100]])
        masks = torch.tensor([[[True, False, False, False], [False, True, False, False]],
                              [[True, False, False, False], [False, False, False, False]]])
        groups = torch.tensor([[0, 1], [1, -1]])
        ann_adv, sample_adv = Fixed(), Fixed()
        loss, diag = annotation_loss(logits, labels, masks, groups, ann_adv)
        loss.backward()
        ann_grad = logits.grad.clone()
        logits.grad.zero_()
        loss, sample_diag = sample_loss(logits, labels, [[0, 1], [1]], sample_adv)
        loss.backward()
        self.assertEqual(ann_adv.memberships, [[0], [1], [1]])
        self.assertEqual(sample_adv.memberships, [[0, 1], [1]])
        self.assertEqual(sample_diag["sample_weights"].tolist(), [2, 1])
        self.assertEqual(diag["token_weights"][0].tolist(), [3, 1, 1, 1])
        self.assertAlmostEqual(float(ann_grad[0, 0].norm() / ann_grad[0, 1].norm()), 3)
        self.assertAlmostEqual(float(logits.grad[0, 0].norm() / logits.grad[0, 1].norm()), 1)
        # Sample means compensate unequal completion lengths: relative token
        # gradients are (2/4)/(1/2) = 1 here, not 2 from token pooling.
        self.assertAlmostEqual(float(logits.grad[0, 0].norm() / logits.grad[1, 0].norm()), 1)
        self.assertTrue(torch.equal(logits.grad[1, 2], torch.zeros(3)))

    def test_real_adversary_sample_overlap_and_annotation_observations(self):
        from star_dro import Config, StarDRO
        logits = torch.tensor([[[2., 0.], [0., 0.], [0., 0.]], [[0., 0.], [0., 0.], [0., 0.]]], requires_grad=True)
        labels = torch.tensor([[-100, 0, 0], [-100, 1, -100]])
        adv = StarDRO(Config(2))
        loss, _ = sample_loss(logits, labels, [[0, 1], [0]], adv)
        ce0 = (torch.nn.functional.cross_entropy(logits[0, 0][None], torch.tensor([0])) + torch.log(torch.tensor(2.))) / 2
        expected = (float(ce0.detach())/2 + float(torch.log(torch.tensor(2.)))) / 1.5
        self.assertAlmostEqual(adv.ema[0], expected, places=6)
        self.assertAlmostEqual(adv.ema[1], float(ce0.detach()), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
