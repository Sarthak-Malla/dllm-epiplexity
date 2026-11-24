"""Run: python -m unittest discover -s /home/juan.guevara/naruto/dllm-epiplexity/scripts/tests -p test_offline_master_gate.py."""

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


PATH = Path(__file__).resolve().parents[1] / "tse/ctca/offline_master_gate.py"
SPEC = importlib.util.spec_from_file_location("offline_master_gate", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class OfflineGateTests(unittest.TestCase):
    def test_tail_bounds_contain_full_distribution_metrics(self):
        p, q = [.4, .3, .2, .1], [.1, .4, .1, .4]
        support = [[str(i), p[i], q[i]] for i in range(2)]
        bounds = audit.agreement_bounds(support)
        hp, hq = sum(map(audit.h, p)), sum(map(audit.h, q))
        js = sum(audit.h((x + y) / 2) for x, y in zip(p, q)) - (hp + hq) / 2
        for metric, actual in (("js", 1 - js / math.log(2)),
                               ("overlap", sum(min(x, y) for x, y in zip(p, q)))):
            self.assertLessEqual(bounds[metric][0], actual + 1e-12)
            self.assertGreaterEqual(bounds[metric][1], actual - 1e-12)
        low, high = audit.uncertainty(p[:2], 4)
        self.assertLessEqual(low, hp / math.log(4))
        self.assertGreaterEqual(high, hp / math.log(4))

    def test_full_support_and_unknown_disjoint_tails(self):
        self.assertEqual(audit.agreement_bounds([["a", 1., 0.], ["b", 0., 1.]])["js"], [0., 0.])
        bounds = audit.agreement_bounds([["a", .5, .5]])
        self.assertAlmostEqual(bounds["js"][0], .5)
        self.assertAlmostEqual(bounds["js"][1], 1.)
        self.assertEqual(bounds["overlap"], [.5, 1.])

    def test_decisive_and_ambiguous_actions(self):
        row = {"js": [.6, .8], "master_uncertainty": [.2, .3],
               "projected_auxiliary_uncertainty": [.7, .9]}
        self.assertEqual(audit.decision(row, "js", "one", .5, .5), "ensemble")
        self.assertEqual(audit.decision(row, "js", "both", .5, .5), "master")
        self.assertEqual(audit.decision(row, "js", "one", .7, .5), "ambiguous")
        self.assertEqual(audit.decision(row, "js", "one", .9, .5), "master")

    def test_bad_mass_fails_but_distinct_ids_with_same_text_are_preserved(self):
        bounds = audit.agreement_bounds([["a", .8, .1], ["a", .2, .9]])
        self.assertAlmostEqual(bounds["overlap"][0], .3)
        self.assertAlmostEqual(bounds["overlap"][1], .3)
        expected = audit.agreement_bounds([["id0", .8, .1], ["id1", .2, .9]])
        self.assertEqual(bounds, expected)
        with self.assertRaises(ValueError):
            audit.uncertainty([.8, .8], 4)

    def test_trace_streaming_preserves_step_and_selection(self):
        distribution = {"top_k": [{"token": "a", "probability": .8},
                                   {"token": "b", "probability": .2}], "vocabulary_size": 2}
        rows = []
        for step in ([0, 0], [0, 1]):
            base = {"step": step, "batch_index": 0}
            rows.extend([
                {**base, "event": "auxiliary_masked_position", "generation_position": 0,
                 "distribution": distribution},
                {**base, "event": "forward_position", "master": {"absolute_position": 2, "distribution": distribution},
                 "spatial_warp": {"distribution": distribution},
                 "auxiliary": {"contributors": [{"generation_position": 0, "overlap": 1}]}},
                {**base, "event": "fusion_candidate", "absolute_position": 2,
                 "projected_auxiliary_distribution": distribution,
                 "fusion_support": [["a", .8, .8], ["b", .2, .2]],
                 "predicted_token": {"token": "a"}, "fused_probability": .8,
                 "weights": {"master": .5, "auxiliary": .5}},
                {**base, "event": "commit_decision", "absolute_position": 2, "selected": True}])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)))
            result = list(audit.audit_trace(path))
        self.assertEqual([r["step"] for r in result], [[0, 0], [0, 1]])
        self.assertTrue(all(r["selected"] for r in result))
        self.assertEqual(result[0]["confidence_change_if_fallback"], 0)


if __name__ == "__main__":
    unittest.main()
