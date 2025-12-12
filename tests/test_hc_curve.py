from __future__ import annotations

import unittest


import hc_curve


class TestCurveMath(unittest.TestCase):
    def test_compute_curve_linear_gain(self) -> None:
        # Linear cumulative gain: G(t) = rate * t.
        times = [float(t) for t in range(0, 101)]
        rate = 2.0  # m/s gain rate in cumulative series
        gains = [rate * t for t in times]
        durations = [10, 50, 100]
        points = hc_curve.compute_curve(times, gains, durations, engine="numpy")
        self.assertEqual([p.duration_s for p in points], durations)
        for p in points:
            expected = rate * p.duration_s
            self.assertAlmostEqual(p.max_climb_m, expected, places=6)


class TestFitParsingHelpers(unittest.TestCase):
    def test_pick_total_gain_key_prefers_total_gain(self) -> None:
        sample = {
            "foo": 1.0,
            "Total Gain": 10.0,
            "total_ascent": 20.0,
        }
        key = hc_curve._pick_total_gain_key(sample)
        self.assertEqual(key, "Total Gain")

    def test_merge_records_overlap_policy_last(self) -> None:
        # Near-duplicate timestamps should coalesce, keeping non-null tg
        # and preferring higher-priority distance.
        recs0 = [
            {"t": 0.0, "file_id": 0, "tg": 5.0, "alt": None, "inc": None, "dist": 10.0, "dist_prio": 1},
        ]
        recs1 = [
            {"t": 0.1, "file_id": 1, "tg": None, "alt": None, "inc": None, "dist": 12.0, "dist_prio": 3},
        ]

        merged = hc_curve._merge_records([recs0, recs1], merge_eps_sec=0.5, overlap_policy="file:last")
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["tg"], 5.0)
        self.assertAlmostEqual(merged[0]["dist"], 12.0)


if __name__ == "__main__":
    unittest.main()
