import unittest

from hyscript.preference_evaluation import bootstrap_interval, summarize_preferences


class PreferenceTests(unittest.TestCase):
    def setUp(self):
        self.mapping = [
            {"blind_id": "P1", "A_source": "editorial_candidates", "B_source": "single_shot", "target_length": "280", "topic_cluster_id": "T1"},
            {"blind_id": "P2", "A_source": "single_shot", "B_source": "editorial_candidates", "target_length": "450", "topic_cluster_id": "T1"},
        ]
        self.rows = [{"blind_id": key, "preference": "A", "A_usability": "可直接采用", "B_usability": "需大改"} for key in ("P1", "P2")]

    def test_side_reversal(self):
        result = summarize_preferences(self.rows, self.mapping)
        self.assertEqual(result["editorial_preference_rate"], .5)
        self.assertEqual(result["usability"]["editorial_candidates"], {"可直接采用": 1, "需大改": 1})
        self.assertEqual(result["clusters"], {"T1": [1, 0]})

    def test_ties_and_unable_are_distinct(self):
        self.rows[0]["preference"] = "无明显偏好"
        self.rows[1]["preference"] = "无法判断"
        result = summarize_preferences(self.rows, self.mapping)
        self.assertEqual(result["preference_denominator"], 1)
        self.assertEqual(result["counts"], {"tie": 1, "unable": 1})
        self.rows[0]["preference"] = ""
        self.assertIsNone(summarize_preferences(self.rows, self.mapping)["editorial_preference_rate"])

    def test_bad_identifiers_rejected(self):
        with self.assertRaises(ValueError):
            summarize_preferences(self.rows + [self.rows[0]], self.mapping)
        with self.assertRaises(ValueError):
            summarize_preferences(self.rows[:1], self.mapping)

    def test_bootstrap_preserves_topic_group(self):
        self.assertEqual(bootstrap_interval({"T1": [1, 0, 1]}, seed=1, repeats=20), [2/3, 2/3])
