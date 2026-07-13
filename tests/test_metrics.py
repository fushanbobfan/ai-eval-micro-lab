import unittest

from ai_eval_micro_lab.metrics import exact_match, evaluate_records, token_f1


class MetricTests(unittest.TestCase):
    def test_exact_match_normalizes_case_and_punctuation(self):
        self.assertEqual(exact_match("Hello, UCLA!", "hello ucla"), 1.0)

    def test_token_f1_counts_duplicate_tokens(self):
        self.assertAlmostEqual(token_f1("a a b", "a b b"), 2 / 3)

    def test_empty_evaluation_is_explicit(self):
        self.assertEqual(
            evaluate_records([]),
            {"count": 0, "exact_match": 0.0, "token_f1": 0.0},
        )


if __name__ == "__main__":
    unittest.main()

