"""
Unit tests for retrospective evaluation metrics.

Tests cover:
- Recall@K: boundary cases, found/not found, k variations
- Rank: found at different positions, not found
- Precision@K: boundary cases, found/not found, k variations
"""

import unittest
from src.metrics import recall_at_k, mean_rank, precision_at_k


class TestRecallAtK(unittest.TestCase):
    """Test cases for recall_at_k function."""

    def test_recall_at_k_found_at_position_1(self):
        """Ground truth at position 1 should return 1.0."""
        result = recall_at_k(["A", "B", "C"], "A", 2)
        self.assertEqual(result, 1.0)

    def test_recall_at_k_found_at_position_k(self):
        """Ground truth at position k should return 1.0."""
        result = recall_at_k(["A", "B", "C"], "C", 3)
        self.assertEqual(result, 1.0)

    def test_recall_at_k_found_within_k(self):
        """Ground truth within top-k should return 1.0."""
        result = recall_at_k(["A", "B", "C", "D"], "B", 3)
        self.assertEqual(result, 1.0)

    def test_recall_at_k_not_found_in_top_k(self):
        """Ground truth beyond top-k should return 0.0."""
        result = recall_at_k(["A", "B", "C"], "C", 2)
        self.assertEqual(result, 0.0)

    def test_recall_at_k_not_found_in_list(self):
        """Ground truth not in list should return 0.0."""
        result = recall_at_k(["A", "B", "C"], "D", 3)
        self.assertEqual(result, 0.0)

    def test_recall_at_k_empty_list(self):
        """Empty predicted list should return 0.0."""
        result = recall_at_k([], "A", 5)
        self.assertEqual(result, 0.0)

    def test_recall_at_k_k_zero(self):
        """k=0 should return 0.0."""
        result = recall_at_k(["A", "B", "C"], "A", 0)
        self.assertEqual(result, 0.0)

    def test_recall_at_k_k_negative(self):
        """k<0 should return 0.0."""
        result = recall_at_k(["A", "B", "C"], "A", -1)
        self.assertEqual(result, 0.0)

    def test_recall_at_k_k_larger_than_list(self):
        """k larger than list length should check entire list."""
        result = recall_at_k(["A", "B", "C"], "C", 100)
        self.assertEqual(result, 1.0)

    def test_recall_at_k_with_duplicates(self):
        """Duplicates in predicted list should be handled correctly."""
        result = recall_at_k(["A", "B", "A", "C"], "A", 2)
        self.assertEqual(result, 1.0)

    def test_recall_at_k_case_sensitive(self):
        """Matching should be case-sensitive."""
        result = recall_at_k(["A", "B", "C"], "a", 3)
        self.assertEqual(result, 0.0)


class TestMeanRank(unittest.TestCase):
    """Test cases for mean_rank function."""

    def test_mean_rank_found_at_position_1(self):
        """Ground truth at position 1 should return 1.0."""
        result = mean_rank(["A", "B", "C"], "A")
        self.assertEqual(result, 1.0)

    def test_mean_rank_found_at_position_2(self):
        """Ground truth at position 2 should return 2.0."""
        result = mean_rank(["A", "B", "C"], "B")
        self.assertEqual(result, 2.0)

    def test_mean_rank_found_at_position_3(self):
        """Ground truth at position 3 should return 3.0."""
        result = mean_rank(["A", "B", "C"], "C")
        self.assertEqual(result, 3.0)

    def test_mean_rank_not_found(self):
        """Ground truth not in list should return len(predicted) + 1."""
        result = mean_rank(["A", "B", "C"], "D")
        self.assertEqual(result, 4.0)

    def test_mean_rank_empty_list(self):
        """Empty predicted list should return 1.0 (0 + 1)."""
        result = mean_rank([], "A")
        self.assertEqual(result, 1.0)

    def test_mean_rank_single_element_found(self):
        """Single element list with match should return 1.0."""
        result = mean_rank(["A"], "A")
        self.assertEqual(result, 1.0)

    def test_mean_rank_single_element_not_found(self):
        """Single element list without match should return 2.0."""
        result = mean_rank(["A"], "B")
        self.assertEqual(result, 2.0)

    def test_mean_rank_with_duplicates(self):
        """Duplicates should return position of first occurrence."""
        result = mean_rank(["A", "B", "A", "C"], "A")
        self.assertEqual(result, 1.0)

    def test_mean_rank_case_sensitive(self):
        """Matching should be case-sensitive."""
        result = mean_rank(["A", "B", "C"], "a")
        self.assertEqual(result, 4.0)

    def test_mean_rank_returns_float(self):
        """Result should be a float."""
        result = mean_rank(["A", "B", "C"], "B")
        self.assertIsInstance(result, float)


class TestPrecisionAtK(unittest.TestCase):
    """Test cases for precision_at_k function."""

    def test_precision_at_k_found_at_position_1(self):
        """Ground truth at position 1 with k=2 should return 0.5."""
        result = precision_at_k(["A", "B", "C"], "A", 2)
        self.assertEqual(result, 0.5)

    def test_precision_at_k_found_at_position_k(self):
        """Ground truth at position k should return 1/k."""
        result = precision_at_k(["A", "B", "C"], "C", 3)
        self.assertAlmostEqual(result, 1.0 / 3, places=5)

    def test_precision_at_k_not_found_in_top_k(self):
        """Ground truth beyond top-k should return 0.0."""
        result = precision_at_k(["A", "B", "C"], "C", 2)
        self.assertEqual(result, 0.0)

    def test_precision_at_k_not_found_in_list(self):
        """Ground truth not in list should return 0.0."""
        result = precision_at_k(["A", "B", "C"], "D", 3)
        self.assertEqual(result, 0.0)

    def test_precision_at_k_k_1(self):
        """k=1 should return 1.0 if found, 0.0 otherwise."""
        result_found = precision_at_k(["A", "B", "C"], "A", 1)
        result_not_found = precision_at_k(["A", "B", "C"], "B", 1)
        self.assertEqual(result_found, 1.0)
        self.assertEqual(result_not_found, 0.0)

    def test_precision_at_k_empty_list(self):
        """Empty predicted list should return 0.0."""
        result = precision_at_k([], "A", 5)
        self.assertEqual(result, 0.0)

    def test_precision_at_k_k_zero(self):
        """k=0 should return 0.0."""
        result = precision_at_k(["A", "B", "C"], "A", 0)
        self.assertEqual(result, 0.0)

    def test_precision_at_k_k_negative(self):
        """k<0 should return 0.0."""
        result = precision_at_k(["A", "B", "C"], "A", -1)
        self.assertEqual(result, 0.0)

    def test_precision_at_k_k_larger_than_list(self):
        """k larger than list length should check entire list."""
        result = precision_at_k(["A", "B", "C"], "C", 100)
        self.assertAlmostEqual(result, 1.0 / 100, places=5)

    def test_precision_at_k_with_duplicates(self):
        """Duplicates in predicted list should be handled correctly."""
        result = precision_at_k(["A", "B", "A", "C"], "A", 2)
        self.assertEqual(result, 0.5)

    def test_precision_at_k_case_sensitive(self):
        """Matching should be case-sensitive."""
        result = precision_at_k(["A", "B", "C"], "a", 3)
        self.assertEqual(result, 0.0)


class TestMetricsIntegration(unittest.TestCase):
    """Integration tests for all metrics together."""

    def test_metrics_consistency_found_in_top_k(self):
        """When ground truth is in top-k, recall and precision should both be positive."""
        predicted = ["assumption_1", "assumption_2", "assumption_3"]
        ground_truth = "assumption_2"
        k = 2

        recall = recall_at_k(predicted, ground_truth, k)
        precision = precision_at_k(predicted, ground_truth, k)
        rank = mean_rank(predicted, ground_truth)

        self.assertEqual(recall, 1.0)
        self.assertEqual(precision, 0.5)
        self.assertEqual(rank, 2.0)

    def test_metrics_consistency_not_found(self):
        """When ground truth is not found, recall and precision should be 0."""
        predicted = ["assumption_1", "assumption_2", "assumption_3"]
        ground_truth = "assumption_4"
        k = 3

        recall = recall_at_k(predicted, ground_truth, k)
        precision = precision_at_k(predicted, ground_truth, k)
        rank = mean_rank(predicted, ground_truth)

        self.assertEqual(recall, 0.0)
        self.assertEqual(precision, 0.0)
        self.assertEqual(rank, 4.0)

    def test_metrics_with_real_assumption_strings(self):
        """Test with realistic assumption strings."""
        predicted = [
            "Transformers require positional encoding",
            "Attention is all you need",
            "Recurrence is necessary for sequence modeling",
            "Batch normalization improves convergence",
        ]
        ground_truth = "Attention is all you need"
        k = 2

        recall = recall_at_k(predicted, ground_truth, k)
        precision = precision_at_k(predicted, ground_truth, k)
        rank = mean_rank(predicted, ground_truth)

        self.assertEqual(recall, 1.0)
        self.assertEqual(precision, 0.5)
        self.assertEqual(rank, 2.0)


if __name__ == "__main__":
    unittest.main()
