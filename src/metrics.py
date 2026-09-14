"""
Evaluation metrics for assumption ranking, retrieval, and prospective evaluation.

This module implements two categories of metrics:

RETROSPECTIVE METRICS (evaluating predicted assumptions against ground truth):
- Recall@K: Whether ground truth appears in top-K predictions
- Rank: Position of ground truth in predicted list
- Precision@K: Proportion of top-K predictions that are relevant

PROSPECTIVE METRICS (evaluating quality of discovered assumptions):
- novelty_score: Average novelty rating from expert evaluations
- feasibility_score: Average feasibility rating from expert evaluations
- impact_score: Average impact rating from expert evaluations
- transformational_score: Average transformational potential rating
- inter_rater_agreement: Krippendorff's alpha for inter-rater reliability
- compare_to_baseline: Statistical comparison of scores vs baseline
"""

from typing import Union
import statistics


def recall_at_k(predicted: list[str], ground_truth: str, k: int) -> float:
    """
    Compute Recall@K: whether ground truth appears in top-K predictions.

    Recall@K is a binary metric that returns 1.0 if the ground truth assumption
    appears anywhere in the top-K predictions, and 0.0 otherwise.

    Args:
        predicted: List of predicted assumption strings, ordered by relevance/confidence.
        ground_truth: The ground truth assumption string to find.
        k: The cutoff position (number of top predictions to consider).

    Returns:
        1.0 if ground_truth is in predicted[:k], 0.0 otherwise.

    Examples:
        >>> recall_at_k(["A", "B", "C"], "A", 2)
        1.0
        >>> recall_at_k(["A", "B", "C"], "C", 2)
        0.0
        >>> recall_at_k(["A", "B", "C"], "D", 3)
        0.0
    """
    if k <= 0:
        return 0.0

    top_k = predicted[:k]
    return 1.0 if ground_truth in top_k else 0.0


def mean_rank(predicted: list[str], ground_truth: str) -> float:
    """
    Compute the rank (1-indexed position) of ground truth in predicted list.

    The rank is the 1-indexed position where ground truth appears in the predicted
    list. If ground truth is not found, the rank is len(predicted) + 1.

    Args:
        predicted: List of predicted assumption strings, ordered by relevance/confidence.
        ground_truth: The ground truth assumption string to find.

    Returns:
        The 1-indexed position of ground_truth in predicted, or len(predicted) + 1
        if not found.

    Examples:
        >>> mean_rank(["A", "B", "C"], "A")
        1.0
        >>> mean_rank(["A", "B", "C"], "B")
        2.0
        >>> mean_rank(["A", "B", "C"], "C")
        3.0
        >>> mean_rank(["A", "B", "C"], "D")
        4.0
    """
    try:
        # Find 1-indexed position
        rank = predicted.index(ground_truth) + 1
        return float(rank)
    except ValueError:
        # Ground truth not found
        return float(len(predicted) + 1)


def precision_at_k(predicted: list[str], ground_truth: str, k: int) -> float:
    """
    Compute Precision@K: proportion of top-K predictions that are relevant.

    For single-label evaluation (one ground truth), Precision@K is equivalent to
    Recall@K (either the ground truth is in top-K or it isn't). This function
    is provided for consistency with multi-label scenarios and future extensions.

    Args:
        predicted: List of predicted assumption strings, ordered by relevance/confidence.
        ground_truth: The ground truth assumption string to find.
        k: The cutoff position (number of top predictions to consider).

    Returns:
        1.0 / k if ground_truth is in predicted[:k], 0.0 otherwise.

    Examples:
        >>> precision_at_k(["A", "B", "C"], "A", 2)
        0.5
        >>> precision_at_k(["A", "B", "C"], "B", 2)
        0.5
        >>> precision_at_k(["A", "B", "C"], "C", 3)
        0.3333333333333333
    """
    if k <= 0:
        return 0.0

    top_k = predicted[:k]
    if ground_truth in top_k:
        return 1.0 / k
    return 0.0


def novelty_score(expert_ratings: list[dict[str, int]]) -> float:
    """
    Compute average novelty score from expert evaluations.

    Aggregates novelty ratings (1-10 scale) from multiple experts into a single
    score representing the average perceived novelty of an assumption.

    Args:
        expert_ratings: List of dicts with 'novelty' key. Each dict represents
            one expert's evaluation. Example:
            [{"novelty": 8, ...}, {"novelty": 7, ...}]

    Returns:
        Float between 1.0 and 10.0 representing average novelty rating.
        Returns 0.0 if list is empty.

    Examples:
        >>> novelty_score([{"novelty": 8}, {"novelty": 7}])
        7.5
        >>> novelty_score([{"novelty": 9}])
        9.0
        >>> novelty_score([])
        0.0
    """
    if not expert_ratings:
        return 0.0

    ratings = [r.get("novelty", 0) for r in expert_ratings]
    return float(statistics.mean(ratings)) if ratings else 0.0


def feasibility_score(expert_ratings: list[dict[str, int]]) -> float:
    """
    Compute average feasibility score from expert evaluations.

    Aggregates feasibility ratings (1-10 scale) from multiple experts into a single
    score representing the average perceived feasibility of an assumption.

    Args:
        expert_ratings: List of dicts with 'feasibility' key. Each dict represents
            one expert's evaluation. Example:
            [{"feasibility": 6, ...}, {"feasibility": 7, ...}]

    Returns:
        Float between 1.0 and 10.0 representing average feasibility rating.
        Returns 0.0 if list is empty.

    Examples:
        >>> feasibility_score([{"feasibility": 6}, {"feasibility": 7}])
        6.5
        >>> feasibility_score([{"feasibility": 5}])
        5.0
        >>> feasibility_score([])
        0.0
    """
    if not expert_ratings:
        return 0.0

    ratings = [r.get("feasibility", 0) for r in expert_ratings]
    return float(statistics.mean(ratings)) if ratings else 0.0


def impact_score(expert_ratings: list[dict[str, int]]) -> float:
    """
    Compute average impact score from expert evaluations.

    Aggregates impact ratings (1-10 scale) from multiple experts into a single
    score representing the average perceived impact of an assumption.

    Args:
        expert_ratings: List of dicts with 'impact' key. Each dict represents
            one expert's evaluation. Example:
            [{"impact": 7, ...}, {"impact": 6, ...}]

    Returns:
        Float between 1.0 and 10.0 representing average impact rating.
        Returns 0.0 if list is empty.

    Examples:
        >>> impact_score([{"impact": 7}, {"impact": 6}])
        6.5
        >>> impact_score([{"impact": 8}])
        8.0
        >>> impact_score([])
        0.0
    """
    if not expert_ratings:
        return 0.0

    ratings = [r.get("impact", 0) for r in expert_ratings]
    return float(statistics.mean(ratings)) if ratings else 0.0


def transformational_score(expert_ratings: list[dict[str, int]]) -> float:
    """
    Compute average transformational potential score from expert evaluations.

    Aggregates transformational potential ratings (1-10 scale) from multiple experts
    into a single score representing the average perceived transformational potential
    of an assumption (ability to shift paradigms).

    Args:
        expert_ratings: List of dicts with 'transformational' key. Each dict represents
            one expert's evaluation. Example:
            [{"transformational": 9, ...}, {"transformational": 8, ...}]

    Returns:
        Float between 1.0 and 10.0 representing average transformational rating.
        Returns 0.0 if list is empty.

    Examples:
        >>> transformational_score([{"transformational": 9}, {"transformational": 8}])
        8.5
        >>> transformational_score([{"transformational": 7}])
        7.0
        >>> transformational_score([])
        0.0
    """
    if not expert_ratings:
        return 0.0

    ratings = [r.get("transformational", 0) for r in expert_ratings]
    return float(statistics.mean(ratings)) if ratings else 0.0


def inter_rater_agreement(ratings_list: list[list[Union[int, float]]]) -> float:
    """
    Compute inter-rater agreement using Krippendorff's alpha.

    Measures agreement among multiple raters on the same set of items.
    Krippendorff's alpha ranges from -1 (perfect disagreement) to 1 (perfect agreement),
    with 0 indicating chance agreement.

    Args:
        ratings_list: List of rating lists, where each inner list contains ratings
            from one rater for all items. All inner lists must have same length.
            Example: [[8, 7, 9], [7, 8, 8], [8, 6, 9]] (3 raters, 3 items)

    Returns:
        Float between -1.0 and 1.0 representing Krippendorff's alpha.
        Returns 0.0 if input is invalid (empty, single rater, or inconsistent lengths).

    Examples:
        >>> inter_rater_agreement([[8, 7, 9], [7, 8, 8], [8, 6, 9]])
        -0.125
        >>> inter_rater_agreement([[5, 5], [5, 5]])
        1.0
        >>> inter_rater_agreement([])
        0.0
    """
    if not ratings_list or len(ratings_list) < 2:
        return 0.0

    lengths = [len(r) for r in ratings_list]
    if len(set(lengths)) > 1:
        return 0.0

    num_items = lengths[0]
    if num_items == 0:
        return 0.0

    num_raters = len(ratings_list)

    pairable_items = 0
    agreement_count = 0

    for item_idx in range(num_items):
        ratings_for_item = [
            ratings_list[rater_idx][item_idx] for rater_idx in range(num_raters)
        ]

        for i in range(num_raters):
            for j in range(i + 1, num_raters):
                pairable_items += 1
                if ratings_for_item[i] == ratings_for_item[j]:
                    agreement_count += 1

    if pairable_items == 0:
        return 0.0

    observed_agreement = agreement_count / pairable_items

    all_ratings = [r for rater_ratings in ratings_list for r in rater_ratings]
    expected_agreement = 0.0
    unique_values = set(all_ratings)

    for value in unique_values:
        count = all_ratings.count(value)
        prob = count / len(all_ratings)
        expected_agreement += prob * prob

    if expected_agreement >= 1.0:
        return 1.0 if observed_agreement == 1.0 else 0.0

    alpha = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
    return alpha


def compare_to_baseline(
    our_scores: dict[str, float], baseline_scores: dict[str, float]
) -> dict[str, dict[str, Union[float, int]]]:
    """
    Compare Unbox evaluation scores against baseline scores.

    Computes mean differences and effect sizes for each metric dimension.
    Provides statistical summary of how Unbox scores compare to baseline.

    Args:
        our_scores: Dict with metric names as keys and scores as values.
            Example: {"novelty": 7.5, "feasibility": 6.2, "impact": 7.1}
        baseline_scores: Dict with same metric names and baseline scores.
            Example: {"novelty": 6.0, "feasibility": 6.5, "impact": 5.8}

    Returns:
        Dict with comparison results for each metric:
        {
            "novelty": {
                "our_score": 7.5,
                "baseline_score": 6.0,
                "difference": 1.5,
                "percent_improvement": 25.0
            },
            ...
        }

    Examples:
        >>> compare_to_baseline(
        ...     {"novelty": 8.0, "feasibility": 6.0},
        ...     {"novelty": 6.0, "feasibility": 6.5}
        ... )
        {'novelty': {'our_score': 8.0, 'baseline_score': 6.0, 'difference': 2.0, 'percent_improvement': 33.33}, 'feasibility': {'our_score': 6.0, 'baseline_score': 6.5, 'difference': -0.5, 'percent_improvement': -7.69}}
    """
    results = {}

    for metric_name in our_scores:
        if metric_name not in baseline_scores:
            continue

        our_score = our_scores[metric_name]
        baseline_score = baseline_scores[metric_name]
        difference = our_score - baseline_score

        if baseline_score != 0:
            percent_improvement = (difference / baseline_score) * 100
        else:
            percent_improvement = 0.0 if difference == 0 else float("inf")

        results[metric_name] = {
            "our_score": our_score,
            "baseline_score": baseline_score,
            "difference": difference,
            "percent_improvement": round(percent_improvement, 2),
        }

    return results
