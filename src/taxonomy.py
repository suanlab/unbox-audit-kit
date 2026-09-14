"""
Assumption taxonomy data structure for ML model assumptions.

Defines a hierarchical taxonomy of assumptions across 5 top-level categories,
each with 3-5 subcategories. Provides utilities for classification and retrieval.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TopCategory(Enum):
    """Top-level assumption categories."""

    ARCHITECTURAL = "Architectural"
    TRAINING = "Training"
    DATA = "Data"
    THEORETICAL = "Theoretical"
    EVALUATION = "Evaluation"


@dataclass
class SubcategoryInfo:
    """Information about a subcategory."""

    name: str
    description: str
    example_assumptions: list[str]


@dataclass
class CategoryInfo:
    """Information about a top-level category."""

    name: str
    description: str
    subcategories: dict[str, SubcategoryInfo]


# Define the complete assumption taxonomy
ASSUMPTION_CATEGORIES: dict[str, CategoryInfo] = {
    "Architectural": CategoryInfo(
        name="Architectural",
        description="Assumptions about model structure, components, and connectivity",
        subcategories={
            "model_structure": SubcategoryInfo(
                name="Model Structure",
                description="Assumptions about the overall architecture and design of the model",
                example_assumptions=[
                    "The model uses a feedforward neural network architecture",
                    "The model employs attention mechanisms for sequence processing",
                    "The model has a hierarchical structure with multiple levels",
                ],
            ),
            "component_design": SubcategoryInfo(
                name="Component Design",
                description="Assumptions about individual components and their properties",
                example_assumptions=[
                    "Convolutional layers are used for spatial feature extraction",
                    "Recurrent units capture temporal dependencies",
                    "Embedding layers provide meaningful vector representations",
                ],
            ),
            "connectivity": SubcategoryInfo(
                name="Connectivity",
                description="Assumptions about how components are connected and interact",
                example_assumptions=[
                    "Skip connections improve gradient flow",
                    "Residual connections enable training of very deep networks",
                    "Cross-attention enables information flow between modalities",
                ],
            ),
            "parameter_sharing": SubcategoryInfo(
                name="Parameter Sharing",
                description="Assumptions about weight sharing and parameter reuse",
                example_assumptions=[
                    "Shared weights reduce model complexity",
                    "Convolutional filters are translation-invariant",
                    "Tied embeddings improve generalization",
                ],
            ),
        },
    ),
    "Training": CategoryInfo(
        name="Training",
        description="Assumptions about optimization, learning procedures, and regularization",
        subcategories={
            "optimization": SubcategoryInfo(
                name="Optimization",
                description="Assumptions about the optimization algorithm and convergence",
                example_assumptions=[
                    "Stochastic gradient descent converges to good local minima",
                    "Adam optimizer adapts learning rates effectively",
                    "Momentum helps escape saddle points",
                ],
            ),
            "learning_procedure": SubcategoryInfo(
                name="Learning Procedure",
                description="Assumptions about the training process and schedule",
                example_assumptions=[
                    "Learning rate scheduling improves convergence",
                    "Warmup phases stabilize training",
                    "Curriculum learning improves sample efficiency",
                ],
            ),
            "regularization": SubcategoryInfo(
                name="Regularization",
                description="Assumptions about regularization techniques and their effects",
                example_assumptions=[
                    "Dropout prevents overfitting by reducing co-adaptation",
                    "L2 regularization encourages smaller weights",
                    "Batch normalization acts as a regularizer",
                ],
            ),
            "loss_function": SubcategoryInfo(
                name="Loss Function",
                description="Assumptions about the loss function and its properties",
                example_assumptions=[
                    "Cross-entropy loss is appropriate for classification",
                    "MSE loss is suitable for regression tasks",
                    "Contrastive loss learns meaningful embeddings",
                ],
            ),
            "initialization": SubcategoryInfo(
                name="Initialization",
                description="Assumptions about weight initialization strategies",
                example_assumptions=[
                    "Xavier initialization maintains activation variance",
                    "He initialization is appropriate for ReLU networks",
                    "Proper initialization accelerates convergence",
                ],
            ),
        },
    ),
    "Data": CategoryInfo(
        name="Data",
        description="Assumptions about data requirements, labeling, and quality",
        subcategories={
            "data_requirements": SubcategoryInfo(
                name="Data Requirements",
                description="Assumptions about the amount and type of data needed",
                example_assumptions=[
                    "The model requires at least 10,000 labeled examples",
                    "Data should be representative of the target distribution",
                    "Balanced class distribution improves performance",
                ],
            ),
            "labeling": SubcategoryInfo(
                name="Labeling",
                description="Assumptions about label quality and annotation process",
                example_assumptions=[
                    "Labels are accurate and consistent",
                    "Annotators have sufficient expertise",
                    "Inter-annotator agreement is above 0.8",
                ],
            ),
            "data_quality": SubcategoryInfo(
                name="Data Quality",
                description="Assumptions about data cleanliness and preprocessing",
                example_assumptions=[
                    "Missing values are handled appropriately",
                    "Outliers are detected and managed",
                    "Data is normalized to a standard range",
                ],
            ),
            "distribution": SubcategoryInfo(
                name="Distribution",
                description="Assumptions about data distribution and stationarity",
                example_assumptions=[
                    "Training and test distributions are similar",
                    "Data distribution is stationary over time",
                    "Features are independently distributed",
                ],
            ),
        },
    ),
    "Theoretical": CategoryInfo(
        name="Theoretical",
        description="Assumptions about mathematical properties, convergence, and generalization",
        subcategories={
            "convergence": SubcategoryInfo(
                name="Convergence",
                description="Assumptions about convergence properties and guarantees",
                example_assumptions=[
                    "The loss function is convex",
                    "Gradient descent converges to a global minimum",
                    "Convergence rate is polynomial in problem size",
                ],
            ),
            "generalization": SubcategoryInfo(
                name="Generalization",
                description="Assumptions about generalization bounds and performance",
                example_assumptions=[
                    "The model generalizes from training to test data",
                    "Generalization gap is bounded by model complexity",
                    "Regularization improves generalization",
                ],
            ),
            "expressiveness": SubcategoryInfo(
                name="Expressiveness",
                description="Assumptions about model capacity and representational power",
                example_assumptions=[
                    "The model is expressive enough to learn the target function",
                    "Universal approximation theorem applies",
                    "Depth increases representational capacity",
                ],
            ),
            "mathematical_properties": SubcategoryInfo(
                name="Mathematical Properties",
                description="Assumptions about mathematical characteristics of the model",
                example_assumptions=[
                    "The loss landscape has benign structure",
                    "Gradients are Lipschitz continuous",
                    "The model satisfies smoothness conditions",
                ],
            ),
        },
    ),
    "Evaluation": CategoryInfo(
        name="Evaluation",
        description="Assumptions about metrics, benchmarks, and validation procedures",
        subcategories={
            "metrics": SubcategoryInfo(
                name="Metrics",
                description="Assumptions about evaluation metrics and their appropriateness",
                example_assumptions=[
                    "Accuracy is the appropriate metric for this task",
                    "F1 score balances precision and recall",
                    "AUC-ROC is suitable for imbalanced datasets",
                ],
            ),
            "benchmarks": SubcategoryInfo(
                name="Benchmarks",
                description="Assumptions about benchmark datasets and comparisons",
                example_assumptions=[
                    "The benchmark is representative of real-world scenarios",
                    "Baseline methods are properly implemented",
                    "Comparison is fair across different approaches",
                ],
            ),
            "validation": SubcategoryInfo(
                name="Validation",
                description="Assumptions about validation procedures and protocols",
                example_assumptions=[
                    "Cross-validation provides reliable estimates",
                    "Train-test split is random and stratified",
                    "Validation set is independent of training",
                ],
            ),
            "statistical_significance": SubcategoryInfo(
                name="Statistical Significance",
                description="Assumptions about statistical testing and significance",
                example_assumptions=[
                    "Performance differences are statistically significant",
                    "Confidence intervals are properly computed",
                    "Multiple comparison corrections are applied",
                ],
            ),
        },
    ),
}


def get_category_hierarchy() -> dict[str, CategoryInfo]:
    """
    Retrieve the complete assumption taxonomy hierarchy.

    Returns:
        Dict mapping category names to CategoryInfo objects containing
        subcategories and their descriptions.
    """
    return ASSUMPTION_CATEGORIES


def classify_assumption(assumption_text: str) -> tuple[str | None, str | None]:
    """
    Classify an assumption into a top-level category and subcategory.

    Uses simple keyword matching to classify assumptions. Returns the best match
    based on keyword overlap with example assumptions and descriptions.

    Args:
        assumption_text: The assumption text to classify

    Returns:
        Tuple of (top_category, subcategory) or (None, None) if no match found.
        Both are strings representing the category/subcategory names.
    """
    if not assumption_text or not assumption_text.strip():
        return None, None

    assumption_lower = assumption_text.lower()
    best_match = (None, None)
    best_score = 0

    # Search through all categories and subcategories
    for top_cat_name, cat_info in ASSUMPTION_CATEGORIES.items():
        for _, subcat_info in cat_info.subcategories.items():
            # Calculate match score based on keyword overlap
            score = 0

            # Check against subcategory name
            if subcat_info.name.lower() in assumption_lower:
                score += 3

            # Check against description keywords
            desc_words = subcat_info.description.lower().split()
            for word in desc_words:
                if len(word) > 3 and word in assumption_lower:
                    score += 1

            # Check against example assumptions
            for example in subcat_info.example_assumptions:
                example_lower = example.lower()
                # Count matching words
                example_words = example_lower.split()
                for word in example_words:
                    if len(word) > 3 and word in assumption_lower:
                        score += 1

            # Update best match if this is better
            if score > best_score:
                best_score = score
                best_match = (top_cat_name, subcat_info.name)

    # Only return a match if we found some keyword overlap
    if best_score > 0:
        return best_match

    return None, None


def get_subcategories(top_category: str) -> dict[str, SubcategoryInfo] | None:
    """
    Get all subcategories for a given top-level category.

    Args:
        top_category: Name of the top-level category

    Returns:
        Dictionary of subcategories or None if category not found.
    """
    if top_category in ASSUMPTION_CATEGORIES:
        return ASSUMPTION_CATEGORIES[top_category].subcategories
    return None


def get_category_info(top_category: str) -> CategoryInfo | None:
    """
    Get detailed information about a top-level category.

    Args:
        top_category: Name of the top-level category

    Returns:
        CategoryInfo object or None if category not found.
    """
    return ASSUMPTION_CATEGORIES.get(top_category)


def list_all_categories() -> list[str]:
    """
    Get a list of all top-level category names.

    Returns:
        List of category names in order.
    """
    return list(ASSUMPTION_CATEGORIES.keys())
