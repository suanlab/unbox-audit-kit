"""
Test script for assumption extraction functionality.

Tests the extract_assumptions function with sample paper text and mock API responses.
No real API calls are made - uses mocking to validate structure and behavior.
"""

import json
import importlib
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

extraction_module = importlib.import_module("src.extraction")
sys.modules["extraction"] = extraction_module
extract_assumptions = extraction_module.extract_assumptions


# Sample paper text from "Neural Machine Translation by Jointly Learning to Align and Translate"
# (Bahdanau et al., 2015)
SAMPLE_PAPER_TEXT = """
Neural Machine Translation by Jointly Learning to Align and Translate

Abstract:
Neural machine translation is a recently proposed approach to machine translation that encodes a source sentence into a fixed-length context vector and decodes the vector into a target sentence. Despite the recent success in applying neural networks to machine translation, it is believed that the performance of a basic encoder-decoder architecture quickly degrades as the length of an input sentence increases. Here, we conjecture that the use of a fixed-length context vector is a bottleneck in improving the performance of this basic encoder-decoder architecture, and propose a novel attention mechanism to address this limitation. We propose an extension to the encoder-decoder model which learns to align and translate jointly. Each time the proposed model generates a word in a translation, it searches for a set of positions in a source sentence where the most relevant information is concentrated. The model then predicts a target word based on the context vectors associated with these source positions and all the previous generated target words. The proposed approach enables the use of a much longer source sentence. The experimental results on the English-French translation tasks show that the proposed approach significantly outperforms the basic encoder-decoder approach. This is the first time that a single sequence-to-sequence model has been shown to match the performance of the conventional phrase-based statistical machine translation system in large-scale tasks.

Introduction:
Neural machine translation (NMT) is a new approach to machine translation that is based purely on neural networks. Unlike the conventional statistical machine translation system which consists of many small sub-components that are tuned separately, neural machine translation attempts to build and train a single, large neural network that reads a source sentence and outputs a correct translation.

The encoder-decoder architecture with recurrent neural networks (RNNs) has been recently proposed as an approach to sequence-to-sequence learning and has shown promising results on relatively short sentences. However, a significant problem with this architecture is that the performance quickly degrades as the length of an input sentence increases. We conjecture that the use of a fixed-length context vector is a bottleneck in improving the performance of this basic encoder-decoder architecture.

Methods:
The encoder reads an input sequence of vectors x = (x1, ..., xT) and encodes it into a context vector c. The decoder then generates an output sequence y = (y1, ..., yT') based on the context vector c. Each output yt is computed based on a recurrent hidden state st, the previously generated word yt-1, and the context vector c.

The fixed-length context vector is a bottleneck in this architecture. To address this, we propose to use an attention mechanism that allows the decoder to search over the source sentence when generating each target word. The attention mechanism computes a set of attention weights αt,i for each source position i when generating the t-th target word.

Results:
We evaluate our approach on the English-French translation task using the WMT14 dataset. The proposed attention-based model significantly outperforms the baseline encoder-decoder model, especially on longer sentences. The model achieves a BLEU score of 34.81 on the test set, compared to 30.69 for the baseline.
"""

# Mock API response with valid assumptions
MOCK_API_RESPONSE = {
    "assumptions": [
        {
            "assumption": "Fixed-length context vectors are sufficient for encoding variable-length source sentences",
            "confidence": 0.95,
            "category": "architectural",
        },
        {
            "assumption": "Recurrent neural networks (RNNs) are the appropriate architecture for sequence-to-sequence learning",
            "confidence": 0.92,
            "category": "architectural",
        },
        {
            "assumption": "Source sentence length is the primary factor limiting encoder-decoder performance",
            "confidence": 0.88,
            "category": "theoretical",
        },
        {
            "assumption": "Attention weights should be computed independently for each target position",
            "confidence": 0.85,
            "category": "architectural",
        },
        {
            "assumption": "BLEU score is a reliable metric for evaluating machine translation quality",
            "confidence": 0.80,
            "category": "evaluation",
        },
        {
            "assumption": "WMT14 dataset is representative of real-world translation tasks",
            "confidence": 0.75,
            "category": "data",
        },
        {
            "assumption": "Gradient descent optimization is sufficient for training attention-based models",
            "confidence": 0.82,
            "category": "training",
        },
        {
            "assumption": "Alignment between source and target words can be learned implicitly through attention",
            "confidence": 0.90,
            "category": "theoretical",
        },
    ]
}


class TestExtractionStructure(unittest.TestCase):
    """Test the structure and validation of extracted assumptions."""

    @patch("extraction.anthropic.Anthropic")
    def test_extraction_returns_list(self, mock_anthropic_class):
        """Test that extract_assumptions returns a list."""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(MOCK_API_RESPONSE))]
        mock_client.messages.create.return_value = mock_message

        # Call function
        result = extract_assumptions(SAMPLE_PAPER_TEXT)

        # Verify
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    @patch("extraction.anthropic.Anthropic")
    def test_each_assumption_is_dict(self, mock_anthropic_class):
        """Test that each assumption is a dictionary."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(MOCK_API_RESPONSE))]
        mock_client.messages.create.return_value = mock_message

        result = extract_assumptions(SAMPLE_PAPER_TEXT)

        for assumption in result:
            self.assertIsInstance(assumption, dict)

    @patch("extraction.anthropic.Anthropic")
    def test_assumption_has_required_fields(self, mock_anthropic_class):
        """Test that each assumption has required fields: assumption, confidence, category."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(MOCK_API_RESPONSE))]
        mock_client.messages.create.return_value = mock_message

        result = extract_assumptions(SAMPLE_PAPER_TEXT)

        required_fields = {"assumption", "confidence", "category"}
        for assumption in result:
            self.assertTrue(
                required_fields.issubset(assumption.keys()),
                f"Assumption missing required fields: {assumption}",
            )

    @patch("extraction.anthropic.Anthropic")
    def test_assumption_field_is_string(self, mock_anthropic_class):
        """Test that 'assumption' field is a string."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(MOCK_API_RESPONSE))]
        mock_client.messages.create.return_value = mock_message

        result = extract_assumptions(SAMPLE_PAPER_TEXT)

        for assumption in result:
            self.assertIsInstance(assumption["assumption"], str)
            self.assertGreater(len(assumption["assumption"]), 0)

    @patch("extraction.anthropic.Anthropic")
    def test_confidence_is_float_in_range(self, mock_anthropic_class):
        """Test that 'confidence' is a float between 0.0 and 1.0."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(MOCK_API_RESPONSE))]
        mock_client.messages.create.return_value = mock_message

        result = extract_assumptions(SAMPLE_PAPER_TEXT)

        for assumption in result:
            confidence = assumption["confidence"]
            self.assertIsInstance(confidence, (int, float))
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    @patch("extraction.anthropic.Anthropic")
    def test_category_is_valid(self, mock_anthropic_class):
        """Test that 'category' is one of the valid categories."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(MOCK_API_RESPONSE))]
        mock_client.messages.create.return_value = mock_message

        result = extract_assumptions(SAMPLE_PAPER_TEXT)

        valid_categories = {
            "architectural",
            "training",
            "data",
            "theoretical",
            "evaluation",
        }
        for assumption in result:
            category = assumption["category"]
            self.assertIsInstance(category, str)
            self.assertIn(
                category,
                valid_categories,
                f"Invalid category: {category}. Must be one of {valid_categories}",
            )

    @patch("extraction.anthropic.Anthropic")
    def test_respects_max_10_assumptions(self, mock_anthropic_class):
        """Test that function returns at most 10 assumptions."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Create response with 15 assumptions
        large_response = {
            "assumptions": [
                {
                    "assumption": f"Assumption {i}",
                    "confidence": 0.5 + (i * 0.01),
                    "category": [
                        "architectural",
                        "training",
                        "data",
                        "theoretical",
                        "evaluation",
                    ][i % 5],
                }
                for i in range(15)
            ]
        }

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(large_response))]
        mock_client.messages.create.return_value = mock_message

        result = extract_assumptions(SAMPLE_PAPER_TEXT)

        self.assertLessEqual(len(result), 10)

    @patch("extraction.anthropic.Anthropic")
    def test_invalid_json_raises_error(self, mock_anthropic_class):
        """Test that invalid JSON response raises ValueError."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="This is not valid JSON")]
        mock_client.messages.create.return_value = mock_message

        with self.assertRaises(ValueError) as context:
            extract_assumptions(SAMPLE_PAPER_TEXT)

        self.assertIn("Failed to parse API response as JSON", str(context.exception))

    @patch("extraction.anthropic.Anthropic")
    def test_missing_assumptions_key_raises_error(self, mock_anthropic_class):
        """Test that response missing 'assumptions' key raises ValueError."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps({"results": []}))]
        mock_client.messages.create.return_value = mock_message

        with self.assertRaises(ValueError) as context:
            extract_assumptions(SAMPLE_PAPER_TEXT)

        self.assertIn("missing 'assumptions' key", str(context.exception))

    @patch("extraction.anthropic.Anthropic")
    def test_missing_required_field_raises_error(self, mock_anthropic_class):
        """Test that assumption missing required field raises ValueError."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        invalid_response = {
            "assumptions": [
                {
                    "assumption": "Some assumption",
                    "confidence": 0.8,
                    # Missing 'category'
                }
            ]
        }

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(invalid_response))]
        mock_client.messages.create.return_value = mock_message

        with self.assertRaises(ValueError) as context:
            extract_assumptions(SAMPLE_PAPER_TEXT)

        self.assertIn("missing required fields", str(context.exception))

    @patch("extraction.anthropic.Anthropic")
    def test_invalid_confidence_type_raises_error(self, mock_anthropic_class):
        """Test that non-numeric confidence raises ValueError."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        invalid_response = {
            "assumptions": [
                {
                    "assumption": "Some assumption",
                    "confidence": "high",  # Should be numeric
                    "category": "architectural",
                }
            ]
        }

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(invalid_response))]
        mock_client.messages.create.return_value = mock_message

        with self.assertRaises(ValueError) as context:
            extract_assumptions(SAMPLE_PAPER_TEXT)

        self.assertIn("Confidence must be numeric", str(context.exception))

    @patch("extraction.anthropic.Anthropic")
    def test_confidence_out_of_range_raises_error(self, mock_anthropic_class):
        """Test that confidence outside [0, 1] raises ValueError."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        invalid_response = {
            "assumptions": [
                {
                    "assumption": "Some assumption",
                    "confidence": 1.5,  # Out of range
                    "category": "architectural",
                }
            ]
        }

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(invalid_response))]
        mock_client.messages.create.return_value = mock_message

        with self.assertRaises(ValueError) as context:
            extract_assumptions(SAMPLE_PAPER_TEXT)

        self.assertIn("between 0.0 and 1.0", str(context.exception))

    @patch("extraction.anthropic.Anthropic")
    def test_api_error_propagates(self, mock_anthropic_class):
        """Test that API errors are properly propagated."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("API connection failed")

        with self.assertRaises(RuntimeError):
            extract_assumptions(SAMPLE_PAPER_TEXT)

    @patch("extraction.anthropic.Anthropic")
    def test_sample_paper_extraction(self, mock_anthropic_class):
        """Integration test: Extract assumptions from sample paper."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(MOCK_API_RESPONSE))]
        mock_client.messages.create.return_value = mock_message

        result = extract_assumptions(SAMPLE_PAPER_TEXT)

        # Verify we got assumptions
        self.assertEqual(len(result), 8)

        # Verify structure
        for assumption in result:
            self.assertIn("assumption", assumption)
            self.assertIn("confidence", assumption)
            self.assertIn("category", assumption)

        # Verify we have assumptions from different categories
        categories = {a["category"] for a in result}
        self.assertGreater(len(categories), 1)

        # Verify we have RNN/recurrence assumption (key to this paper)
        assumptions_text = " ".join([a["assumption"] for a in result])
        self.assertIn("Recurrent", assumptions_text)


def print_sample_output():
    """Print sample output for manual inspection."""
    print("\n" + "=" * 80)
    print("SAMPLE EXTRACTION OUTPUT")
    print("=" * 80)
    print(json.dumps(MOCK_API_RESPONSE, indent=2))
    print("\n" + "=" * 80)
    print("SAMPLE PAPER TEXT (first 500 chars)")
    print("=" * 80)
    print(SAMPLE_PAPER_TEXT[:500] + "...")


if __name__ == "__main__":
    # Print sample output for reference
    print_sample_output()

    # Run tests
    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80 + "\n")

    unittest.main(verbosity=2)
