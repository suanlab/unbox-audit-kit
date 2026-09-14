#!/usr/bin/env python3
"""
Create sample paper data for testing the Unbox pipeline.
This creates 5 mock papers from 2014-2016 representing pre-Transformer research.
"""

import json
from pathlib import Path

SAMPLE_PAPERS = [
    {
        "paperId": "sample_001",
        "title": "Neural Machine Translation by Jointly Learning to Align and Translate",
        "authors": ["Dzmitry Bahdanau", "Kyunghyun Cho", "Yoshua Bengio"],
        "year": 2015,
        "abstract": "We propose a novel neural architecture that learns to align and translate jointly. The model uses a bidirectional RNN as an encoder and a decoder that attends over the source sequence during each decoding step. This attention mechanism allows the model to focus on relevant parts of the source sequence.",
        "venue": "ICLR",
        "citationCount": 15000,
    },
    {
        "paperId": "sample_002",
        "title": "Sequence to Sequence Learning with Neural Networks",
        "authors": ["Ilya Sutskever", "Oriol Vinyals", "Quoc V. Le"],
        "year": 2014,
        "abstract": "We present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses a multilayered LSTM to map the input sequence to a vector of fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector.",
        "venue": "NeurIPS",
        "citationCount": 20000,
    },
    {
        "paperId": "sample_003",
        "title": "Effective Approaches to Attention-based Neural Machine Translation",
        "authors": ["Minh-Thang Luong", "Hieu Pham", "Christopher D. Manning"],
        "year": 2015,
        "abstract": "We examine two simple and effective classes of attention mechanism: a global approach which always attends to all source positions, and a local approach which only attends to a subset of source positions. We test the effectiveness of our models on the WMT translation tasks.",
        "venue": "EMNLP",
        "citationCount": 8000,
    },
    {
        "paperId": "sample_004",
        "title": "Gated Feedback Recurrent Neural Networks",
        "authors": ["Junyoung Chung", "Caglar Gulcehre", "Kyunghyun Cho"],
        "year": 2015,
        "abstract": "We propose a novel recurrent neural network architecture that uses multiple layers and feedback connections. The model uses gating mechanisms to control the flow of information between layers, allowing it to capture complex patterns in sequential data.",
        "venue": "ICML",
        "citationCount": 500,
    },
    {
        "paperId": "sample_005",
        "title": "LSTM: A Search Space Odyssey",
        "authors": ["Klaus Greff", "Rupesh K. Srivastava", "Jan Koutnik"],
        "year": 2017,
        "abstract": "We present the first large-scale analysis of eight LSTM variants on three representative tasks. We find that none of the variants can improve upon the standard LSTM architecture significantly, suggesting that the standard LSTM is already well-suited for most sequence modeling tasks.",
        "venue": "IEEE",
        "citationCount": 3000,
    },
]


def main():
    output_dir = Path("data/transformer")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "papers.jsonl"

    with open(output_file, "w") as f:
        for paper in SAMPLE_PAPERS:
            f.write(json.dumps(paper) + "\n")

    print(f"Created {len(SAMPLE_PAPERS)} sample papers at {output_file}")
    print(f"Papers represent pre-Transformer era (2014-2016)")
    print(f"All papers assume 'recurrence is necessary for sequence modeling'")


if __name__ == "__main__":
    main()
