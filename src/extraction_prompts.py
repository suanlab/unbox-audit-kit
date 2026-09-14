PAPER_SPECIFIC_PROMPT = """You are an expert at identifying implicit assumptions in academic papers.

Read the following paper text and identify the implicit assumptions (assumptions not explicitly stated but underlying the work).

Focus on these categories:
- Architectural assumptions: Design choices, model structure assumptions
- Training assumptions: Data distribution, optimization assumptions
- Data assumptions: Dataset characteristics, preprocessing assumptions
- Theoretical assumptions: Mathematical foundations, theoretical constraints
- Evaluation assumptions: Benchmark selection, metric assumptions

For each assumption:
1. State it clearly and concisely
2. Provide a confidence score (0.0-1.0) based on how strongly the text implies this assumption
3. Categorize it into one of the above categories

Return ONLY valid JSON with no additional text. Limit to top 10 assumptions.

Format:
{{
  "assumptions": [
    {{
      "assumption": "string describing the assumption",
      "confidence": 0.0-1.0,
      "category": "architectural|training|data|theoretical|evaluation"
    }}
  ]
}}

Paper text:
{paper_text}
"""

FIELD_WIDE_PROMPT = """You are an expert at identifying the foundational, field-wide assumptions that academic papers inherit from their broader research community.

Your task is NOT to find this specific paper's design choices. Instead, identify the **paradigmatic assumptions** — beliefs shared across the entire subfield that this paper takes for granted without questioning. These are assumptions so deeply ingrained that the research community treats them as self-evident truths.

Examples of paradigmatic assumptions (the kind you should find):
- "Sequential processing requires recurrence" (pre-Transformer seq2seq field)
- "High-quality image generation requires adversarial training" (pre-Diffusion generative modeling)
- "Task adaptation requires fine-tuning model weights" (pre-GPT-3 transfer learning)
- "Visual feature learning requires convolutional inductive biases" (pre-ViT computer vision)

These are NOT what we want:
- Paper-specific implementation details ("we use Adam optimizer with lr=0.001")
- Narrow technical choices ("batch size of 32 is sufficient")
- Obvious truisms ("more data helps")

For each assumption, phrase it as a general declarative statement about what the field believes is NECESSARY, REQUIRED, or ESSENTIAL. Use the pattern: "[X] is necessary/required/essential for [Y]".

Categories:
- architectural: Structural requirements (e.g., "recurrence is needed for sequences")
- training: Learning procedure requirements (e.g., "adversarial training is needed for generation")
- data: Data requirements (e.g., "labeled data is essential for classification")
- theoretical: Theoretical constraints (e.g., "bias-variance tradeoff always holds")
- evaluation: Evaluation paradigm assumptions (e.g., "accuracy is the right metric")

Return ONLY valid JSON. Limit to top 10 assumptions, ranked by how foundational they are to the entire field (not just this paper).

Format:
{{
  "assumptions": [
    {{
      "assumption": "string describing the field-wide assumption",
      "confidence": 0.0-1.0,
      "category": "architectural|training|data|theoretical|evaluation"
    }}
  ]
}}

Paper text:
{paper_text}
"""
