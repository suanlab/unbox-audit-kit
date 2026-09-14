# Cross-LLM Comparison (Primary 4 Paradigms)

| Model | Transformer | Diffusion | ICL | ViT | R@5 | R@10 | Avg Sim |
|-------|-------------|-----------|-----|-----|-----|------|---------|
| gpt-4o | ✓ (rank 1, sim 0.674) | ✓ (rank 2, sim 0.759) | ✓ (rank 3, sim 0.702) | ✓ (rank 5, sim 0.676) | 4/4 | 4/4 | 0.703 |
| claude | ✗ (sim 0.625) | ✓ (rank 1, sim 0.738) | ✗ (sim 0.571) | ✓ (rank 11, sim 0.698) | 1/4 | 1/4 | 0.658 |
| llama-8b | ✓ (rank 10, sim 0.674) | ✗ (sim 0.521) | ✗ (sim 0.392) | ✓ (rank 5, sim 0.664) | 1/4 | 2/4 | 0.563 |
| qwen-7b | ✗ (sim 0.645) | ✓ (rank 9, sim 0.693) | ✓ (rank 5, sim 0.764) | ✓ (rank 1, sim 0.718) | 2/4 | 3/4 | 0.705 |
| mistral-7b | ✗ (sim 0.610) | ✗ (sim 0.588) | ✗ (sim 0.500) | ✗ (sim 0.625) | 0/4 | 0/4 | 0.581 |

## LaTeX-ready rows for tab:cross_llm
```latex
gpt-4o                    & \checkmark & \checkmark & \checkmark & \checkmark & 4/4 & 0.703 \\
claude                    & \texttimes & \checkmark & \texttimes & \texttimes & 1/4 & 0.658 \\
llama-8b                  & \texttimes & \texttimes & \texttimes & \checkmark & 1/4 & 0.563 \\
qwen-7b                   & \texttimes & \texttimes & \checkmark & \checkmark & 2/4 & 0.705 \\
mistral-7b                & \texttimes & \texttimes & \texttimes & \texttimes & 0/4 & 0.581 \\
```

## Sources
- **gpt-4o**: OK  (loaded)
- **claude**: OK  (loaded)
- **llama-8b**: OK  (loaded)
- **qwen-7b**: OK  (loaded)
- **mistral-7b**: OK  (loaded)
