# AI2Pot Development Instructions

AI2Pot is a unified atomistic machine-learning framework and runtime built around a stable atomistic data interface, high-performance C++/CUDA operators, PyTorch-based model development, and consistent training/deployment workflows.

## Development priorities

When modifying AI2Pot, preserve these priorities in order:

1. Maintain architectural consistency and backward compatibility.
2. Preserve the AI2Pot Core Atomistic Interface.
3. Keep model-specific representations inside the model whenever possible.
4. Maintain compatibility between training and deployment paths.
5. Prefer reusable infrastructure over model-specific special cases.
6. Preserve CPU/GPU portability and performance.

Detailed architectural rules are defined under `.claude/rules/`.

Before making architectural changes, inspect the relevant rules and existing implementations rather than introducing a new convention.
