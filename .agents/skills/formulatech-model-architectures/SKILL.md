---
name: formulatech-model-architectures
description: Use when creating or modifying FormulaTech tyre-safety model architectures, detector forward methods, model tensor shapes, sequence models, or files under model/model-like source directories.
---

# FormulaTech Model Architectures

For model architecture work, read and follow:

- `../../../.github/instructions/copilot-instructions.md`

Prefer simple, explainable models for the hackathon safety diagnosis surface unless a
more complex model is justified by tests and available data. Sequence models should
document expected input shape, sampling rate, units, and whether each feature is
measured, simulated, estimated, or inferred.

Keep loss functions and training orchestration out of model architecture files. Document
model input/output shapes in `forward()` docstrings or equivalent public methods.

Use test-driven development for shape contracts, detector thresholds, confidence
wording, and explanation outputs. When testing model shapes, use synthetic telemetry
unless the user explicitly asks for real data.

For public FastF1/OpenF1-only features, do not build a model that claims confirmed
wheel lock-up, tyre pressure, tyre temperature, or actual fuel mass. Those outputs must
be framed as possible anomalies or estimates unless direct source data is present.
