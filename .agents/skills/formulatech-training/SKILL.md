---
name: formulatech-training
description: Use when working on FormulaTech model training, tyre-safety detectors, losses, optimizer or scheduler configuration, reproducibility, checkpoints, run directories, or experiments/configs.
---

# FormulaTech Training

For training work, read and follow:

- `../../../.github/skills/trainer.agent.md`
- `../../../.github/instructions/copilot-instructions.md`

Keep hyperparameters config-driven, preserve reproducibility settings, and save run
artifacts under `experiments/runs/<run_name>/`.

Treat training data, labels, run artifacts, logs, checkpoints, cached telemetry, and
notebook outputs as potentially sensitive or bulky. Do not print or commit raw
downloaded records, credentials, private identifiers, or checkpoints.

Do not modify data pipeline or evaluation logic as part of training work unless the
user explicitly asks for that broader change.

Use a test-driven workflow for training changes: config validation tests, tiny synthetic
dataset smoke tests, checkpoint/resume tests, and regression tests for fixed bugs.
Training code must support small, fast test configurations as well as full runs.

Do not expect high accuracy from tiny smoke tests. Use them to verify wiring, shapes,
loss computation, deterministic seeding, and output artifact creation.
