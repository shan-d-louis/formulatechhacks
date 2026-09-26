---
name: formulatech-evaluation
description: Use when working on FormulaTech tyre-safety evaluation metrics, detector comparison plots, calibration/coverage, model comparisons, reports/figures, evaluation code, or evaluator workflows.
---

# FormulaTech Evaluation

For evaluation work, read and follow:

- `../../../.github/skills/evaluator.agent.md`
- `../../../.github/instructions/copilot-instructions.md`

Evaluation must be consistent across model variants and honest about evidence level.
Always preserve required metrics, confidence/coverage expectations, and output file
conventions from the source instructions or the current README.

For tyre-safety detectors, report precision/recall/F1 against rule labels or reviewed
labels, false alarms per lap where relevant, and latency for live alerts. For
laps-to-cliff or remaining-life models, report lap error and empirical coverage when
using conformal or quantile bounds. For hazard/risk models, report ranking or survival
metrics suitable to the implementation. Keep paired samples and splits identical for
comparisons.

Treat evaluation inputs, labels, plots, reports, notebook outputs, cached telemetry, and
run artifacts as potentially sensitive or bulky. Report aggregate metrics and
anonymized examples; do not disclose credentials, private identifiers, or unnecessary
raw downloaded records.

Use test-driven development for metric code and report generation. Add regression tests
for metric edge cases, missing signals, and user-facing wording such as "possible
lock-up" versus confirmed events.

Do not train models, alter checkpoints, or report training-split metrics as model
performance.
