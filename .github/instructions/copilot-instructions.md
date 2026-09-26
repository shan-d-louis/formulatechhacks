# GitHub Copilot Instructions - FormulaTech Hacks

## Agent-Specific Instructions

These instructions are written for GitHub Copilot-style agents. Other coding agents
should first read their own native instruction directory or runtime-specific files,
then use `.github/skills/AGENTS.md` and `README.md` for shared repository context.

When adding or updating instructions for another agent, place them in that agent's own
directory or conventional file, and keep shared repository facts synchronized here,
in `.github/skills/AGENTS.md`, and in `README.md` when they affect users.

---

## Project context
This repository is for the FormulaTech Hacks SIDEWALL concept: an AI tyre-safety
digital twin and pit-wall monitor for motorsport safety diagnosis.

---

## Project Overview

This repository implements a software-first tyre safety diagnosis prototype. It uses
public motorsport telemetry, simulated per-wheel telemetry, and optional local demo
inputs to estimate tyre health, flag possible safety anomalies, and explain pit-wall
recommendations. The core pitch is that tyres are a richer diagnosis problem than a
simple acceleration anomaly because pressure, temperature, grip, wear, and driving
context change together over time.

---

## Domain Context

- **Input data**: Public FastF1/OpenF1 telemetry, race-control/status data, weather,
  tyre stint metadata, Kaggle F1 tyre strategy stint summaries, simulated telemetry,
  and optional higher-frequency per-wheel datasets when available.
- **Output**: Tyre health/risk indicators, possible lock-up/braking anomaly flags,
  overheating/cold-tyre/pressure-anomaly alerts, laps-to-cliff estimates, fatigue or
  stint-cap guidance, and explainable pit-wall calls.
- **Evidence labels**: Public FastF1 data does not include individual wheel rotation,
  actual fuel mass, tyre pressure, or tyre temperature in its standard telemetry
  channels. Anything inferred from public telemetry must be labelled as estimated,
  possible, or candidate unless a source directly measures the signal.

---

## Repository Structure

```
formulatechhacks/
├── frontend/
├── backend/
│   ├── model/
│   ├── view/
│   ├── controller/
│   ├── adaptors/
├── database/
├── scripts/            # For deployment or testing
├── tests/              # Unit tests
├── presentation/
└── README.md
└── .env                # Do not commit credentials. Only duplicate a .env.example without credentials if you wish to commit.
```

---

## Technology Stack

- **Language**: Python for data/model/backend work, plus frontend code when building
  the pit-wall, driver, crew, or atlas interfaces.
- **Environment**: Keep dependencies explicit in project configuration and prefer
  reproducible local commands.

---

## Test-Driven Development

- Start non-trivial functional changes by writing or updating a failing test that
  describes the intended behaviour.
- Keep the red-green-refactor loop small: add the narrowest test, implement the
  smallest useful change, run the focused test, then refactor with tests passing.
- Use synthetic or anonymized fixtures for telemetry, tyre states, and race context.
  Do not rely on raw downloaded datasets for unit tests.
- Add integration or end-to-end tests when work crosses source adapters, models,
  decision logic, backend APIs, or frontend workflows.
- For safety diagnosis logic, tests must assert both the alert decision and the
  explanatory evidence/reason string where practical.
- Every bug fix should include a regression test that would have failed before the
  fix.

---

## Documentation During Development

- Update agent docs, README files, schemas, config examples, and test coverage notes
  incrementally as behaviour changes. Do not leave documentation as a final cleanup
  task.
- Document whether tyre values are measured, simulated, estimated by the digital twin,
  or unavailable from the chosen data source.
- Keep user-facing copy honest: use "possible lock-up", "braking anomaly", or
  "candidate visual cue" unless wheel-speed or other direct evidence confirms the
  event.
- When adding a data source, model, alert, or pit-call rule, update the relevant docs
  with inputs, outputs, assumptions, limitations, and verification commands.
- After running tests with coverage, update the relevant `TEST_COVERAGE.md` file with
  the observed percentage and the command used.

---

## Coding Standards

- All functions and classes must have docstrings including arguments, return data,
  and exception raising sections.
- If an AI model is involved, `forward()` methods must document input/output tensor shapes in the docstring.
- Use type hints throughout (`torch.Tensor`, `np.ndarray`, etc.).
- Follow PEP 8. Line length limit: 100 characters.
- Do not commit large binary files (model weights, raw videos) to the repository.
- Use a fixed seed for reproducibility; document the seed used.

---

## Development conventions
- Keep code modular: separate components into respective subdirectories.
- Prefer explicit typing and clear function docstrings.
- Avoid hard-coded paths; use configuration or CLI arguments.
- Avoid exact paths; use relative paths wherever necessary for a program to locate modules or other files.
- Keep codes reproducible by setting random seeds where relevant.

---

## Quality expectations
- Add or update tests before or alongside new logic.
- Keep changes focused and minimal.
- Update README setup steps when workflow changes.
- Keep agent docs synchronized with code-level changes throughout the task.

---

## Evaluation Requirements

- For tyre digital twin estimates, report validation error against datasets that
  provide ground truth, such as core temperature or pressure MAE when available.
- For laps-to-cliff or remaining-life models, report lap-error metrics and empirical
  interval coverage when using conformal bounds.
- For event detectors, report precision/recall against rule labels or reviewed labels,
  and clearly separate possible anomalies from confirmed events.
- For end-to-end demos, verify that pit-wall/crew alerts include latency, reason text,
  and the tyre or condition that triggered the call.

---

## Constraints

- Always anonymize sensitive data. Do not commit sensitive or identifiable data.
- Redact identifiers in examples, logs, docs, plots, and error reports.
- Ask before running commands that could transmit data outside the local workspace..
- Reproducibility: every experiment result must be re-runnable from its config file alone.
