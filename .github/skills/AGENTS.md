# AGENTS.md - FormulaTech Hacks SIDEWALL Repository

This file orients AI coding agents to this repository so they can work efficiently
without unnecessary exploration.

---

## Agent Instruction Hierarchy

This repository may be used by different coding agents and assistants. Before acting,
each agent should read the instruction files native to its own runtime or directory
layout, then use this file as the shared project-level guide.

Examples of agent-specific places to check when present:

- `.agents/` for local Codex-style skills or repository guidance.
- `.github/instructions/` and `.github/skills/` for GitHub Copilot-style instructions
  and agent roles.
- Tool-specific files such as `CLAUDE.md`, `GEMINI.md`, `.cursorrules`,
  `.windsurfrules`, or other assistant-specific docs if they are added later.

If these sources conflict, follow the user's current request first, then the most
specific local agent instruction, then this shared repository guide, then the README.
Do not assume instructions for one agent runtime automatically apply to another unless
they describe repository facts, safety limits, data contracts, or development workflow.

---

## What This Repository Does

Builds SIDEWALL, an AI tyre-safety digital twin and pit-wall monitor for FormulaTech
Hacks. The project focuses on motorsport safety diagnosis: estimating tyre state,
detecting possible anomalies, forecasting tyre risk, and explaining pit-wall calls
before degradation becomes dangerous.

The core product should distinguish normal tyre degradation from unusual behaviour
such as overheating, pressure anomalies, wheelspin, flat spots, fatigue, cold tyres,
and possible lock-ups. Acceleration anomalies may be useful context, but tyre safety
is the richer diagnosis problem and should remain the main development focus unless
the user asks otherwise.

---

## Development Workflow

Use test-driven development for meaningful functional changes:

1. Write or update a focused failing test for the intended behaviour.
2. Implement the smallest useful change.
3. Run the focused test, then expand to related tests when the change crosses modules.
4. Refactor only with tests passing.
5. Update relevant docs, schemas, examples, and `TEST_COVERAGE.md` files before the
   change is considered complete.

Documentation should evolve incrementally with the code. When data contracts, alert
wording, model assumptions, commands, or demo flows change, update the relevant
README, agent skill, schema/config example, or coverage note in the same pass.

---

## Data Sources and Limits

- **FastF1 / OpenF1**: Public race telemetry, lap/stint metadata, weather, and race
  context. Standard FastF1 telemetry includes speed, throttle, brake, RPM, gear,
  position, and session/lap metadata, but not per-wheel rotation speed, actual fuel
  mass, tyre pressure, or tyre temperature.
- **Race-control and status data**: Useful for failure labels such as tyre, puncture,
  wheel, or wheel nut events.
- **Sim racing or academic telemetry**: Prefer these for per-wheel slip, tyre
  temperature, tyre pressure, wear, and detector ground truth when available.
- **Optional local demo sources**: Phone controller/sensor streams, visual cues, or
  hardware-free simulation routes must feed the same documented source schema.

Always document whether a signal is measured, simulated, estimated by the digital
twin, inferred from public telemetry, or unavailable.

---

## Evidence and Safety Language

- Public telemetry alone can flag a **possible lock-up** or **braking anomaly**; it
  cannot directly confirm individual wheel lock-up without wheel-speed data or another
  confirming source.
- Visual model outputs, smoke cues, and image/video interpretations are candidate
  evidence for human review, not confirmed tyre-failure labels.
- Tyre pressure, tyre temperature, fuel load, and wear estimates must be labelled
  `estimated` unless the data source directly provides them.
- Pit-wall alerts must include the reason and evidence used, not only a severity.
- Avoid overstating safety claims. This is a decision-support prototype unless the
  user explicitly asks for a different positioning.

---

## Suggested Repository Areas

| What you need | Likely location |
|---|---|
| Data ingestion and source adapters | `sidewall/data/` or `backend/` adapters |
| Digital twin / virtual TPMS | `sidewall/twin/` or model services |
| Tyre risk and event detectors | `sidewall/models/` |
| Pit-call logic and explanations | `sidewall/engine/` |
| Replay, sim, and phone inputs | `sidewall/sources/` |
| Backend API / WebSockets | `sidewall/server/` or `backend/` |
| Pit-wall, crew, driver, atlas UI | `sidewall/web/` or `frontend/` |
| Tests | `tests/` near the code under test |

Prefer the structure already present in the repository over creating these folders
from scratch. If a new area is introduced, document its purpose and verification
commands.

---

## Quality Expectations

- Keep changes focused and incremental. Feel free to update agent documentations and README files.
- Use type hints and docstrings for public functions/classes.
- Avoid hard-coded paths; use configuration, CLI arguments, or documented defaults.
- Keep reproducibility explicit: seeds, configs, commands, and fixture assumptions.
- Use synthetic/anonymized fixtures for tests.
- Add regression tests for bug fixes.
- For alerting and diagnosis logic, test both the decision and the explanation.
- Update test coverage notes after coverage runs, including the command used and the
  observed percentage.

---

## Ask First

- Adding a new external dependency.
- Downloading large datasets or running commands that may transmit data outside the
  workspace.
- Changing public data schemas, alert severity conventions, or source adapter output
  contracts.
- Repositioning the project away from tyre safety diagnosis.

---

## Never Do

- Present inferred tyre state as measured ground truth.
- Confirm a lock-up from public FastF1 telemetry alone.
- Commit credentials, raw large datasets, binary model artifacts, or generated caches.
- Hide uncertainty in demo or dashboard wording.
- Leave docs stale after changing behaviour.
