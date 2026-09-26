---
name: evgs-repo-guidance
description: Use for general work in the FormulaTech SIDEWALL tyre-safety repository, including setup, repository conventions, data-source limits, code quality, testing, and cross-cutting project guidance.
---

# FormulaTech Repository Guidance

Before making general repository changes, read and follow:

- `../../../.github/skills/AGENTS.md`
- `../../../.github/instructions/copilot-instructions.md`

Use these files as the source of truth for the project context, FormulaTech tyre-safety
scope, coding standards, test-driven workflow, documentation expectations, and agent
boundaries.

Persistent memory for Codex in this repository: tyre and race telemetry may include
large downloaded datasets, cached API data, logs, generated model artifacts, and
competition/pitch materials. Inspect schemas, aggregate metadata, and code paths before
raw contents; avoid printing raw downloaded records unnecessarily; redact credentials
or identifiers; and ask before any action that could transmit data outside the local
workspace.

Work test-first for meaningful logic changes and update docs incrementally during the
same development pass. For tyre diagnosis, always distinguish measured, simulated,
estimated, possible, candidate, and unavailable signals.

When other FormulaTech skills also apply, combine this guidance with the more specific
skill for the area you are changing.

Document any significant changes in a suitable `README.md` file, for reproducibility.
