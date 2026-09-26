---
name: formulatech-coding-general
description: Always use this as a general coding guide
---

# General Coding Instructions

Use test-driven development for meaningful logic changes. Add or update the focused test first, implement the smallest useful change, run the focused test, then broaden verification as needed. Refer to testing guides [here](../formulatech-testing/SKILL.md).

Keep documentation current while developing, not only at the end. When behaviour, data contracts, model assumptions, commands, or user-facing alert wording changes, update the relevant README, agent skill, schema/config example, or `TEST_COVERAGE.md` in the same development pass.

For FormulaTech tyre-safety work, be precise about evidence. Label tyre pressure, temperature, fuel load, lock-up, and wheelspin values as measured, simulated, estimated, possible, candidate, or unavailable according to the data source. Public FastF1 telemetry alone can flag braking anomalies or possible lock-ups, but it cannot directly confirm individual wheel lock-up, tyre pressure, or tyre temperature.

Modularize common helper functions in a `utils.py` and common constants at a `constants.py` at a suitable location. 
