# ---
name: formulatech-testing
description: Use for managing code quality via tests.
---

# FormulaTech Testing
For each function, class or module, which involves functional logic, we should have tests to ensure the code quality. This skill is designed to help us manage our tests effectively.

## Test-Driven Workflow
- For non-trivial functional changes, write or update the test first so it captures the intended behaviour before implementation.
- Work in small red-green-refactor cycles: failing focused test, minimal implementation, passing focused test, then cleanup.
- Every bug fix should include a regression test that would have caught the bug.
- Use synthetic, anonymized, or minimal fixture data for telemetry and tyre states. Do not require raw downloaded race data for unit tests.
- When a test covers an inferred safety event, assert the confidence level or wording too: public telemetry should produce "possible lock-up", "braking anomaly", or "candidate" language unless direct wheel-speed or visual confirmation is part of the fixture.

## Test Types
1. **Unit Tests**: Test individual functions or classes in isolation.
2. **Integration Tests**: Test how different modules or components work together.
3. **End-to-End Tests**: Test the entire application flow from start to finish.

## Tools
- **pytest**: A popular testing framework for Python.
- **mock**: A library for mocking/stubbing objects in tests.

## Best Practices
- Write tests for both expected and edge cases.
- Use descriptive names for test functions to indicate what they are testing.
- Keep tests independent of each other to avoid cascading failures.
- Use fixtures to set up any necessary test data or state.
- Regularly run tests to catch issues early in the development process.
- For alerting, pit-call, and diagnosis logic, test both the decision and the explanation shown to a user.
- In each subdirectory, maintain a `TEST_COVERAGE.md` file to track test coverage % (most importantly with the number in percent), and optionally, to identify areas for improvement. Update this file every time after running new tests.
  - Note: It's the best if you run the test suite (or just bounded by a certain subdirectory) with a coverage tool (e.g., `pytest-cov`) to get accurate coverage metrics from the terminal, and thus, copy the relevant coverage information into the `TEST_COVERAGE.md` file. This way, we can ensure that our coverage tracking is based on actual test runs rather than estimates. 
- Aim for high test coverage, but prioritize meaningful tests over achieving 100% coverage.
