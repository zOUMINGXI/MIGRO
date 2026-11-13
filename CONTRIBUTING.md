# Contributing to MIGRO

Thank you for helping improve MIGRO. Contributions to documentation, memory interfaces, integrations and evaluation tooling are welcome.

## Start a discussion

Use the issue templates to report a bug or propose an improvement. For substantial interface or experiment changes, describe the intended behavior in an issue first so the design and scope can be discussed.

For runtime questions, start with the [setup guide](docs/runtime.md) and include the setup step that failed. For evaluation questions, identify the benchmark, host, memory construction path, read budget and scoring definition.

## Prepare a change

1. Fork the repository and create a branch for one focused change.
2. Install dependencies for that component using the [runtime guide](docs/runtime.md).
3. Keep code, documentation and command examples consistent.
4. Run relevant existing checks. The component suite is invoked with:

~~~bash
python -m unittest discover -s tests -v
~~~

Some checks require optional packages or model assets. Include the commands you ran and any skips in the pull request. For documentation changes, check relative links, command names and rendered formatting.

## Open a pull request

Explain the problem and resulting behavior, link the relevant issue, and describe how the change was checked. Keep unrelated formatting and refactoring in separate changes.

When changing evaluation behavior, specify the metric, cohort, memory operation and budget affected. New reported results should identify the evaluated state and settings so comparisons can be interpreted correctly.

## Licensing and attribution

MIGRO-owned code is published under Apache-2.0. Preserve upstream licenses and attribution when incorporating third-party material. See [third-party notices](THIRD_PARTY_NOTICES.md).
