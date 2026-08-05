# Contributing to Guard Local Detector

Thank you for considering contributing to Guard! We welcome contributions from everyone, whether it’s fixing bugs, 
adding new features, optimizing detection, or improving documentation.

The following is a set of guidelines for contributing to this repository.

## The Contributor License Agreement (CLA)

Before we can merge your first Pull Request, you will need to sign our Contributor License Agreement.

Don't worry, the process is fully automated! When you open your first Pull Request, our CLA bot will automatically 
comment on it with instructions. You will simply need to reply to that comment to sign the agreement. You only have to 
do this once.

Note that this repository is licensed under the **AGPL-3.0**, unlike the Apache-2.0 Guard client that consumes it. Your 
contributions are published under those terms.

## Getting Started

Before you start writing code, make sure your development environment is set up properly.

0. Fork the repository to your own GitHub account.

1. Install prerequisites: [uv](https://docs.astral.sh/uv/) and Python 3.9 or newer. You do not need to install Python 
yourself — uv will fetch a suitable interpreter.

2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/guard-local-python.git`

3. Sync the environment: `uv sync`

That last command creates a `.venv`, reads `uv.lock`, and installs the runtime and development dependencies exactly as 
they were locked. Nothing else is needed — the detection model ships inside the repository, so there is no download 
step and no network access at any point.

## Branching Strategy

To keep the repository organized, please use descriptive branch names based on the type of work you are doing:

* **Features:** `feature/animated-gif-sampling` or `feat/coreml-execution-provider`
* **Bug Fixes:** `fix/video-keyframe-seek`
* **Documentation:** `docs/update-readme`

Always branch off of the `main` branch, and make sure your fork is up to date before starting new work.

## Development, Linting & Testing

Please ensure your changes pass all code quality checks and tests before opening a Pull Request.

### Code Quality & Types

Run the formatter, linter, and type-checker to ensure your code complies with our standards:

```bash
# format the code
uv run ruff format

# check for lint issues
uv run ruff check

# automatically fix lint issues where possible
uv run ruff check --fix

# check types (strict mode)
uv run mypy
```

### Running Tests

We use pytest. Please add or update tests whenever you introduce new features or fix bugs.

```bash
# run the whole suite
uv run pytest

# run the conformance suite the Guard client depends on
uv run pytest tests/test_contract.py

# run one test with output
uv run pytest tests/test_engine.py::TestAnalyze -v
```

The suite needs no fixture files: images and video clips are generated in `tests/conftest.py` at run time. If you add a 
new media format, add its builder there rather than committing a binary.

### Verifying Against the Guard Client

Changes to the public surface — the constructor, `analyze`, the returned keys, or the labels — should be checked against 
the real client and not only against our own tests:

```bash
cd ../guard-python
uv sync --extra local
uv run pytest tests/test_local.py
```

`guard-python` resolves this package from the sibling checkout, so no publish step is involved.

## What to Watch Out For

This package is small, but a few things in it are load-bearing in ways that are not obvious from the code.

**`tests/test_contract.py` is not ours to edit.** It is owned by `guard-python` and pasted here verbatim, so a diff 
between the two copies always means the contract changed. It is excluded from `ruff format` for the same reason. If it 
fails, fix the engine.

**The labels in `src/guard_local/models.py` are a public API.** The client derives each result's task id from 
`uuid5(namespace, label)`, so renaming a label silently changes the id that callers key on. They also mirror the tasks 
the cloud API seeds, which is what lets someone test locally and then route to the cloud unchanged.

**The preprocessing must match the browser extension.** Both run the same ONNX file, and the scores only agree because 
the transform agrees down to the rounding. Before touching `src/guard_local/media_utils.py`, read 
`src/lib/localModel/runner.ts` in `guard-browser-extension`. If a change is intentional, re-record the expected values 
in `test_golden_scores_for_a_deterministic_image` and say in the PR why the scores moved.

**Scores must be floats between `0.0` and `1.0`.** The client rescales by value rather than by a declared scale, so an 
integer `1` would be read as full confidence rather than as near-zero.

**Never add `guard-client` as a dependency.** The dependency runs the other way, as an optional extra; adding it here 
would make it circular and defeat the point of keeping the AGPL engine opt-in.

**Everything raised must subclass `GuardLocalError`.** The client calls `analyze` without a `try`/`except`, so anything 
else reaches the caller as a raw Pillow, PyAV, or ONNX Runtime error.

## Submitting a Pull Request

When you are ready to submit your code, open a Pull Request (PR) against the main branch of the original repository.

Please include the following in your PR description:

* **The Problem:** What issue does this PR solve? (Link to an existing Issue if applicable).
* **The Solution:** A brief explanation of how you solved it.
* **Testing:** How did you test your changes? Mention which tests were added, and whether you ran the Guard client 
against your branch.
* **Score Impact:** If your change can move detection scores — anything in the decoding or preprocessing path — say by 
how much, and on what input.

Once submitted, a maintainer will review your code. We may request some changes before merging, but we will always be 
respectful and constructive!

## Reporting Bugs & Requesting Features

If you aren't writing code but found a bug or have a feature idea, please open an Issue!

* Provide as much detail as possible, including your Python version, operating system, CPU architecture, and the 
versions of `guard-local-detector` and `guard-client` you are running.
* For a decoding or inference failure, include the full traceback and the media type you passed in. A file that 
reproduces it helps enormously — please only attach media you are comfortable sharing publicly.
* If media was scored incorrectly, include the scores you got and the scores you expected. Note that this engine is a 
lightweight baseline: for a second opinion, run the same file through the cloud API, whose larger models are the 
reference.
