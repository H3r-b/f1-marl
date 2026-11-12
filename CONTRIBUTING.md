# Contributing

Thank you for contributing to F1-MARL. The project prioritizes correctness, reproducibility, readable code, and small reviewable changes.

## Before starting

- Check existing issues and pull requests.
- For substantial work, open an issue first to agree on scope.
- Keep pull requests focused on one logical change.

## Local setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Branching

Create branches from an up-to-date `main`:

```powershell
git switch main
git pull --ff-only
git switch -c feature/short-description
```

Use `feature/`, `fix/`, `docs/`, `test/`, or `refactor/` prefixes.

## Quality gates

Before pushing:

```powershell
python -m pytest
ruff check .
ruff format --check .
```

Add or update tests for behaviour changes. Do not weaken or delete tests simply to make CI pass.

## Commits

Use Conventional Commit-style messages:

```text
feat(simulation): add vehicle state model
fix(physics): clamp invalid steering input
test(environment): cover reset behaviour
docs(architecture): explain simulation boundaries
```

## Pull requests

- Explain the problem and the solution.
- Link the relevant issue.
- List important design decisions.
- Include test commands and their results.
- Include screenshots or short recordings for visual changes.
- Call out limitations and follow-up work.

Do not merge failing CI checks. Request review before merging when collaborating. For a solo project, perform a deliberate self-review of the diff before merging.

## Research reproducibility

Record random seeds, configuration files, environment versions, and evaluation conditions. Avoid committing private data, credentials, large generated outputs, or model checkpoints.
