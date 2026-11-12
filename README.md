# F1-MARL

**Multi-Agent Reinforcement Learning for Competitive Autonomous Formula-Style Racing**

F1-MARL is a research-oriented software project exploring how multiple autonomous racing agents learn to compete through reinforcement learning and self-play.

> **Project status:** Phase 0 — repository foundation. The racing simulator and learning algorithms are not implemented yet.

## Research direction

The initial research direction is to investigate whether population-based self-play can produce robust competitive behaviours—such as overtaking, defending, and collision avoidance—in a multi-agent racing environment.

The project will begin with a transparent, testable 2D vehicle model and progressively add complexity. It is not intended to be an FIA-accurate Formula 1 simulator.

## Planned capabilities

- Deterministic 2D track and vehicle simulation
- Multi-agent environment with simultaneous actions
- Vehicle dynamics, track limits, and collision handling
- PPO baseline followed by MAPPO-style centralized training / decentralized execution
- Population-based self-play and opponent evaluation
- Reproducible experiments and performance metrics
- Race visualization and experiment dashboards

## Repository layout

```text
.
├── .github/                 # CI and contribution templates
├── configs/                 # Version-controlled experiment configuration
├── docs/                    # Architecture and research notes
├── src/f1_marl/             # Installable Python package
├── tests/                   # Automated tests
├── scripts/                 # Developer and experiment entry points
├── experiments/             # Experiment notes and metadata
└── pyproject.toml            # Package and tool configuration
```

## Requirements

- Windows 11, macOS, or Linux
- Python 3.11
- Git

## Setup (Windows PowerShell)

```powershell
# Create and activate an isolated environment
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# Upgrade packaging tools
python -m pip install --upgrade pip

# Install the package and developer tools
python -m pip install -e ".[dev]"
```

If PowerShell blocks activation, run this for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## Verify the installation

```powershell
python -m pytest
ruff check .
ruff format --check .
```

## Configuration

Initial environment settings live in `configs/environment.yaml`. Training settings will be added when the first learning baseline is implemented. Configuration files are deliberately kept separate from source code so experiments can be reproduced and compared.

## Development workflow

1. Open or select a GitHub issue describing one focused change.
2. Create a branch from `main`.
3. Implement the change with tests and documentation.
4. Run the local quality checks.
5. Push the branch and open a pull request.
6. Merge only after review and passing CI.

Branch naming examples:

- `feature/track-representation`
- `feature/vehicle-dynamics`
- `fix/collision-boundary`
- `docs/research-methodology`

Commit message convention:

```text
type(scope): concise imperative description
```

Examples: `feat(simulation): add vehicle state model`, `test(environment): cover reset behaviour`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Reproducibility

Experiments should record their configuration, random seed, software version, and evaluation metrics. Large datasets, model checkpoints, and generated outputs should not be committed unless there is a clear, documented reason.

## Research integrity

Results must distinguish measured outcomes from hypotheses. Baselines, ablations, evaluation conditions, and known limitations should be documented. Do not describe the environment as real-world F1-accurate.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
