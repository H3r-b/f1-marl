# Architecture

## Design principles

1. **Separation of concerns:** simulation mechanics, environment API, policies, training, and visualization remain independent modules.
2. **Determinism:** a fixed seed should reproduce environment initialization and stochastic choices where practical.
3. **Testability:** physics and rules should be testable without rendering or training a neural network.
4. **Explicit interfaces:** observations, actions, rewards, and termination conditions must be documented.
5. **Progressive fidelity:** begin with a simple 2D model; add complexity only when validated by tests and experiments.

## Planned boundaries

- `simulation`: track geometry, vehicle state, dynamics, tyres, and collision geometry.
- `environment`: multi-agent reset/step API, observation construction, action validation, rewards, and episode lifecycle.
- `agents`: policy interfaces and baseline agents.
- `algorithms`: PPO/MAPPO implementations and self-play components.
- `training`: orchestration, checkpointing, evaluation, and metrics.
- `visualization`: optional rendering and dashboards.

## Initial execution model

The intended multi-agent environment uses simultaneous actions: every active agent supplies an action for a simulation tick, then the world advances once and returns the next observations, rewards, terminations, truncations, and infos.

## Current status

Phase 0 only establishes packaging, configuration conventions, tests, and CI. No vehicle or racing physics are claimed to exist yet.
