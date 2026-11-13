# F1-MARL Environment Architecture

## 1. Purpose

This document defines the architecture and interface contract for integrating the F1-MARL racing simulator with multi-agent reinforcement learning algorithms.

The environment will initially use the PettingZoo Parallel API, with Gymnasium spaces defining agent observations and actions.

The initial objective is to support simultaneous competitive racing between multiple autonomous vehicles.

The simulator is a simplified Formula-style racing environment. It is not intended to reproduce the complete physical behaviour of an FIA Formula One car.

## 2. Technology Stack

- Python 3.11
- NumPy
- PyTorch
- Gymnasium
- PettingZoo
- Pytest
- Ruff

The initial environment implementation will use PettingZoo's `ParallelEnv` interface.

## 3. Architecture Overview

The environment acts as the interface between reinforcement learning agents and the underlying racing simulator.

The main components are:

1. Multi-agent environment: manages agent lifecycle and API interaction.
2. Observation system: constructs individual observations for each active vehicle.
3. Action system: validates and translates agent actions into vehicle controls.
4. Reward system: calculates individual competitive rewards.
5. Simulation manager: advances all vehicles synchronously.
6. Episode manager: handles race completion, termination, truncation and reset.

### Data Flow

1. Each active agent receives an observation.
2. Each agent independently selects an action.
3. The environment collects actions from all active agents.
4. Actions are converted into vehicle controls.
5. The simulation manager advances all vehicles by one timestep.
6. Track interactions and simulation events are evaluated.
7. Individual rewards and episode statuses are calculated.
8. Updated observations and information are returned.

All active agents act simultaneously.

### Environment API Contract

The racing environment will implement the PettingZoo Parallel API, allowing all active agents to submit actions simultaneously.

The public API will follow these signatures:

- `reset(seed=None, options=None) -> observations, infos`
- `step(actions) -> observations, rewards, terminations, truncations, infos`

All returned values will be dictionaries keyed by agent ID.

The environment will expose:

- `possible_agents`: all agents supported by the environment.
- `agents`: agents currently participating in the episode.
- `observation_spaces`: observation space for each possible agent.
- `action_spaces`: action space for each possible agent.

The implementation must pass PettingZoo's `parallel_api_test`.

Agent lifecycle management must follow the Parallel API termination and truncation conventions. Finished agents must not receive further control actions, and their final transition information must be preserved.

### Global State Interface

The environment should reserve an optional global state interface through a `state()` method and corresponding `state_space`.

The global state will contain information about all vehicles and relevant simulation conditions.

Individual agents will continue receiving decentralized local observations.

The global state is intended for centralized critic training in future MAPPO implementations, while actors will use only their respective local observations.

This interface is reserved for future CTDE functionality and is not required for the initial PPO baseline.

## 4. Agent Model

Each vehicle is represented by an individual agent.

Agent identifiers will be stable strings, for example:

- `car_1`
- `car_2`
- `car_3`
- `car_4`

The environment will maintain:

- `possible_agents`: all agents configured for the episode.
- `agents`: agents currently active in the episode.

The number of agents will be configurable through the race scenario.

Each agent controls only its corresponding vehicle.

## 5. Observation Space

### 5.1 Design Requirements

Observations must:

- Have a fixed shape for every agent.
- Use normalized numerical values where appropriate.
- Use relative coordinates for opponent information.
- Avoid exposing unrestricted simulator state.
- Support different numbers of racing vehicles.
- Be reproducible for a given simulator state.

The initial observation will be a flat `float32` vector.

### 5.2 Observation Components

| Component | Features |
|---|---|
| Vehicle state | Speed, lateral offset, heading error, previous steering action, previous longitudinal action |
| Racing progress | Lap progress, normalized distance to finish |
| Track preview | Curvature at five lookahead points |
| Opponents | Relative longitudinal position, relative lateral position, relative speed, relative heading, presence mask |
| Vehicle status | On-track indicator, collision indicator |

### 5.3 Track Preview

The initial lookahead distances are:

- 10 metres
- 20 metres
- 35 metres
- 55 metres
- 80 metres

Track curvature will be calculated from the track geometry at each lookahead point.

### 5.4 Track Curvature Estimation

The initial track representation uses piecewise-linear segments. Since curvature is not directly defined at segment vertices and tangent direction may change abruptly, curvature observations must use a discrete estimator.

The estimator will calculate heading change over arc length:

```text
curvature = change_in_heading / change_in_arc_length
```

The calculation must include smoothing to reduce discontinuities caused by track vertices.

For closed tracks, heading differences must handle angular wraparound and curvature estimation must account for the connection between the final and initial track segments.

The estimator must be tested on straight sections, curved sections, and closed-track boundaries.

### 5.5 Opponent Perception

The initial opponent perception configuration is:

- Perception radius: 100 metres.
- Maximum visible opponents: 3.
- Selection: three nearest opponents within the perception radius.
- Ordering: nearest to farthest.
- Coordinates: relative to the observing vehicle.
- Missing opponents: zero-padded.

Opponent detection uses a Euclidean perception radius of 100 metres from the controlled vehicle.

For detected opponents, relative features are represented in the controlled vehicle's Frenet coordinate frame.

The longitudinal separation calculation must account for wraparound when the track is closed.

A maximum of three opponents will be included in the observation. Opponents will be sorted by Euclidean distance, nearest first. Equal-distance cases will use agent ID as a deterministic tie-breaker.

If fewer than three opponents are detected, the remaining slots will be zero-padded. The presence mask will indicate whether each slot contains a valid opponent.

The Euclidean distance is used for perception filtering, while relative Frenet coordinates are used for the observation features.

The perception radius and maximum number of opponents must be configurable.

The observation shape must remain constant regardless of the number of vehicles in the race.

### 5.6 Normalization

Continuous observation features will be normalized using documented, configurable, feature-specific physical bounds or scaling functions.

Normalization will use documented feature-specific bounds.

Features with known bounded ranges will use linear scaling to `[-1, 1]`, where appropriate. Angular features will use a consistent wrapped representation. Binary indicators will remain in `[0, 1]`.

Values outside configured physical bounds will be clipped after scaling.

The normalization configuration must be shared between training, evaluation and inference.

Missing opponent slots will use zero values with a zero presence mask.

### 5.7 Observation Vector Layout

Each agent receives a fixed-size observation vector containing 29 `float32` values.

The observation indices are defined as follows:

| Index | Feature | Description |
|---|---|---|
| 0 | Speed | Normalized vehicle speed |
| 1 | Lateral offset | Normalized lateral distance from track centre |
| 2 | Heading error | Vehicle heading relative to track direction |
| 3 | Previous steering | Previous normalized steering action |
| 4 | Previous longitudinal action | Previous normalized longitudinal action |
| 5 | Lap progress | Normalized progress within the current lap |
| 6 | Distance to finish | Normalized remaining race distance |
| 7–11 | Track curvature | Curvature at five preview distances |
| 12–16 | Opponent 1 | Five opponent features |
| 17–21 | Opponent 2 | Five opponent features |
| 22–26 | Opponent 3 | Five opponent features |
| 27 | On-track indicator | Binary indicator |
| 28 | Collision indicator | Binary indicator |

The five features for each opponent are ordered as follows:

1. Relative longitudinal position.
2. Relative lateral position.
3. Relative speed.
4. Relative heading difference.
5. Presence mask.

The resulting observation dimension is 29.

All observations will use `float32` values. Continuous features must be normalized using documented physical bounds or scaling functions. Binary indicators remain 0 or 1.

Lap progress represents progress within the current lap, while distance to finish represents the normalized remaining distance to complete the race. These features must not be treated as interchangeable.

## 6. Action Space

### 6.1 Design

Each agent will use a continuous two-dimensional action space.

The action vector is:

`[steering, longitudinal_control]`

Both values are bounded between -1.0 and +1.0.

The Gymnasium action space is:

`Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)`

### 6.2 Action Interpretation

| Action | Interpretation |
|---|---|
| Steering = -1.0 | Maximum left steering |
| Steering = 0.0 | Straight steering |
| Steering = +1.0 | Maximum right steering |
| Longitudinal = -1.0 | Maximum braking |
| Longitudinal = 0.0 | No throttle or braking |
| Longitudinal = +1.0 | Maximum throttle |

The longitudinal action will be converted into mutually exclusive throttle and brake commands.

For longitudinal control `u`:

- If `u > 0`: `throttle = u`, `brake = 0`.
- If `u < 0`: `throttle = 0`, `brake = abs(u)`.
- If `u = 0`: `throttle = 0`, `brake = 0`.

Existing vehicle parameter limits will remain authoritative for maximum steering angle, acceleration and braking.

### 6.3 Action Validation

Actions must contain exactly two finite values:

- Steering
- Longitudinal control

Non-finite values, including NaN and infinity, must be rejected with a clear error.

Finite values outside the supported `[-1, 1]` range will be clipped to the valid range before being converted into vehicle controls.

The longitudinal action will be converted into mutually exclusive throttle and braking inputs.

## 7. Reward Function

### 7.1 Design Objectives

The reward function should encourage competitive and safe racing behaviour.

The initial reward will contain:

- Forward progress.
- Competitive speed.
- Successful overtaking.
- Collision penalties.
- Track-limit penalties.
- Excessive or erratic control penalties.

### 7.2 Composite Reward

The reward for agent `i` is composed of the following components:

```text
R_i = w_p R_progress
    + w_s R_speed
    + w_o R_overtake
    - w_c P_collision
    - w_t P_track
    - w_a P_control
```

Where:

- `R_progress`: progress made during the current simulation step.
- `R_speed`: reward for maintaining useful forward speed.
- `R_overtake`: reward for a newly completed overtake event.
- `P_collision`: penalty associated with collision events.
- `P_track`: penalty for track-limit violations.
- `P_control`: penalty for excessive or oscillatory control inputs.

Progress reward must be based on the change in progress between consecutive steps rather than cumulative progress.

Speed reward must not encourage stationary behaviour or high-speed wrong-way driving.

An overtake reward must only be issued once for each qualifying overtake event, rather than repeatedly rewarding the same relative position.

Reward weights and per-step scaling will be configurable and tuned through controlled experiments.

### 7.3 Reward Principles

- Reward actual progress along the track.
- Avoid rewarding speed without considering progress.
- Reward genuine overtaking events rather than repeated position fluctuations.
- Penalize collisions.
- Penalize track-limit violations.
- Penalize excessive control oscillation.
- Avoid incentives that encourage deliberate collisions or blocking.

Reward components should be independently measurable and testable.

## 8. Collision Handling

A collision will initially produce a reward penalty.

A collision will not automatically eliminate a vehicle.

The simulation will continue unless another termination condition is reached.

Vehicle elimination may be enabled through scenario configuration in future experiments.

The initial implementation will use the existing simulation collision events. Collision response and impact physics are separate future simulation improvements.

## 9. Episode Lifecycle

### 9.1 Reset

Reset must:

1. Reset the simulation clock.
2. Restore or regenerate the starting grid.
3. Reset all vehicle states.
4. Clear previous simulation events.
5. Reset lap and progress tracking.
6. Reset agent-specific reward and event history.
7. Return initial observations and information.

The reset interface must support seeded reproducibility.

### 9.2 Termination

An individual agent may terminate when:

- It completes the configured race distance.
- It is eliminated under an enabled elimination rule.
- Its vehicle becomes unrecoverable under a defined rule.

An individual agent terminating does not automatically terminate the other agents.

The entire episode terminates when the race-level completion condition is satisfied, including when all participating agents have terminated.

### 9.3 Truncation

An episode may be truncated when:

- The maximum number of simulation steps is reached.
- The maximum simulation time is exceeded.
- An external experiment limit stops the episode.

Termination and truncation must be represented separately in accordance with the PettingZoo Parallel API.

### 9.4 Individual Agent Completion

An agent that completes the race will be marked as finished and will no longer receive control actions.

Its final reward and termination information must be preserved.

A finished vehicle must not interfere with the remaining active vehicles. The implementation may achieve this by safely parking the vehicle or excluding it from active collision participation.

The race may continue for the remaining agents. The episode ends when all participating agents have finished or terminated, or when the episode-level truncation condition is reached.

When an agent finishes, its final termination flag will be set to `True`. The environment must preserve the final transition before removing the agent from the active agent list according to the PettingZoo Parallel API.

The episode-level completion condition is reached when no active agents remain, unless the episode has already been truncated.

Agent removal and final transition handling must remain compatible with the PettingZoo Parallel API.

### 9.5 Reproducibility

The environment must support seeded resets.

All environment-level randomness, including starting-grid generation and randomized scenario configuration, must use the environment's seeded random-number generator.

Identical seeds and identical configurations must produce reproducible initial conditions.

## 10. Environment Configuration

The following values should be configurable:

- Track layout.
- Number of vehicles.
- Vehicle parameters.
- Starting-grid configuration.
- Simulation timestep.
- Race distance or lap count.
- Maximum episode steps.
- Maximum episode duration.
- Opponent perception radius.
- Maximum visible opponents.
- Observation normalization bounds.
- Reward weights.
- Vehicle elimination rules.

Configuration should integrate with the existing project configuration system.

## 11. Simulation Integration

The environment will use the existing simulation infrastructure:

- `Track`
- `SimulationManager`
- `RaceScenario`
- `VehicleState`
- `VehicleParameters`
- `ControlInput`
- Track interaction and collision event systems

The environment must not duplicate vehicle physics or collision detection.

The simulation manager remains responsible for advancing vehicle states.

The environment is responsible for translating agent actions, constructing observations, calculating rewards and managing episode lifecycle.

## 12. Testing Requirements

The implementation must include tests for:

- Environment initialization.
- Agent registration and lifecycle.
- Observation shape and data type.
- Observation bounds and normalization.
- Opponent selection and perception radius.
- Missing opponent padding and presence masks.
- Action validation and control conversion.
- Simultaneous action execution.
- Reward component calculations.
- Collision penalties.
- Termination and truncation.
- Seeded reset reproducibility.
- Different numbers of agents.
- PettingZoo Parallel API compatibility.

The environment implementation must pass PettingZoo's `parallel_api_test`.

## 13. Future Extensions

The architecture should allow future integration of:

- PPO baseline.
- Multi-Agent PPO (MAPPO).
- Centralized training and decentralized execution.
- Self-play training.
- Population-based training.
- Curriculum learning.
- More detailed vehicle dynamics.
- Tire and energy management.
- Advanced opponent perception.

These extensions are not part of the initial environment implementation.

## 14. Acceptance Criteria

Phase 2.1 is complete when:

- [ ] Environment architecture is documented and reviewed.
- [ ] Agent lifecycle is specified.
- [ ] Observation contract is specified, including the initial 29-dimensional layout.
- [ ] Observation feature ordering and normalization approach are documented.
- [ ] Action contract is specified.
- [ ] Reward design requirements are documented.
- [ ] Collision handling is specified.
- [ ] Termination and truncation behaviour is specified.
- [ ] Reset and reproducibility requirements are documented.
- [ ] Simulation integration boundaries are defined.
- [ ] Optional global state interface is reserved for future MAPPO support.
- [ ] Architecture is reviewed and approved before implementation.

Implementation-level checks, including PettingZoo API tests, observation tests, action validation tests and reward tests, will be verified in the subsequent environment implementation issue.
