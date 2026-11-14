"""PettingZoo Parallel environment for multi-agent Formula-style racing."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from f1_marl.simulation.scenarios import RaceScenario
from f1_marl.simulation.vehicle import ControlInput


class ParallelRacingEnv(ParallelEnv):
    """Simultaneous multi-agent racing environment.

    This initial implementation establishes the PettingZoo API and connects
    agent actions to the existing simulation manager.

    Observation construction and reward calculation are placeholders.
    """

    metadata = {"name": "f1_marl_parallel_v0"}

    OBSERVATION_DIM = 29
    ACTION_DIM = 2

    def __init__(
        self,
        scenario: RaceScenario,
        *,
        dt: float = 0.05,
        max_episode_steps: int = 10_000,
    ) -> None:
        if not isinstance(scenario, RaceScenario):
            raise TypeError("scenario must be a RaceScenario.")

        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Timestep must be finite and positive.")

        if (
            isinstance(max_episode_steps, bool)
            or not isinstance(max_episode_steps, int)
            or max_episode_steps <= 0
        ):
            raise ValueError("max_episode_steps must be a positive integer.")

        self.scenario = scenario
        self.dt = float(dt)
        self.max_episode_steps = max_episode_steps

        self.possible_agents = list(scenario.vehicle_ids)
        self.agents: list[str] = []

        self.observation_spaces = {
            agent: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.OBSERVATION_DIM,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.ACTION_DIM,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self.np_random, self.np_random_seed = seeding.np_random(None)

        self._simulation = None
        self._step_count = 0

    def observation_space(self, agent: str) -> spaces.Box:
        """Return the observation space for an agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Box:
        """Return the action space for an agent."""
        return self.action_spaces[agent]

    def _placeholder_observation(self) -> np.ndarray:
        """Return a temporary zero observation.

        The actual 29-feature observation builder will be implemented
        in a subsequent milestone.
        """
        return np.zeros(self.OBSERVATION_DIM, dtype=np.float32)

    def _convert_action(self, action: np.ndarray) -> ControlInput:
        """Validate and convert a normalized action into vehicle controls."""
        try:
            values = np.asarray(action, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Action must contain two numeric values.") from exc

        if values.shape != (self.ACTION_DIM,):
            raise ValueError("Action must have shape (2,).")

        if not np.isfinite(values).all():
            raise ValueError("Action values must be finite.")

        steering, longitudinal = np.clip(values, -1.0, 1.0)

        parameters = self._simulation.parameters

        steering_angle = float(steering * parameters.max_steering_angle_rad)

        throttle = float(max(longitudinal, 0.0))
        brake = float(max(-longitudinal, 0.0))

        return ControlInput(
            steering_angle_rad=steering_angle,
            throttle=throttle,
            brake=brake,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset the racing episode and return initial observations."""
        self.np_random, self.np_random_seed = seeding.np_random(seed)

        self._simulation = self.scenario.create_simulation()
        self._step_count = 0
        self.agents = self.possible_agents.copy()

        observations = {agent: self._placeholder_observation() for agent in self.agents}

        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(
        self,
        actions: Mapping[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Advance all active vehicles by one synchronized timestep."""
        if self._simulation is None:
            raise RuntimeError("Call reset() before step().")

        if not self.agents:
            raise RuntimeError("The episode has ended. Call reset().")

        if not isinstance(actions, Mapping):
            raise TypeError("Actions must be a mapping of agent IDs to actions.")

        expected_agents = set(self.agents)
        supplied_agents = set(actions)

        if supplied_agents != expected_agents:
            missing = expected_agents - supplied_agents
            unknown = supplied_agents - expected_agents

            raise ValueError(
                f"Action IDs must match active agents. "
                f"Missing: {sorted(missing)}; "
                f"unknown: {sorted(unknown)}."
            )

        controls = {agent: self._convert_action(actions[agent]) for agent in self.agents}

        active_agents = self.agents.copy()

        self._simulation.step(controls, self.dt)
        self._step_count += 1

        episode_truncated = self._step_count >= self.max_episode_steps

        observations = {agent: self._placeholder_observation() for agent in active_agents}

        # Reward calculation is intentionally deferred.
        rewards = {agent: 0.0 for agent in active_agents}

        # Individual race termination is not implemented yet.
        terminations = {agent: False for agent in active_agents}

        truncations = {agent: episode_truncated for agent in active_agents}

        infos = {
            agent: {
                "simulation_time_s": self._simulation.simulation_time_s,
                "step_count": self._step_count,
            }
            for agent in active_agents
        }

        if episode_truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos
