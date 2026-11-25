"""PettingZoo Parallel environment for multi-agent Formula-style racing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from f1_marl.environment.observations import ObservationBuilder
from f1_marl.environment.race_lifecycle import RaceConfig, RaceLifecycleManager
from f1_marl.environment.rewards import RewardCalculator, RewardConfig
from f1_marl.simulation.scenarios import RaceScenario
from f1_marl.simulation.vehicle import ControlInput


class ParallelRacingEnv(ParallelEnv):
    """Simultaneous multi-agent racing environment.

    Connects agent actions to the simulation manager and provides
    structured observations and configurable racing rewards.

    Supports configurable multi-lap races and individual agent lifecycle
    management.
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
        reward_config: RewardConfig | None = None,
        race_config: RaceConfig | None = None,
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

        if reward_config is not None and not isinstance(reward_config, RewardConfig):
            raise TypeError("reward_config must be a RewardConfig.")
        if race_config is not None and not isinstance(race_config, RaceConfig):
            raise TypeError("race_config must be a RaceConfig.")

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
        self._observation_builder = ObservationBuilder()
        self._reward_calculator = RewardCalculator(reward_config)
        self._race_lifecycle = RaceLifecycleManager(race_config)
        self._previous_actions: dict[str, tuple[float, float]] = {}

    def observation_space(self, agent: str) -> spaces.Box:
        """Return the observation space for an agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Box:
        """Return the action space for an agent."""
        return self.action_spaces[agent]

    def _build_observation(
        self,
        agent: str,
        *,
        excluded_opponents: tuple[str, ...] = (),
    ) -> np.ndarray:
        """Build the current race-aware observation for an agent."""
        return self._observation_builder.build(
            agent,
            self._simulation,
            self._previous_actions,
            race_progress=self._race_lifecycle.progress[agent],
            race_distance_m=self._race_lifecycle.race_distance_m,
            excluded_opponents=excluded_opponents,
        )

    def _build_race_info(self, agent: str) -> dict:
        """Build race lifecycle information for an agent."""
        progress = self._race_lifecycle.progress[agent]

        return {
            "total_laps": self._race_lifecycle.config.total_laps,
            "race_distance_m": self._race_lifecycle.race_distance_m,
            "cumulative_progress_m": progress.cumulative_progress_m,
            "completed_laps": progress.completed_laps,
            "race_position": progress.race_position,
            "finished": progress.finished,
            "finish_position": progress.finish_position,
            "finish_time_s": progress.finish_time_s,
        }

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Validate an action and clip it to the normalized action bounds."""
        try:
            values = np.asarray(action, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Action must contain two numeric values.") from exc

        if values.shape != (self.ACTION_DIM,):
            raise ValueError("Action must have shape (2,).")

        if not np.isfinite(values).all():
            raise ValueError("Action values must be finite.")

        return np.clip(values, -1.0, 1.0)

    def _convert_action(self, action: np.ndarray) -> ControlInput:
        """Convert a normalized action into vehicle controls."""
        steering, longitudinal = self._normalize_action(action)

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

        self._race_lifecycle.reset(
            self._simulation,
            start_s_m=self.scenario.start_s_m,
        )

        self._reward_calculator.reset(self._simulation)
        self._step_count = 0
        self.agents = self.possible_agents.copy()

        # Previous actions start at neutral for every agent.
        self._previous_actions = {agent: (0.0, 0.0) for agent in self.agents}

        observations = {agent: self._build_observation(agent) for agent in self.agents}

        infos = {
            agent: {
                "simulation_time_s": self._simulation.simulation_time_s,
                "step_count": self._step_count,
                "race": self._build_race_info(agent),
            }
            for agent in self.agents
        }

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

        # Validate and clip each action once, then use the same values
        # for both vehicle controls and the previous-action observations.
        normalized_actions = {
            agent: self._normalize_action(actions[agent]) for agent in self.agents
        }

        controls = {agent: self._convert_action(normalized_actions[agent]) for agent in self.agents}

        active_agents = self.agents.copy()

        self._simulation.step(controls, self.dt)
        self._step_count += 1

        newly_finished = self._race_lifecycle.update(self._simulation)
        finished_agents = set(newly_finished)

        self._previous_actions = {
            agent: (
                float(normalized_actions[agent][0]),
                float(normalized_actions[agent][1]),
            )
            for agent in active_agents
        }

        episode_truncated = self._step_count >= self.max_episode_steps

        # Calculate rewards before removing any finished vehicles.
        reward_breakdowns = self._reward_calculator.calculate(
            self._simulation,
            normalized_actions,
        )

        rewards = {agent: reward_breakdowns[agent].total for agent in active_agents}

        # Preserve terminal observations and final-step collision events.
        observations = {
            agent: self._build_observation(
                agent,
                excluded_opponents=(() if agent in finished_agents else newly_finished),
            )
            for agent in active_agents
        }

        # Natural race completion takes precedence over truncation.
        terminations = {agent: agent in finished_agents for agent in active_agents}

        truncations = {
            agent: episode_truncated and not terminations[agent] for agent in active_agents
        }

        infos = {
            agent: {
                "simulation_time_s": self._simulation.simulation_time_s,
                "step_count": self._step_count,
                "reward_components": asdict(reward_breakdowns[agent]),
                "race": self._build_race_info(agent),
            }
            for agent in active_agents
        }

        # Remove finished vehicles only after capturing their transitions.
        if newly_finished:
            self._reward_calculator.remove_agents(newly_finished)

            for agent in newly_finished:
                self._simulation.remove_vehicle(agent)
                self._previous_actions.pop(agent, None)

        # Retain only agents that can participate in the next step.
        self.agents = [
            agent for agent in active_agents if not terminations[agent] and not truncations[agent]
        ]

        return observations, rewards, terminations, truncations, infos
