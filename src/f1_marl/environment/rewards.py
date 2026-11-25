"""Configurable reward components for multi-agent racing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from f1_marl.simulation.collision import (
    TrackBoundaryEvent,
    VehicleCollisionEvent,
)
from f1_marl.simulation.manager import SimulationManager


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Configuration for individual racing reward components."""

    overtake_detection_distance_m: float = 20.0
    overtake_confirmation_distance_m: float = 2.0
    progress_weight: float = 1.0
    speed_weight: float = 0.2
    overtaking_weight: float = 2.0
    collision_penalty: float = 5.0
    track_limit_penalty: float = 1.0
    oscillation_penalty: float = 0.1

    target_speed_mps: float = 50.0
    oscillation_threshold: float = 0.1

    def __post_init__(self) -> None:
        """Reject invalid reward weights and parameters."""
        non_negative_fields = (
            "progress_weight",
            "speed_weight",
            "overtaking_weight",
            "collision_penalty",
            "track_limit_penalty",
            "oscillation_penalty",
        )

        for field_name in non_negative_fields:
            value = getattr(self, field_name)

            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative.")

        if not math.isfinite(self.target_speed_mps) or self.target_speed_mps <= 0:
            raise ValueError("target_speed_mps must be finite and positive.")

        if not math.isfinite(self.oscillation_threshold) or self.oscillation_threshold < 0:
            raise ValueError("oscillation_threshold must be finite and non-negative.")

        if (
            not math.isfinite(self.overtake_detection_distance_m)
            or self.overtake_detection_distance_m <= 0
        ):
            raise ValueError("overtake_detection_distance_m must be finite and positive.")

        if (
            not math.isfinite(self.overtake_confirmation_distance_m)
            or self.overtake_confirmation_distance_m <= 0
            or self.overtake_confirmation_distance_m >= self.overtake_detection_distance_m
        ):
            raise ValueError(
                "overtake_confirmation_distance_m must be finite, "
                "positive, and smaller than overtake_detection_distance_m."
            )


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Individual reward components for a single racing agent."""

    progress: float = 0.0
    speed: float = 0.0
    overtaking: float = 0.0
    collision: float = 0.0
    track_limits: float = 0.0
    oscillation: float = 0.0

    @property
    def total(self) -> float:
        """Return the sum of all reward components."""
        return (
            self.progress
            + self.speed
            + self.overtaking
            + self.collision
            + self.track_limits
            + self.oscillation
        )


class RewardCalculator:
    """Stateful reward calculator for multi-agent racing."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self.reset()

    def reset(self, simulation: SimulationManager | None = None) -> None:
        """Reset reward history, optionally initializing from a simulation."""
        self._previous_positions: dict[str, float] = {}
        self._previous_gaps: dict[tuple[str, str], float] = {}
        self._confirmed_order: dict[tuple[str, str], int] = {}
        self._previous_actions: dict[str, np.ndarray] = {}
        self._previous_action_deltas: dict[str, np.ndarray] = {}
        self._active_collision_pairs: set[tuple[str, str]] = set()

        if simulation is None:
            return

        positions = {
            vehicle_id: simulation.get_track_interaction(vehicle_id).s
            for vehicle_id in simulation.vehicle_ids
        }

        self._previous_positions = positions.copy()
        self._previous_actions = {
            vehicle_id: np.zeros(2, dtype=np.float64) for vehicle_id in simulation.vehicle_ids
        }

        self._previous_gaps = {
            (first, second): (
                self._signed_gap(
                    positions[first],
                    positions[second],
                    simulation.track.length_m,
                )
                if simulation.track.closed
                else positions[first] - positions[second]
            )
            for index, first in enumerate(simulation.vehicle_ids)
            for second in simulation.vehicle_ids[index + 1 :]
        }

        confirmation_distance = self.config.overtake_confirmation_distance_m

        self._confirmed_order = {
            pair: (
                1 if gap >= confirmation_distance else -1 if gap <= -confirmation_distance else 0
            )
            for pair, gap in self._previous_gaps.items()
        }

    def remove_agents(self, agent_ids: tuple[str, ...]) -> None:
        """Remove finished agents while preserving active reward history."""
        removed_agents = set(agent_ids)

        unknown_agents = removed_agents - self._previous_positions.keys()

        if unknown_agents:
            raise ValueError(f"Unknown reward agents: {sorted(unknown_agents)}")

        for agent in removed_agents:
            self._previous_positions.pop(agent, None)
            self._previous_actions.pop(agent, None)
            self._previous_action_deltas.pop(agent, None)

        self._previous_gaps = {
            pair: gap
            for pair, gap in self._previous_gaps.items()
            if removed_agents.isdisjoint(pair)
        }

        self._confirmed_order = {
            pair: order
            for pair, order in self._confirmed_order.items()
            if removed_agents.isdisjoint(pair)
        }

        self._active_collision_pairs = {
            pair for pair in self._active_collision_pairs if removed_agents.isdisjoint(pair)
        }

    @staticmethod
    def _signed_gap(
        first_s: float,
        second_s: float,
        track_length: float,
    ) -> float:
        """Return the shortest signed circular distance between vehicles."""
        return float((first_s - second_s + track_length / 2) % track_length - track_length / 2)

    def calculate(
        self,
        simulation: SimulationManager,
        actions: Mapping[str, np.ndarray],
    ) -> dict[str, RewardBreakdown]:
        """Calculate weighted reward components for every vehicle."""
        vehicle_ids = simulation.vehicle_ids

        if set(actions) != set(vehicle_ids):
            raise ValueError("Actions must be provided for every vehicle.")

        if set(self._previous_positions) != set(vehicle_ids):
            raise RuntimeError("RewardCalculator must be reset with the current simulation.")

        normalized_actions: dict[str, np.ndarray] = {}

        for vehicle_id in vehicle_ids:
            action = np.asarray(actions[vehicle_id], dtype=np.float64)

            if action.shape != (2,) or not np.isfinite(action).all():
                raise ValueError(f"Action for '{vehicle_id}' must contain two finite values.")

            normalized_actions[vehicle_id] = np.clip(action, -1.0, 1.0)

        config = self.config

        components = {
            vehicle_id: {
                "progress": 0.0,
                "speed": 0.0,
                "overtaking": 0.0,
                "collision": 0.0,
                "track_limits": 0.0,
                "oscillation": 0.0,
            }
            for vehicle_id in vehicle_ids
        }

        current_positions = {
            vehicle_id: simulation.get_track_interaction(vehicle_id).s for vehicle_id in vehicle_ids
        }

        # Forward progress, including closed-track wraparound.
        progress_deltas = {}

        for vehicle_id in vehicle_ids:
            previous_s = self._previous_positions[vehicle_id]
            current_s = current_positions[vehicle_id]

            if simulation.track.closed:
                progress_delta = self._signed_gap(
                    current_s,
                    previous_s,
                    simulation.track.length_m,
                )
            else:
                progress_delta = current_s - previous_s

            progress_deltas[vehicle_id] = progress_delta

            components[vehicle_id]["progress"] = progress_delta * config.progress_weight

            self._previous_positions[vehicle_id] = current_s

        # Bounded reward for maintaining an appropriate speed.
        for vehicle_id in vehicle_ids:
            speed = simulation.get_state(vehicle_id).speed_mps

            speed_score = max(
                0.0,
                1.0 - abs(speed - config.target_speed_mps) / config.target_speed_mps,
            )

            components[vehicle_id]["speed"] = speed_score * config.speed_weight

        # Detect confirmed changes in relative racing order.
        current_gaps = {}

        confirmation_distance = config.overtake_confirmation_distance_m
        distance_limit = config.overtake_detection_distance_m

        for index, first in enumerate(vehicle_ids):
            for second in vehicle_ids[index + 1 :]:
                pair = (first, second)

                previous_gap = self._previous_gaps[pair]

                gap = previous_gap + progress_deltas[first] - progress_deltas[second]

                current_gaps[pair] = gap

                previous_order = self._confirmed_order[pair]
                current_order = previous_order

                if gap >= confirmation_distance:
                    current_order = 1
                elif gap <= -confirmation_distance:
                    current_order = -1

                within_detection_range = (
                    abs(previous_gap) <= distance_limit and abs(gap) <= distance_limit
                )

                if within_detection_range:
                    if previous_order == -1 and current_order == 1:
                        components[first]["overtaking"] += config.overtaking_weight
                    elif previous_order == 1 and current_order == -1:
                        components[second]["overtaking"] += config.overtaking_weight

                self._confirmed_order[pair] = current_order

        self._previous_gaps = current_gaps
        # Penalize newly occurring collisions.
        current_collision_pairs = set()

        for event in simulation.events:
            if isinstance(event, VehicleCollisionEvent):
                pair = tuple(sorted((event.vehicle_id_a, event.vehicle_id_b)))
                current_collision_pairs.add(pair)

        new_collisions = current_collision_pairs - self._active_collision_pairs

        for first, second in new_collisions:
            components[first]["collision"] -= config.collision_penalty
            components[second]["collision"] -= config.collision_penalty

        self._active_collision_pairs = current_collision_pairs

        # Penalize vehicles outside track boundaries.
        for event in simulation.events:
            if isinstance(event, TrackBoundaryEvent):
                components[event.vehicle_id]["track_limits"] -= config.track_limit_penalty

        # Penalize significant reversals in control direction.
        for vehicle_id, action in normalized_actions.items():
            previous_action = self._previous_actions[vehicle_id]
            current_delta = action - previous_action
            previous_delta = self._previous_action_deltas.get(vehicle_id)

            if previous_delta is not None:
                reversals = (
                    (previous_delta * current_delta < 0)
                    & (np.abs(previous_delta) >= config.oscillation_threshold)
                    & (np.abs(current_delta) >= config.oscillation_threshold)
                )

                components[vehicle_id]["oscillation"] = -config.oscillation_penalty * int(
                    np.count_nonzero(reversals)
                )

            self._previous_actions[vehicle_id] = action.copy()
            self._previous_action_deltas[vehicle_id] = current_delta.copy()

        rewards = {
            vehicle_id: RewardBreakdown(**values) for vehicle_id, values in components.items()
        }

        if not all(math.isfinite(reward.total) for reward in rewards.values()):
            raise FloatingPointError("Reward calculation produced a non-finite value.")

        return rewards
