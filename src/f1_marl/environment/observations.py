"""Structured observations for multi-agent racing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from f1_marl.simulation.collision import VehicleCollisionEvent
from f1_marl.simulation.manager import SimulationManager
from f1_marl.simulation.track_interaction import calculate_track_interaction

OBSERVATION_DIM = 29
LOOKAHEAD_DISTANCES_M = (10.0, 20.0, 35.0, 55.0, 80.0)
MAX_OPPONENTS = 3


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    """Physical scales and perception settings for observations."""

    speed_scale_mps: float = 100.0
    perception_radius_m: float = 100.0
    curvature_window_m: float = 5.0
    curvature_scale_per_m: float = 0.1

    def __post_init__(self) -> None:
        values = (
            self.speed_scale_mps,
            self.perception_radius_m,
            self.curvature_window_m,
            self.curvature_scale_per_m,
        )

        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("All observation configuration values must be finite and positive.")


class ObservationBuilder:
    """Build fixed-size, normalized observations for racing agents."""

    def __init__(self, config: ObservationConfig | None = None) -> None:
        self.config = config or ObservationConfig()

    def build(
        self,
        agent_id: str,
        simulation: SimulationManager,
        previous_actions: dict[str, tuple[float, float]] | None = None,
    ) -> np.ndarray:
        """Build the 29-dimensional observation for one vehicle."""
        if agent_id not in simulation.vehicle_ids:
            raise KeyError(f"Unknown vehicle: {agent_id}")

        previous_actions = previous_actions or {}

        states = simulation.states
        track = simulation.track

        ego_state = states[agent_id]
        ego_interaction = calculate_track_interaction(ego_state, track)

        observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)

        # ---------------------------------------------------------
        # 0-6: Ego vehicle state
        # ---------------------------------------------------------

        half_width = track.half_width_m

        observation[0] = np.clip(
            ego_state.speed_mps / self.config.speed_scale_mps,
            0.0,
            1.0,
        )

        observation[1] = np.clip(
            ego_interaction.d / half_width,
            -1.0,
            1.0,
        )

        observation[2] = np.clip(
            ego_interaction.heading_error_rad / math.pi,
            -1.0,
            1.0,
        )

        previous_steering, previous_longitudinal = previous_actions.get(agent_id, (0.0, 0.0))

        observation[3] = np.clip(previous_steering, -1.0, 1.0)
        observation[4] = np.clip(previous_longitudinal, -1.0, 1.0)

        progress = float(np.clip(ego_interaction.progress_fraction, 0.0, 1.0))

        observation[5] = progress

        # Provisional one-lap race assumption.
        observation[6] = 1.0 - progress

        # ---------------------------------------------------------
        # 7-11: Track curvature ahead
        # ---------------------------------------------------------

        for index, lookahead in enumerate(LOOKAHEAD_DISTANCES_M):
            curvature = self._estimate_curvature(
                track,
                ego_interaction.s + lookahead,
            )

            observation[7 + index] = np.clip(
                curvature / self.config.curvature_scale_per_m,
                -1.0,
                1.0,
            )

        # ---------------------------------------------------------
        # 12-26: Three nearest opponents
        # ---------------------------------------------------------

        opponents = []

        for opponent_id, opponent_state in states.items():
            if opponent_id == agent_id:
                continue

            distance = math.hypot(
                opponent_state.x - ego_state.x,
                opponent_state.y - ego_state.y,
            )

            if distance > self.config.perception_radius_m:
                continue

            opponent_interaction = calculate_track_interaction(
                opponent_state,
                track,
            )

            if track.closed:
                delta_s = (
                    opponent_interaction.s - ego_interaction.s + track.length_m / 2.0
                ) % track.length_m - track.length_m / 2.0
            else:
                delta_s = opponent_interaction.s - ego_interaction.s

            delta_d = opponent_interaction.d - ego_interaction.d

            opponents.append(
                (
                    distance,
                    opponent_id,
                    opponent_state,
                    delta_s,
                    delta_d,
                )
            )

        # Stable ordering: distance first, vehicle ID second.
        opponents.sort(key=lambda item: (item[0], item[1]))

        for slot, opponent in enumerate(opponents[:MAX_OPPONENTS]):
            _, _, opponent_state, delta_s, delta_d = opponent

            offset = 12 + slot * 5

            observation[offset] = np.clip(
                delta_s / self.config.perception_radius_m,
                -1.0,
                1.0,
            )

            observation[offset + 1] = np.clip(
                delta_d / (2.0 * half_width),
                -1.0,
                1.0,
            )

            observation[offset + 2] = np.clip(
                (opponent_state.speed_mps - ego_state.speed_mps) / self.config.speed_scale_mps,
                -1.0,
                1.0,
            )

            observation[offset + 3] = np.clip(
                _wrap_angle(opponent_state.heading_rad - ego_state.heading_rad) / math.pi,
                -1.0,
                1.0,
            )

            observation[offset + 4] = 1.0

        # ---------------------------------------------------------
        # 27-28: Track and collision indicators
        # ---------------------------------------------------------

        observation[27] = float(ego_interaction.is_on_track)

        observation[28] = float(
            any(
                isinstance(event, VehicleCollisionEvent)
                and agent_id in (event.vehicle_id_a, event.vehicle_id_b)
                for event in simulation.events
            )
        )

        return observation

    def _estimate_curvature(self, track, s: float) -> float:
        """Estimate signed track curvature using tangent direction changes."""
        half_window = self.config.curvature_window_m / 2.0

        if track.closed:
            start_s = s - half_window
            end_s = s + half_window
        else:
            start_s = max(0.0, s - half_window)
            end_s = min(track.length_m, s + half_window)

        if end_s <= start_s:
            return 0.0

        start_sample = track.sample(start_s)
        end_sample = track.sample(end_s)

        start_heading = math.atan2(
            start_sample.tangent_y,
            start_sample.tangent_x,
        )

        end_heading = math.atan2(
            end_sample.tangent_y,
            end_sample.tangent_x,
        )

        heading_change = _wrap_angle(end_heading - start_heading)

        return heading_change / (end_s - start_s)
