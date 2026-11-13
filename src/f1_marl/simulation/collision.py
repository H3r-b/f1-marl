"""Vehicle collision geometry and simulation event detection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from f1_marl.simulation.track import Track
from f1_marl.simulation.track_interaction import calculate_track_interaction
from f1_marl.simulation.vehicle import VehicleState


@dataclass(frozen=True, slots=True)
class VehicleFootprint:
    """Rectangular vehicle dimensions, measured in metres."""

    length_m: float = 5.0
    width_m: float = 2.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.length_m) or self.length_m <= 0:
            raise ValueError("Vehicle length must be finite and positive.")

        if not math.isfinite(self.width_m) or self.width_m <= 0:
            raise ValueError("Vehicle width must be finite and positive.")


@dataclass(frozen=True, slots=True)
class VehicleCollisionEvent:
    """A collision detected between two vehicles."""

    vehicle_id_a: str
    vehicle_id_b: str
    simulation_time_s: float


@dataclass(frozen=True, slots=True)
class TrackBoundaryEvent:
    """A vehicle whose centre has crossed the track boundary."""

    vehicle_id: str
    simulation_time_s: float
    lateral_offset_m: float
    track_half_width_m: float


SimulationEvent = VehicleCollisionEvent | TrackBoundaryEvent


def oriented_box_corners(
    state: VehicleState,
    footprint: VehicleFootprint,
) -> tuple[tuple[float, float], ...]:
    """Calculate the four world-coordinate corners of a vehicle."""
    forward_x = math.cos(state.heading_rad)
    forward_y = math.sin(state.heading_rad)

    left_x = -forward_y
    left_y = forward_x

    half_length = footprint.length_m / 2
    half_width = footprint.width_m / 2

    front_x = state.x + forward_x * half_length
    front_y = state.y + forward_y * half_length

    rear_x = state.x - forward_x * half_length
    rear_y = state.y - forward_y * half_length

    return (
        (
            front_x + left_x * half_width,
            front_y + left_y * half_width,
        ),
        (
            front_x - left_x * half_width,
            front_y - left_y * half_width,
        ),
        (
            rear_x - left_x * half_width,
            rear_y - left_y * half_width,
        ),
        (
            rear_x + left_x * half_width,
            rear_y + left_y * half_width,
        ),
    )


def _projection_interval(
    corners: tuple[tuple[float, float], ...],
    axis: tuple[float, float],
) -> tuple[float, float]:
    """Project rectangle corners onto an axis."""
    projections = [x * axis[0] + y * axis[1] for x, y in corners]

    return min(projections), max(projections)


def boxes_intersect(
    state_a: VehicleState,
    state_b: VehicleState,
    footprint_a: VehicleFootprint | None = None,
    footprint_b: VehicleFootprint | None = None,
    tolerance_m: float = 1e-9,
) -> bool:
    """Check whether two oriented rectangular vehicle footprints intersect.

    Uses the Separating Axis Theorem (SAT). Contact at an edge or corner
    is treated as an intersection.
    """
    if not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("Tolerance must be finite and non-negative.")

    footprint_a = footprint_a or VehicleFootprint()
    footprint_b = footprint_b or VehicleFootprint()

    corners_a = oriented_box_corners(state_a, footprint_a)
    corners_b = oriented_box_corners(state_b, footprint_b)

    axes = (
        (
            math.cos(state_a.heading_rad),
            math.sin(state_a.heading_rad),
        ),
        (
            -math.sin(state_a.heading_rad),
            math.cos(state_a.heading_rad),
        ),
        (
            math.cos(state_b.heading_rad),
            math.sin(state_b.heading_rad),
        ),
        (
            -math.sin(state_b.heading_rad),
            math.cos(state_b.heading_rad),
        ),
    )

    for axis in axes:
        min_a, max_a = _projection_interval(corners_a, axis)
        min_b, max_b = _projection_interval(corners_b, axis)

        if max_a < min_b - tolerance_m:
            return False

        if max_b < min_a - tolerance_m:
            return False

    return True


def detect_vehicle_collisions(
    states: Mapping[str, VehicleState],
    footprint: VehicleFootprint | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return all intersecting vehicle pairs in deterministic order."""
    footprint = footprint or VehicleFootprint()
    vehicle_ids = sorted(states)
    collisions: list[tuple[str, str]] = []

    for index, vehicle_id_a in enumerate(vehicle_ids):
        for vehicle_id_b in vehicle_ids[index + 1 :]:
            if boxes_intersect(
                states[vehicle_id_a],
                states[vehicle_id_b],
                footprint,
                footprint,
            ):
                collisions.append((vehicle_id_a, vehicle_id_b))

    return tuple(collisions)


def detect_simulation_events(
    states: Mapping[str, VehicleState],
    track: Track,
    footprint: VehicleFootprint,
    simulation_time_s: float,
) -> tuple[SimulationEvent, ...]:
    """Detect vehicle collisions and track-boundary violations."""
    if not math.isfinite(simulation_time_s) or simulation_time_s < 0:
        raise ValueError("Simulation time must be finite and non-negative.")

    events: list[SimulationEvent] = []

    for vehicle_id_a, vehicle_id_b in detect_vehicle_collisions(
        states,
        footprint,
    ):
        events.append(
            VehicleCollisionEvent(
                vehicle_id_a=vehicle_id_a,
                vehicle_id_b=vehicle_id_b,
                simulation_time_s=simulation_time_s,
            )
        )

    for vehicle_id in sorted(states):
        interaction = calculate_track_interaction(
            states[vehicle_id],
            track,
        )

        if not interaction.is_on_track:
            events.append(
                TrackBoundaryEvent(
                    vehicle_id=vehicle_id,
                    simulation_time_s=simulation_time_s,
                    lateral_offset_m=interaction.d,
                    track_half_width_m=track.half_width_m,
                )
            )

    return tuple(events)
