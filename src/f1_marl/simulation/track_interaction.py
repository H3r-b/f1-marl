"""Vehicle interaction with track geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

from f1_marl.simulation.track import Track
from f1_marl.simulation.vehicle import VehicleState


@dataclass(frozen=True, slots=True)
class TrackInteraction:
    """Vehicle position and orientation relative to a track."""

    s: float
    d: float
    heading_error_rad: float
    is_on_track: bool
    progress_fraction: float
    segment_index: int


def calculate_track_interaction(
    vehicle: VehicleState,
    track: Track,
) -> TrackInteraction:
    """Calculate a vehicle's position and orientation relative to a track.

    Positive lateral offset means the vehicle is to the left of the
    track direction. Heading error is normalized to [-pi, pi).

    Args:
        vehicle: Current vehicle state in Cartesian coordinates.
        track: Track geometry used for projection.

    Returns:
        Track-relative position, heading error, boundary status,
        and normalized longitudinal progress.
    """
    frenet = track.project(vehicle.x, vehicle.y)
    track_sample = track.sample(frenet.s)

    track_heading = math.atan2(
        track_sample.tangent_y,
        track_sample.tangent_x,
    )

    heading_difference = vehicle.heading_rad - track_heading

    heading_error = (heading_difference + math.pi) % (2.0 * math.pi) - math.pi

    is_on_track = abs(frenet.d) <= track.half_width_m

    progress_fraction = frenet.s / track.length_m

    return TrackInteraction(
        s=frenet.s,
        d=frenet.d,
        heading_error_rad=heading_error,
        is_on_track=is_on_track,
        progress_fraction=progress_fraction,
        segment_index=frenet.segment_index,
    )
