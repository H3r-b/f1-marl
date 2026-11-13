"""Predefined racing track layouts for F1-MARL."""

from __future__ import annotations

import math

import numpy as np

from f1_marl.simulation.track import Track


def _validate_layout_parameters(
    semi_major_axis_m: float,
    semi_minor_axis_m: float,
    track_half_width_m: float,
    num_points: int,
) -> None:
    """Validate parameters shared by predefined track layouts."""
    dimensions = (
        semi_major_axis_m,
        semi_minor_axis_m,
        track_half_width_m,
    )

    if not all(math.isfinite(value) and value > 0 for value in dimensions):
        raise ValueError("Track dimensions must be finite and positive.")

    if isinstance(num_points, bool) or not isinstance(num_points, int):
        raise ValueError("Number of points must be an integer.")

    if num_points < 8:
        raise ValueError("A racing track requires at least 8 points.")


def create_oval_track(
    *,
    semi_major_axis_m: float = 120.0,
    semi_minor_axis_m: float = 60.0,
    track_half_width_m: float = 8.0,
    num_points: int = 160,
) -> Track:
    """Create a closed elliptical racing circuit.

    Args:
        semi_major_axis_m: Horizontal semi-axis of the circuit.
        semi_minor_axis_m: Vertical semi-axis of the circuit.
        track_half_width_m: Half-width of the drivable track.
        num_points: Number of centreline samples.

    Returns:
        A closed Track instance.
    """
    _validate_layout_parameters(
        semi_major_axis_m,
        semi_minor_axis_m,
        track_half_width_m,
        num_points,
    )

    angles = np.linspace(
        0.0,
        2.0 * math.pi,
        num_points,
        endpoint=False,
    )

    centerline = np.column_stack(
        (
            semi_major_axis_m * np.cos(angles),
            semi_minor_axis_m * np.sin(angles),
        )
    )

    return Track(
        centerline,
        closed=True,
        half_width_m=track_half_width_m,
    )


def create_technical_track(
    *,
    semi_major_axis_m: float = 140.0,
    semi_minor_axis_m: float = 75.0,
    track_half_width_m: float = 8.0,
    num_points: int = 240,
) -> Track:
    """Create a closed technical circuit with varying curvature.

    The radial modulation produces a smooth, asymmetric layout
    without introducing self-intersections.

    Args:
        semi_major_axis_m: Horizontal base semi-axis.
        semi_minor_axis_m: Vertical base semi-axis.
        track_half_width_m: Half-width of the drivable track.
        num_points: Number of centreline samples.

    Returns:
        A closed Track instance.
    """
    _validate_layout_parameters(
        semi_major_axis_m,
        semi_minor_axis_m,
        track_half_width_m,
        num_points,
    )

    angles = np.linspace(
        0.0,
        2.0 * math.pi,
        num_points,
        endpoint=False,
    )

    radial_scale = 1.0 + 0.12 * np.cos(3.0 * angles) + 0.06 * np.sin(2.0 * angles)

    centerline = np.column_stack(
        (
            semi_major_axis_m * radial_scale * np.cos(angles),
            semi_minor_axis_m * radial_scale * np.sin(angles),
        )
    )

    return Track(
        centerline,
        closed=True,
        half_width_m=track_half_width_m,
    )
