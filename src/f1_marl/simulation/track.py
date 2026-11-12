"""Track geometry and Cartesian-to-Frenet coordinate transformations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TrackSample:
    """Geometric properties of a point sampled along the track."""

    s: float
    x: float
    y: float
    tangent_x: float
    tangent_y: float
    normal_x: float
    normal_y: float


@dataclass(frozen=True, slots=True)
class FrenetCoordinate:
    """Position projected onto the track centerline."""

    s: float
    d: float
    segment_index: int


class Track:
    """Piecewise-linear track centerline with Frenet projection.

    Closed tracks implicitly connect the final point to the first point.
    Positive lateral displacement is to the left of the direction of travel.
    """

    def __init__(
        self,
        centerline: Sequence[Sequence[float]] | FloatArray,
        *,
        closed: bool = False,
        half_width_m: float = 5.0,
    ) -> None:
        points = np.asarray(centerline, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Centerline must have shape (N, 2).")

        if len(points) < 2:
            raise ValueError("A track requires at least two centerline points.")

        if not np.isfinite(points).all():
            raise ValueError("Centerline coordinates must be finite.")

        if not np.isfinite(half_width_m) or half_width_m <= 0:
            raise ValueError("Track half-width must be finite and positive.")

        # Avoid a redundant final point on closed tracks.
        if closed and np.linalg.norm(points[-1] - points[0]) <= 1e-9:
            points = points[:-1]

        if len(points) < 2:
            raise ValueError("A track requires at least two distinct points.")

        self._closed = closed
        self._half_width_m = float(half_width_m)
        self._points = points.copy()

        if closed:
            next_points = np.roll(self._points, -1, axis=0)
        else:
            next_points = self._points[1:]

        start_points = self._points if closed else self._points[:-1]

        vectors = next_points - start_points
        lengths = np.linalg.norm(vectors, axis=1)

        if np.any(lengths <= 1e-9):
            raise ValueError("Track contains a zero-length segment.")

        self._segment_lengths = lengths
        self._segment_vectors = vectors
        self._tangents = vectors / lengths[:, None]

        self._cumulative_lengths = np.concatenate(([0.0], np.cumsum(lengths)))
        self._length_m = float(np.sum(lengths))

    @property
    def length_m(self) -> float:
        """Total centerline length in metres."""
        return self._length_m

    @property
    def half_width_m(self) -> float:
        """Track half-width in metres."""
        return self._half_width_m

    @property
    def closed(self) -> bool:
        """Whether the track wraps around its endpoint."""
        return self._closed

    @property
    def centerline(self) -> FloatArray:
        """Return a copy of the centerline coordinates."""
        return self._points.copy()

    def _normalize_s(self, s: float) -> float:
        if not np.isfinite(s):
            raise ValueError("Track position s must be finite.")

        if self._closed:
            return float(s % self._length_m)

        return float(np.clip(s, 0.0, self._length_m))

    def sample(self, s: float) -> TrackSample:
        """Interpolate position, tangent, and left normal at distance s."""
        s_normalized = self._normalize_s(s)

        segment_index = int(
            np.searchsorted(
                self._cumulative_lengths,
                s_normalized,
                side="right",
            )
            - 1
        )

        segment_index = min(
            max(segment_index, 0),
            len(self._segment_lengths) - 1,
        )

        segment_start_s = self._cumulative_lengths[segment_index]
        segment_length = self._segment_lengths[segment_index]

        t = (s_normalized - segment_start_s) / segment_length

        start = self._points[segment_index]
        position = start + t * self._segment_vectors[segment_index]

        tangent = self._tangents[segment_index]
        normal = np.array([-tangent[1], tangent[0]])

        return TrackSample(
            s=s_normalized,
            x=float(position[0]),
            y=float(position[1]),
            tangent_x=float(tangent[0]),
            tangent_y=float(tangent[1]),
            normal_x=float(normal[0]),
            normal_y=float(normal[1]),
        )

    def project(self, x: float, y: float) -> FrenetCoordinate:
        """Project a Cartesian point onto the nearest track segment.

        Returns:
            FrenetCoordinate containing longitudinal progress, signed
            lateral displacement, and nearest segment index.
        """
        if not np.isfinite([x, y]).all():
            raise ValueError("Cartesian coordinates must be finite.")

        point = np.array([x, y], dtype=np.float64)

        starts = self._points if self._closed else self._points[:-1]
        offsets = point - starts

        segment_lengths_squared = self._segment_lengths**2

        t = (
            np.sum(
                offsets * self._segment_vectors,
                axis=1,
            )
            / segment_lengths_squared
        )

        t = np.clip(t, 0.0, 1.0)

        projections = starts + t[:, None] * self._segment_vectors
        distances_squared = np.sum((point - projections) ** 2, axis=1)

        segment_index = int(np.argmin(distances_squared))

        closest_point = projections[segment_index]
        tangent = self._tangents[segment_index]
        displacement = point - closest_point

        # 2D cross product: positive means left of the track direction.
        lateral_offset = tangent[0] * displacement[1] - tangent[1] * displacement[0]

        s = (
            self._cumulative_lengths[segment_index]
            + t[segment_index] * self._segment_lengths[segment_index]
        )

        if self._closed:
            s %= self._length_m

        return FrenetCoordinate(
            s=float(s),
            d=float(lateral_offset),
            segment_index=segment_index,
        )
