"""
VIBECODED!!!!

Jobard & Lefer (1997) evenly spaced streamline algorithm for 2D regular grids.

Reference:
    Jobard, B., & Lefer, W. (1997). Creating Evenly-Spaced Streamlines of
    Arbitrary Density. Visualization in Scientific Computing '97, 43-55.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Distance lookup grid
# ---------------------------------------------------------------------------


class _LookupGrid:
    """Spatial hash grid for fast nearest-streamline distance queries.

    Divides the domain into cells of size d_sep / sqrt(2) so that any point
    within distance d_sep of a query point is guaranteed to be in one of the
    3x3 neighbouring cells.
    """

    def __init__(self, shape: tuple[int, int], d_sep: float):
        self.cell_size = d_sep / np.sqrt(2)
        nx, ny = shape
        self.grid_nx = int(np.ceil(nx / self.cell_size)) + 1
        self.grid_ny = int(np.ceil(ny / self.cell_size)) + 1
        self.cells: dict[tuple[int, int], list[np.ndarray]] = {}

    def _cell(self, point: np.ndarray) -> tuple[int, int]:
        return (
            int(point[0] / self.cell_size),
            int(point[1] / self.cell_size),
        )

    def add_point(self, point: np.ndarray) -> None:
        key = self._cell(point)
        if key not in self.cells:
            self.cells[key] = []
        self.cells[key].append(point)

    def is_valid(self, point: np.ndarray, d_sep: float) -> bool:
        """Return True if point is at least d_sep away from all stored points."""
        cx, cy = self._cell(point)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (cx + dx, cy + dy)
                if key not in self.cells:
                    continue
                for p in self.cells[key]:
                    if np.hypot(point[0] - p[0], point[1] - p[1]) < d_sep:
                        return False
        return True

    def add_streamline(self, points: np.ndarray) -> None:
        for p in points:
            self.add_point(p)


# ---------------------------------------------------------------------------
# Vector field interpolator
# ---------------------------------------------------------------------------


def _make_interpolators(
    arr: np.ndarray,
) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    """Build (u, v) interpolators from a (nx, ny, 2+) array."""
    x = np.arange(arr.shape[0], dtype=float)
    y = np.arange(arr.shape[1], dtype=float)
    u = RegularGridInterpolator((x, y), arr[..., 0], bounds_error=False, fill_value=None)
    v = RegularGridInterpolator((x, y), arr[..., 1], bounds_error=False, fill_value=None)
    return u, v


# ---------------------------------------------------------------------------
# Streamline integration (RK4, arc-length parameterised)
# ---------------------------------------------------------------------------


def _integrate(
    seed: np.ndarray,
    u_interp: RegularGridInterpolator,
    v_interp: RegularGridInterpolator,
    shape: tuple[int, int],
    step: float,
    max_steps: int,
    d_sep: float,
    d_test: float,
    lookup: _LookupGrid,
    direction: int = 1,
) -> np.ndarray:
    """Integrate a single streamline from seed in the given direction (+1/-1).

    Integration stops when:
      - The point leaves the domain
      - The velocity is below a minimum threshold
      - The point comes within d_test of an existing streamline
    """
    nx, ny = shape
    points = [seed.copy()]
    pos = seed.copy()

    for _ in range(max_steps):
        # RK4 step (arc-length parameterised: velocity normalised to unit length)
        def velocity(p: np.ndarray) -> np.ndarray:
            u = float(u_interp([[p[0], p[1]]])[0])
            v = float(v_interp([[p[0], p[1]]])[0])
            mag = np.hypot(u, v)
            if mag < 1e-10:
                return np.zeros(2)
            return direction * np.array([u, v]) / mag

        k1 = velocity(pos)
        k2 = velocity(pos + 0.5 * step * k1)
        k3 = velocity(pos + 0.5 * step * k2)
        k4 = velocity(pos + step * k3)
        next_pos = pos + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # stop if out of bounds
        if not (0 <= next_pos[0] < nx and 0 <= next_pos[1] < ny):
            break

        # stop if too close to existing streamline
        if not lookup.is_valid(next_pos, d_test):
            break

        # stop if velocity is zero
        if np.all(velocity(next_pos) == 0):
            break

        pos = next_pos
        points.append(pos.copy())

    return np.array(points)


def _integrate_both_directions(
    seed: np.ndarray,
    u_interp: RegularGridInterpolator,
    v_interp: RegularGridInterpolator,
    shape: tuple[int, int],
    step: float,
    max_steps: int,
    d_sep: float,
    d_test: float,
    lookup: _LookupGrid,
) -> np.ndarray:
    """Integrate forward and backward and join into a single streamline."""
    fwd = _integrate(
        seed,
        u_interp,
        v_interp,
        shape,
        step,
        max_steps,
        d_sep,
        d_test,
        lookup,
        direction=1,
    )
    bwd = _integrate(
        seed,
        u_interp,
        v_interp,
        shape,
        step,
        max_steps,
        d_sep,
        d_test,
        lookup,
        direction=-1,
    )
    # reverse backward segment (excluding duplicate seed) and prepend
    if len(bwd) > 1:
        return np.concatenate([bwd[1:][::-1], fwd])
    return fwd


# ---------------------------------------------------------------------------
# Candidate seed generation
# ---------------------------------------------------------------------------


def _candidate_seeds(streamline: np.ndarray, d_sep: float) -> list[np.ndarray]:
    """Generate candidate seeds at d_sep distance on each side of the streamline."""
    candidates = []
    for i in range(len(streamline) - 1):
        # tangent direction
        tangent = streamline[i + 1] - streamline[i]
        norm = np.linalg.norm(tangent)
        if norm < 1e-10:
            continue
        tangent /= norm
        # normal (perpendicular)
        normal = np.array([-tangent[1], tangent[0]])
        midpoint = streamline[i]
        for side in (+1, -1):
            candidates.append(midpoint + side * d_sep * normal)
    return candidates


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


def evenly_spaced_streamlines(
    arr: np.ndarray,
    d_sep: float = 5.0,
    d_test: float | None = None,
    step: float = 1.0,
    max_steps: int = 1000,
    seed: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Generate evenly spaced streamlines using the Jobard & Lefer algorithm.

    Parameters
    ----------
    arr : np.ndarray
        Vector field array of shape (ny, nx, 2) or (ny, nx, 3). Only the
        first two components (in-plane) are used.
    d_sep : float
        Separation distance between streamlines in pixels. Controls density.
    d_test : float, optional
        Minimum distance from existing streamlines before integration stops.
        Defaults to d_sep / 2.
    step : float
        Integration step size in pixels.
    max_steps : int
        Maximum number of integration steps per streamline direction.
    seed : np.ndarray, optional
        Starting seed point (x, y). Defaults to the centre of the domain.

    Returns
    -------
    list of np.ndarray
        Each element is an (N, 2) array of (x, y) coordinates for one
        streamline.
    """
    if d_test is None:
        d_test = d_sep / 2.0

    arr = np.transpose(arr, (1, 0, 2))
    shape = arr.shape[:2]
    nx, ny = shape

    if seed is None:
        seed = np.array([nx / 2.0, ny / 2.0])

    u_interp, v_interp = _make_interpolators(arr)
    lookup = _LookupGrid(shape, d_sep)

    streamlines = []
    queue = deque()

    # --- first streamline from seed ---
    first = _integrate_both_directions(
        seed, u_interp, v_interp, shape, step, max_steps, d_sep, d_test, lookup
    )
    if len(first) > 1:
        lookup.add_streamline(first)
        streamlines.append(first)
        queue.append(first)

    # --- process queue ---
    while queue:
        current = queue.popleft()
        for candidate in _candidate_seeds(current, d_sep):
            # skip out-of-bounds candidates
            if not (0 <= candidate[0] < nx and 0 <= candidate[1] < ny):
                continue
            # skip candidates too close to existing streamlines
            if not lookup.is_valid(candidate, d_sep):
                continue
            # integrate new streamline
            new_line = _integrate_both_directions(
                candidate,
                u_interp,
                v_interp,
                shape,
                step,
                max_steps,
                d_sep,
                d_test,
                lookup,
            )
            if len(new_line) < 2:
                continue
            lookup.add_streamline(new_line)
            streamlines.append(new_line)
            queue.append(new_line)

    return streamlines


# ---------------------------------------------------------------------------
# Matplotlib helper
# ---------------------------------------------------------------------------
