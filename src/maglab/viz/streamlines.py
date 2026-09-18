"""
Jobard & Lefer (1997) evenly spaced streamline algorithm for 2D regular grids.

Accepts vector fields of shape ``(ny, nx, vectordim)`` with ``vectordim >= 2``.
Streamlines follow the in-plane components ``(arr[..., 0], arr[..., 1])``; when
a third component is present it is sampled along each streamline and returned
as a per-point scalar, ready for colouring lines by (for example) ``m_z``.

Reference:
    Jobard, B., & Lefer, W. (1997). Creating Evenly-Spaced Streamlines of
    Arbitrary Density. Visualization in Scientific Computing '97, 43-55.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.interpolate import RegularGridInterpolator

__all__ = ["evenly_spaced_streamlines", "plot_streamlines"]


# ---------------------------------------------------------------------------
# Distance lookup grid
# ---------------------------------------------------------------------------


class _LookupGrid:
    """Spatial hash grid for fast nearest-streamline distance queries.

    The domain is divided into square cells of side ``cell_size``. For a query
    point ``q`` in cell ``(i, j)``, the 3x3 block of cells centred on ``(i, j)``
    extends at least ``cell_size`` beyond ``q`` along every axis, so every
    stored point within Euclidean distance ``cell_size`` of ``q`` is guaranteed
    to lie inside that block. The caller must therefore choose
    ``cell_size >= max(d_sep, d_test)``.
    """

    __slots__ = ("cell_size", "cells")

    def __init__(self, cell_size: float):
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.cell_size = float(cell_size)
        self.cells: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        # floor, not int(): int() truncates towards zero, which would merge
        # cell -1 into cell 0 for the (transiently out-of-bounds) candidates.
        return (
            int(np.floor(x / self.cell_size)),
            int(np.floor(y / self.cell_size)),
        )

    def add_point(self, point: np.ndarray) -> None:
        x = float(point[0])
        y = float(point[1])
        self.cells.setdefault(self._cell(x, y), []).append((x, y))

    def add_streamline(self, points: np.ndarray) -> None:
        for p in points:
            self.add_point(p)

    def is_valid(self, point: np.ndarray, d: float) -> bool:
        """Return True if ``point`` is at least ``d`` away from all stored points."""
        x = float(point[0])
        y = float(point[1])
        cx, cy = self._cell(x, y)
        d2 = d * d
        cells = self.cells
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                bucket = cells.get((cx + dx, cy + dy))
                if bucket is None:
                    continue
                for px, py in bucket:
                    ddx = x - px
                    ddy = y - py
                    if ddx * ddx + ddy * ddy < d2:
                        return False
        return True


# ---------------------------------------------------------------------------
# Vector field interpolation
# ---------------------------------------------------------------------------


class _Field:
    """Bilinear interpolator over a ``(ny, nx, vectordim)`` array.

    Queries are made in ``(x, y)`` pixel coordinates and clipped to the domain,
    so intermediate Runge-Kutta stages can never extrapolate wildly.
    """

    def __init__(self, arr: np.ndarray):
        arr = np.ascontiguousarray(arr, dtype=float)
        if arr.ndim != 3 or arr.shape[2] < 2:
            raise ValueError(
                f"arr must have shape (ny, nx, vectordim) with vectordim >= 2; got {arr.shape}"
            )
        self.ny, self.nx, self.vectordim = arr.shape
        if self.ny < 2 or self.nx < 2:
            raise ValueError("arr must be at least 2x2 in the spatial dimensions")

        self._interp = RegularGridInterpolator(
            (np.arange(self.ny, dtype=float), np.arange(self.nx, dtype=float)),
            arr,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        # Reference in-plane magnitude, used to turn the relative
        # `min_in_plane` threshold into an absolute one.
        self.max_in_plane = float(np.max(np.hypot(arr[..., 0], arr[..., 1])))

    def sample(self, xy: np.ndarray) -> np.ndarray:
        """Sample all components at ``(m, 2)`` array of (x, y) points -> ``(m, k)``."""
        pts = np.atleast_2d(np.asarray(xy, dtype=float))
        qy = np.clip(pts[:, 1], 0.0, self.ny - 1.0)
        qx = np.clip(pts[:, 0], 0.0, self.nx - 1.0)
        return self._interp(np.column_stack((qy, qx)))

    def direction(self, x: float, y: float, sign: int, min_mag: float) -> np.ndarray:
        """Unit in-plane direction at (x, y), or a zero vector below threshold."""
        qy = 0.0 if y < 0.0 else (self.ny - 1.0 if y > self.ny - 1.0 else y)
        qx = 0.0 if x < 0.0 else (self.nx - 1.0 if x > self.nx - 1.0 else x)
        vec = self._interp(np.array([[qy, qx]]))[0]
        u = vec[0]
        v = vec[1]
        mag = np.hypot(u, v)
        if mag < min_mag:
            return np.zeros(2)
        return np.array([sign * u / mag, sign * v / mag])

    def in_bounds(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.nx - 1.0 and 0.0 <= y <= self.ny - 1.0


# ---------------------------------------------------------------------------
# Streamline integration (RK4, arc-length parameterised)
# ---------------------------------------------------------------------------


def _integrate(
    seed: np.ndarray,
    field: _Field,
    step: float,
    max_steps: int,
    d_test: float,
    min_mag: float,
    lookup: _LookupGrid,
    self_lag: int,
    self_prefix: np.ndarray | None = None,
    direction: int = 1,
) -> np.ndarray:
    """Integrate one streamline from ``seed`` in ``direction`` (+1 / -1).

    Integration stops when the point leaves the domain, the in-plane magnitude
    drops below ``min_mag``, the point comes within ``d_test`` of an existing
    streamline, or the line comes within ``d_test`` of itself (which is what
    terminates closed loops and spirals in a vortex field).

    ``self_prefix`` is the already-integrated half of the same streamline; its
    points participate in the self-intersection test.
    """
    points = [np.asarray(seed, dtype=float).copy()]
    pos = points[0].copy()

    # Separate grid for the streamline's own points, populated with a lag of
    # `self_lag` samples so the line does not immediately trip over its own
    # tail.
    self_grid = _LookupGrid(lookup.cell_size)
    if self_prefix is not None and len(self_prefix) > self_lag:
        self_grid.add_streamline(self_prefix[self_lag:])
    pending: deque[np.ndarray] = deque()

    for _ in range(max_steps):
        k1 = field.direction(pos[0], pos[1], direction, min_mag)
        if not k1.any():
            break
        h = 0.5 * step
        p2 = pos + h * k1
        k2 = field.direction(p2[0], p2[1], direction, min_mag)
        p3 = pos + h * k2
        k3 = field.direction(p3[0], p3[1], direction, min_mag)
        p4 = pos + step * k3
        k4 = field.direction(p4[0], p4[1], direction, min_mag)
        if not (k2.any() and k3.any() and k4.any()):
            break

        next_pos = pos + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not field.in_bounds(next_pos[0], next_pos[1]):
            break
        if not lookup.is_valid(next_pos, d_test):
            break
        if not self_grid.is_valid(next_pos, d_test):
            break

        pos = next_pos
        points.append(pos.copy())

        pending.append(pos)
        if len(pending) > self_lag:
            self_grid.add_point(pending.popleft())

    return np.asarray(points, dtype=float)


def _integrate_both_directions(
    seed: np.ndarray,
    field: _Field,
    step: float,
    max_steps: int,
    d_test: float,
    min_mag: float,
    lookup: _LookupGrid,
    self_lag: int,
) -> np.ndarray:
    """Integrate forward and backward and join into a single streamline."""
    fwd = _integrate(
        seed,
        field,
        step,
        max_steps,
        d_test,
        min_mag,
        lookup,
        self_lag,
        self_prefix=None,
        direction=1,
    )
    bwd = _integrate(
        seed,
        field,
        step,
        max_steps,
        d_test,
        min_mag,
        lookup,
        self_lag,
        self_prefix=fwd,
        direction=-1,
    )
    if len(bwd) > 1:
        return np.concatenate([bwd[1:][::-1], fwd])
    return fwd


# ---------------------------------------------------------------------------
# Candidate seed generation
# ---------------------------------------------------------------------------


def _candidate_seeds(streamline: np.ndarray, d_sep: float, stride: int) -> list[np.ndarray]:
    """Candidate seeds at +/- d_sep normal to the streamline, every `stride` points."""
    candidates: list[np.ndarray] = []
    n = len(streamline)
    for i in range(0, n - 1, max(stride, 1)):
        tangent = streamline[i + 1] - streamline[i]
        norm = float(np.hypot(tangent[0], tangent[1]))
        if norm < 1e-10:
            continue
        tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0]])
        base = streamline[i]
        candidates.append(base + d_sep * normal)
        candidates.append(base - d_sep * normal)
    return candidates


def _sample_lattice(field: _Field, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Raster lattice of points at ``spacing`` and their in-plane magnitudes."""
    xs = np.arange(0.0, field.nx - 1.0 + 1e-9, spacing)
    ys = np.arange(0.0, field.ny - 1.0 + 1e-9, spacing)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack((gx.ravel(), gy.ravel()))
    vals = field.sample(pts)
    return pts, np.hypot(vals[:, 0], vals[:, 1])


def _fallback_seed(field: _Field) -> np.ndarray:
    """Pixel with the largest in-plane magnitude, as an (x, y) point."""
    pts, mag = _sample_lattice(field, 1.0)
    return pts[int(np.argmax(mag))]


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
    min_in_plane: float = 1e-3,
    scalar_index: int | None = 2,
    sweep: bool = True,
    sweep_spacing: float | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    """Generate evenly spaced streamlines using the Jobard & Lefer algorithm.

    Parameters
    ----------
    arr : np.ndarray
        Vector field of shape ``(ny, nx, vectordim)`` with ``vectordim >= 2``.
        Components 0 and 1 are the in-plane ``(u, v)`` that the streamlines
        follow; component ``scalar_index`` (if present) is carried along.
    d_sep : float
        Target separation between streamlines, in pixels. Controls density.
    d_test : float, optional
        Distance from an existing streamline at which integration stops.
        Defaults to ``d_sep / 2``. Must satisfy ``0 < d_test <= d_sep``.
    step : float
        Integration step size in pixels (arc length, since the field is
        normalised to unit length before stepping).
    max_steps : int
        Maximum integration steps per direction.
    seed : np.ndarray, optional
        Starting point as ``(x, y)`` in pixel coordinates. Defaults to the
        centre of the domain; if that seed produces no streamline (a vortex
        core, say), the pixel of largest in-plane magnitude is used instead.
    min_in_plane : float
        Integration stops where the in-plane magnitude falls below this
        fraction of the field's maximum in-plane magnitude. This is what keeps
        streamlines out of regions that are essentially fully out-of-plane.
    scalar_index : int or None
        Which component to sample along each streamline. Default 2 (``m_z``
        for a magnetisation field). Set to ``None``, or leave the field with
        ``vectordim == 2``, to skip it.
    sweep : bool
        After the seeding front is exhausted, sweep the domain for regions it
        never reached and restart there. Required for fields made of
        disconnected elements — an array of nanomagnet islands separated by
        vacuum, say — because the published algorithm only ever seeds normal
        to an existing streamline and so cannot cross a gap wider than
        ``d_sep``. Costs little on a connected field; turn it off for the
        strict Jobard & Lefer behaviour.
    sweep_spacing : float, optional
        Lattice spacing for that sweep, in pixels. Defaults to ``d_sep / 2``.
        Must be small enough to land inside the smallest element you care
        about.

    Returns
    -------
    streamlines : list of np.ndarray
        Each element is an ``(N, 2)`` array of ``(x, y)`` pixel coordinates.
    scalars : list of (np.ndarray or None)
        Parallel list; each element is an ``(N,)`` array of the sampled
        component, or ``None`` when no scalar is available.

    Notes
    -----
    Coordinates are returned in pixel units with ``x`` along ``nx`` (axis 1 of
    ``arr``) and ``y`` along ``ny`` (axis 0), which is what ``imshow`` and
    ``pcolormesh`` with default extents expect.
    """
    if d_sep <= 0:
        raise ValueError("d_sep must be positive")
    if d_test is None:
        d_test = d_sep / 2.0
    if not (0 < d_test <= d_sep):
        raise ValueError("d_test must satisfy 0 < d_test <= d_sep")
    if step <= 0:
        raise ValueError("step must be positive")

    field = _Field(arr)

    use_scalar = scalar_index is not None and field.vectordim > scalar_index
    min_mag = min_in_plane * field.max_in_plane
    if field.max_in_plane == 0.0:
        return [], []

    if seed is None:
        seed = np.array([(field.nx - 1) / 2.0, (field.ny - 1) / 2.0])
    else:
        seed = np.asarray(seed, dtype=float).reshape(2)

    # Cell size must cover the largest radius ever queried.
    lookup = _LookupGrid(max(d_sep, d_test))
    # Lag before a streamline starts testing against its own points: enough
    # arc length that the immediate neighbourhood of the current point is
    # excluded, but short enough to catch a genuine loop closure.
    self_lag = int(np.ceil(2.0 * d_test / step)) + 2

    streamlines: list[np.ndarray] = []
    queue: deque[np.ndarray] = deque()
    stride = max(1, int(round(d_sep / step)))

    def grow(start: np.ndarray) -> bool:
        line = _integrate_both_directions(
            start, field, step, max_steps, d_test, min_mag, lookup, self_lag
        )
        if len(line) < 2:
            return False
        lookup.add_streamline(line)
        streamlines.append(line)
        queue.append(line)
        return True

    def drain() -> None:
        """Run the Jobard & Lefer seeding front until it is exhausted."""
        while queue:
            current = queue.popleft()
            for candidate in _candidate_seeds(current, d_sep, stride):
                if not field.in_bounds(candidate[0], candidate[1]):
                    continue
                if not lookup.is_valid(candidate, d_sep):
                    continue
                grow(candidate)

    if not grow(seed):
        grow(_fallback_seed(field))
    drain()

    if sweep:
        # The front propagates only by stepping d_sep sideways off an existing
        # streamline, so it cannot reach a region separated from the seed by
        # more than d_sep of dead field. Restart it anywhere still uncovered.
        pts, mag = _sample_lattice(
            field, sweep_spacing if sweep_spacing is not None else d_sep / 2.0
        )
        for candidate in pts[mag >= min_mag]:
            if not lookup.is_valid(candidate, d_sep):
                continue
            if grow(candidate):
                drain()

    if not use_scalar:
        return streamlines, [None] * len(streamlines)

    scalars: list[np.ndarray | None] = []
    for line in streamlines:
        scalars.append(field.sample(line)[:, scalar_index])
    return streamlines, scalars


# ---------------------------------------------------------------------------
# Matplotlib helper
# ---------------------------------------------------------------------------


def _arclength(line: np.ndarray) -> np.ndarray:
    """Cumulative arc length along an ``(N, 2)`` polyline."""
    seg = np.hypot(np.diff(line[:, 0]), np.diff(line[:, 1]))
    return np.concatenate(([0.0], np.cumsum(seg)))


def _arrow_anchors(
    line: np.ndarray, spacing: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arrow placement along a streamline, evenly spaced by arc length.

    Returns ``(points, unit_tangents, arc_positions)``. With ``spacing=None``
    a single arrow is placed at the midpoint of the line.
    """
    s = _arclength(line)
    total = float(s[-1])
    empty = (np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
    if total <= 0.0:
        return empty

    if spacing is None or spacing <= 0.0:
        targets = np.array([0.5 * total])
    else:
        # Evenly distributed, with half-spacing margins at both ends so no
        # arrow head hangs off the end of the line.
        n = max(1, int(round(total / spacing)))
        targets = (np.arange(n) + 0.5) * (total / n)

    px = np.interp(targets, s, line[:, 0])
    py = np.interp(targets, s, line[:, 1])
    points = np.column_stack((px, py))

    # Tangent from the segment each anchor falls in.
    j = np.clip(np.searchsorted(s, targets, side="right") - 1, 0, len(line) - 2)
    tangents = line[j + 1] - line[j]
    norms = np.hypot(tangents[:, 0], tangents[:, 1])
    ok = norms > 1e-12
    if not ok.any():
        return empty
    return points[ok], tangents[ok] / norms[ok, None], targets[ok]


def draw_streamlines(
    streamlines,
    scalars=None,
    ax=None,
    linewidth: float = 1.0,
    color: str = "k",
    cmap: str = "coolwarm",
    clim: tuple[float, float] | None = None,
    arrows: bool = False,
    arrow_every: int = 1,
    arrow_spacing: float | None = None,
    arrowstyle: str = "-|>",
    arrow_size: float = 10.0,
    arrow_color=None,
    arrow_kwargs: dict | None = None,
    **kwargs,
):
    """Draw streamlines, colour-mapped by their carried scalar when available.

    Parameters
    ----------
    streamlines, scalars
        As returned by :func:`evenly_spaced_streamlines`. ``scalars`` may be
        ``None`` or a list of ``None`` for an unmapped (single-colour) plot.
    ax : matplotlib Axes, optional
        Defaults to the current axes.
    linewidth, color, cmap, clim
        Line appearance. ``clim`` defaults to a symmetric range, which is the
        sensible choice for a signed component such as ``m_z`` with a diverging
        colourmap.
    arrows : bool
        Draw direction arrows along the streamlines.
    arrow_every : int
        Place arrows on every ``arrow_every``-th streamline. Streamlines are
        ordered by the seeding front, so consecutive lines are neighbours in
        space and this thins the arrows spatially rather than at random.
    arrow_spacing : float, optional
        Arc length in pixels between arrows on a line. ``None`` (default)
        places a single arrow at the midpoint of each selected line.
    arrowstyle, arrow_size
        Passed to :class:`~matplotlib.patches.FancyArrowPatch` as
        ``arrowstyle`` and ``mutation_scale``.
    arrow_color : optional
        Overrides the arrow colour. By default arrows take the colourmapped
        value at their own position, so they match the line they sit on; with
        no scalar they fall back to ``color``.
    arrow_kwargs : dict, optional
        Extra keyword arguments for each ``FancyArrowPatch``.
    **kwargs
        Extra keyword arguments for the ``LineCollection``.

    Returns
    -------
    matplotlib.collections.LineCollection
        The line collection, for attaching a colourbar. The arrow patches are
        available on it as the ``.arrows`` attribute.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    has_scalar = scalars is not None and any(s is not None for s in scalars)

    segments = []
    values = []
    for i, line in enumerate(streamlines):
        if len(line) < 2:
            continue
        segs = np.stack([line[:-1], line[1:]], axis=1)
        segments.append(segs)
        if has_scalar:
            s = scalars[i]
            values.append(np.zeros(len(segs)) if s is None else 0.5 * (s[:-1] + s[1:]))

    if not segments:
        return None

    segments = np.concatenate(segments, axis=0)

    if has_scalar:
        values = np.concatenate(values)
        if clim is None:
            vmax = float(np.max(np.abs(values))) or 1.0
            clim = (-vmax, vmax)
        lc = LineCollection(segments, cmap=cmap, linewidth=linewidth, **kwargs)
        lc.set_array(values)
        lc.set_clim(*clim)
    else:
        lc = LineCollection(segments, colors=color, linewidth=linewidth, **kwargs)

    ax.add_collection(lc)

    arrow_patches: list = []
    if arrows:
        akw = {"shrinkA": 0.0, "shrinkB": 0.0, "linewidth": linewidth}
        akw.update(arrow_kwargs or {})
        akw.setdefault("zorder", lc.get_zorder() + 1)

        norm = Normalize(*clim) if has_scalar else None
        colormap = lc.get_cmap() if has_scalar else None

        for i in range(0, len(streamlines), max(1, int(arrow_every))):
            line = streamlines[i]
            if len(line) < 2:
                continue
            points, tangents, positions = _arrow_anchors(line, arrow_spacing)
            if len(points) == 0:
                continue

            if arrow_color is not None:
                colours = [arrow_color] * len(points)
            elif has_scalar and scalars[i] is not None:
                s = scalars[i]
                vals = np.interp(positions, _arclength(line), s)
                colours = colormap(norm(vals))
            else:
                colours = [color] * len(points)

            # A short segment centred on each anchor; it only sets the
            # orientation, the visible size comes from `mutation_scale`.
            half = 0.5 * max(np.median(np.hypot(*np.diff(line, axis=0).T)), 1e-9)
            tails = points - half * tangents
            heads = points + half * tangents

            for tail, head, col in zip(tails, heads, colours):
                patch = FancyArrowPatch(
                    posA=tuple(tail),
                    posB=tuple(head),
                    arrowstyle=arrowstyle,
                    mutation_scale=arrow_size,
                    color=col,
                    **akw,
                )
                ax.add_patch(patch)
                arrow_patches.append(patch)

    lc.arrows = arrow_patches
    ax.autoscale_view()
    ax.set_aspect("equal")
    return lc
