"""Classical log-odds occupancy-grid mapping (the same representation GMapping/Hector
SLAM/Cartographer all build on): each cell accumulates evidence for "occupied" vs. "free"
in log-odds space (additive, numerically stable, trivially invertible to a probability),
from an inverse sensor model: every cell a lidar beam passes through on its way to a hit is
evidence of FREE space, and the cell at the hit itself is evidence of OCCUPIED space (a
beam that reaches max range with no hit contributes free evidence along its whole length
and no occupied evidence at all, since nothing was actually detected there).

Stage 2 scope: this module only MAPS, given whatever pose it's told (the caller decides
whether that's the true pose or a noisy estimate -- see demo_stage2_mapping.py's two modes).
Localization/correction is Stage 4's job, not this module's.
"""
import numpy as np


def _bresenham(x0: int, y0: int, x1: int, y1: int):
    """Integer Bresenham line, inclusive of both endpoints -- the standard way to rasterize
    a lidar beam onto which grid cells it actually passes through."""
    points = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points


class OccupancyGrid:
    def __init__(self, bounds, resolution: float = 0.1,
                 l_occ: float = 0.85, l_free: float = -0.4, l_clip: float = 10.0):
        """bounds: (xmin, xmax, ymin, ymax), same convention as World.bounds.
        l_occ/l_free: log-odds increments per observation (p_occ=0.7/p_free=0.4 in
        probability terms -- free evidence deliberately weaker than occupied, the standard
        asymmetric choice: a single stray reflection shouldn't erase a wall, but consistent
        free-space passes should still accumulate). l_clip: saturate accumulated log-odds so
        one area's long observation history doesn't take forever to update if it later turns
        out wrong."""
        xmin, xmax, ymin, ymax = bounds
        self.xmin, self.ymin = xmin, ymin
        self.resolution = resolution
        self.nx = int(np.ceil((xmax - xmin) / resolution))
        self.ny = int(np.ceil((ymax - ymin) / resolution))
        self.log_odds = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.l_occ, self.l_free, self.l_clip = l_occ, l_free, l_clip

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (int((x - self.xmin) / self.resolution), int((y - self.ymin) / self.resolution))

    def _in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.nx and 0 <= cy < self.ny

    def update(self, pose, ranges: np.ndarray, hit: np.ndarray, angles: np.ndarray) -> None:
        x, y, theta = pose
        cx0, cy0 = self.world_to_cell(x, y)
        beam_theta = theta + angles
        end_x = x + ranges * np.cos(beam_theta)
        end_y = y + ranges * np.sin(beam_theta)

        for i in range(len(ranges)):
            cx1, cy1 = self.world_to_cell(end_x[i], end_y[i])
            cells = _bresenham(cx0, cy0, cx1, cy1)
            for cx, cy in cells[:-1]:
                if self._in_bounds(cx, cy):
                    self.log_odds[cy, cx] = np.clip(
                        self.log_odds[cy, cx] + self.l_free, -self.l_clip, self.l_clip)
            cx, cy = cells[-1]
            if self._in_bounds(cx, cy):
                increment = self.l_occ if hit[i] else self.l_free
                self.log_odds[cy, cx] = np.clip(
                    self.log_odds[cy, cx] + increment, -self.l_clip, self.l_clip)

    def probability_map(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.log_odds))
