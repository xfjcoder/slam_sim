"""Ground-truth 2D world: a set of line-segment walls the lidar sensor ray-casts against.

Deliberately NOT a MuJoCo/physics-engine world -- this is a simplified kinematic SLAM
sandbox (robot has no dynamics, mass, or balance to worry about), so the simplest
representation that supports ray-casting is the right one, not a heavier one.

The default layout is a rectangular outer boundary with a rectangular "island" obstacle in
the middle, leaving a loop-shaped corridor around it -- deliberately chosen (not just a
single room) so a robot driving around the loop eventually revisits a familiar area, the
basic scenario loop-closure SLAM is built to handle. Stage 1 doesn't need a loop yet, but
building the world this way from the start avoids having to redo it later.
"""
import numpy as np


class World:
    def __init__(self, walls: np.ndarray, bounds: tuple[float, float, float, float]):
        """walls: (M, 4) array of (x1, y1, x2, y2) line segments.
        bounds: (xmin, xmax, ymin, ymax), for plotting/sanity checks only -- ray-casting
        itself only ever looks at `walls`."""
        self.walls = np.asarray(walls, dtype=np.float64)
        self.bounds = bounds

    @classmethod
    def loop_corridor(cls, outer=(12.0, 8.0), island=(4.0, 2.0, 8.0, 6.0)):
        """outer: (width, height) of the outer rectangle, corner at (0, 0).
        island: (x1, y1, x2, y2) of the rectangular obstacle in the middle."""
        w, h = outer
        ix1, iy1, ix2, iy2 = island
        outer_walls = [
            (0, 0, w, 0), (w, 0, w, h), (w, h, 0, h), (0, h, 0, 0),
        ]
        island_walls = [
            (ix1, iy1, ix2, iy1), (ix2, iy1, ix2, iy2),
            (ix2, iy2, ix1, iy2), (ix1, iy2, ix1, iy1),
        ]
        return cls(np.array(outer_walls + island_walls, dtype=np.float64),
                    bounds=(0.0, w, 0.0, h))

    def is_free(self, x: float, y: float, margin: float = 0.2) -> bool:
        """True if (x, y) is inside the outer boundary, outside the island, and at least
        `margin` away from every wall -- used to keep a scripted/random trajectory from
        driving the robot through a wall (this sim has no collision physics of its own)."""
        xmin, xmax, ymin, ymax = self.bounds
        if not (xmin + margin <= x <= xmax - margin and ymin + margin <= y <= ymax - margin):
            return False
        # Island check assumes the specific axis-aligned rectangle loop_corridor builds --
        # fine for this sandbox's one default layout, not a general point-in-polygon test.
        island = self.walls[4:8]
        ix1, iy1 = island[0, 0], island[0, 1]
        ix2, iy2 = island[1, 2], island[2, 3]
        if (ix1 - margin <= x <= ix2 + margin) and (iy1 - margin <= y <= iy2 + margin):
            return False
        return True
