"""2D lidar sensor: ray-casts a fan of beams from the robot's pose against the world's wall
segments, vectorized over (beam, wall) pairs with numpy (trivial problem size here -- a few
hundred beams x a handful of walls -- so no need for anything fancier than brute-force
ray-segment intersection).

Deliberately "richer sensing" than this project's quadruped-SLAM precursor (the 9-ray fan
built for the Go2 walking policy, see go1_rl_walk/HISTORY.md Stage 7/10): a real 360-degree
scan at this resolution is the standard sensor classic occupancy-grid SLAM (GMapping, Hector
SLAM, Cartographer) is built around, not a sparse near-field fan meant for footstep planning.
"""
import numpy as np


def _cross2(ax, ay, bx, by):
    return ax * by - ay * bx


class Lidar2D:
    def __init__(self, n_beams: int = 360, max_range: float = 8.0,
                 range_noise_std: float = 0.02, rng: np.random.Generator | None = None):
        self.n_beams = n_beams
        self.max_range = max_range
        self.range_noise_std = range_noise_std
        self.angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
        self._rng = rng if rng is not None else np.random.default_rng()

    def scan(self, pose: np.ndarray, world) -> tuple[np.ndarray, np.ndarray]:
        """pose: (x, y, theta). Returns (ranges, hit) each shape (n_beams,) -- ranges is
        max_range wherever hit is False (no wall within range), noise only added to actual
        hits (a real lidar's max-range return isn't a noisy measurement of anything)."""
        x, y, theta = pose
        beam_theta = theta + self.angles
        rx, ry = np.cos(beam_theta), np.sin(beam_theta)   # (N,) unit beam directions

        walls = world.walls
        p1x, p1y, p2x, p2y = walls[:, 0], walls[:, 1], walls[:, 2], walls[:, 3]
        sx, sy = p2x - p1x, p2y - p1y          # (M,) segment direction
        dx, dy = p1x - x, p1y - y              # (M,) wall start relative to ray origin

        denom = _cross2(rx[:, None], ry[:, None], sx[None, :], sy[None, :])   # (N, M)
        t_num = _cross2(dx, dy, sx, sy)                                       # (M,)
        u_num = _cross2(dx[None, :], dy[None, :], rx[:, None], ry[:, None])   # (N, M)

        with np.errstate(divide="ignore", invalid="ignore"):
            t = t_num[None, :] / denom
            u = u_num / denom
        valid = (np.abs(denom) > 1e-12) & (t >= 1e-9) & (u >= 0.0) & (u <= 1.0)
        t = np.where(valid, t, np.inf)

        ranges = np.min(t, axis=1)
        hit = ranges <= self.max_range
        ranges = np.where(hit, ranges, self.max_range)
        ranges = ranges + np.where(hit, self._rng.normal(0.0, self.range_noise_std, self.n_beams), 0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)
        return ranges, hit

    def scan_points_world(self, pose, world):
        """Convenience: (ranges, hit) -> (N, 2) hit-point world coordinates, hits only --
        what a mapping algorithm actually consumes (not the raw ranges array)."""
        x, y, theta = pose
        ranges, hit = self.scan(pose, world)
        beam_theta = theta + self.angles
        px = x + ranges * np.cos(beam_theta)
        py = y + ranges * np.sin(beam_theta)
        return np.stack([px[hit], py[hit]], axis=1)
