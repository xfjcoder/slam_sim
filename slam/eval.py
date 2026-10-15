"""Shared map-accuracy evaluation, used identically by every mapping-stage demo (Stage
2/3/4) so each reports a number computed the same way, not its own slightly-different
one-off -- the whole point of comparing accuracy *across* stages.
"""
import numpy as np

from .trajectory import WAYPOINTS


def wall_sample_points(world, n_per_wall: int = 20, inset: float = 0.05) -> np.ndarray:
    """Points just INSIDE each wall segment (by `inset` m, along its own inward normal) --
    the cell a correct map should mark occupied is the wall's own location, not empty
    space half a cell away from it, so sampling exactly on the line (inset=0) would be
    too sensitive to which side of a cell boundary the wall happens to fall on."""
    pts = []
    cx, cy = (world.bounds[0] + world.bounds[1]) / 2, (world.bounds[2] + world.bounds[3]) / 2
    for x1, y1, x2, y2 in world.walls:
        t = np.linspace(0.0, 1.0, n_per_wall)
        xs, ys = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        # inward normal: perpendicular to the wall, pointing toward the map's own centroid
        # (works for this axis-aligned rectangle/island layout, not a general polygon).
        nx, ny = -(y2 - y1), (x2 - x1)
        norm = np.hypot(nx, ny) + 1e-9
        nx, ny = nx / norm, ny / norm
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if (mx + nx * inset - cx) ** 2 + (my + ny * inset - cy) ** 2 > \
           (mx - nx * inset - cx) ** 2 + (my - ny * inset - cy) ** 2:
            nx, ny = -nx, -ny
        pts.append(np.stack([xs + nx * inset, ys + ny * inset], axis=1))
    return np.concatenate(pts, axis=0)


def _wall_sample_points_both_sides(world, n_per_wall: int = 20, inset: float = 0.05):
    """Like `wall_sample_points`, but returns the pair of candidate points on EITHER side
    of each wall instead of guessing one via the "toward the centroid" heuristic. Needed
    for layouts (e.g. Stage 7's office, with walls open to room on BOTH sides) where that
    single-centroid guess has no reason to match whichever side a zero-thickness wall's
    hit actually got rasterized onto -- unlike Stage 1-6's loop_corridor, where every wall
    is either on the convex outer boundary (robot always approaches from the one inside
    direction) or the heuristic happens to agree with it."""
    pts_a, pts_b = [], []
    for x1, y1, x2, y2 in world.walls:
        t = np.linspace(0.0, 1.0, n_per_wall)
        xs, ys = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        nx, ny = -(y2 - y1), (x2 - x1)
        norm = np.hypot(nx, ny) + 1e-9
        nx, ny = nx / norm, ny / norm
        pts_a.append(np.stack([xs + nx * inset, ys + ny * inset], axis=1))
        pts_b.append(np.stack([xs - nx * inset, ys - ny * inset], axis=1))
    return np.concatenate(pts_a, axis=0), np.concatenate(pts_b, axis=0)


def map_accuracy(grid, world, free_points=None, wall_both_sides: bool = False):
    """Returns (wall_accuracy, free_accuracy, n_wall_in_bounds, n_wall_total).

    `free_points`: known-free (x, y) sample points to score against -- defaults to
    Stage 1-6's own loop-corridor WAYPOINTS corners, but callers with a different
    ground-truth layout (e.g. Stage 7's office floorplan) should pass their own, since
    those corners aren't guaranteed free space in a different world.

    `wall_both_sides`: score each wall point against whichever of its two adjacent cells
    is more occupied, instead of a single centroid-guessed side -- see
    `_wall_sample_points_both_sides`. Stage 2-4's numbers depend on the single-sided
    default, so this stays opt-in rather than changing their behavior."""
    prob = grid.probability_map()

    def _lookup(pts):
        cells = np.array([grid.world_to_cell(x, y) for x, y in pts])
        in_bounds = ((cells[:, 0] >= 0) & (cells[:, 0] < grid.nx) &
                     (cells[:, 1] >= 0) & (cells[:, 1] < grid.ny))
        p = np.full(len(pts), np.nan)
        p[in_bounds] = prob[cells[in_bounds, 1], cells[in_bounds, 0]]
        return p, in_bounds

    if wall_both_sides:
        pts_a, pts_b = _wall_sample_points_both_sides(world)
        prob_a, in_a = _lookup(pts_a)
        prob_b, in_b = _lookup(pts_b)
        in_bounds = in_a | in_b
        best = np.fmax(np.where(in_a, prob_a, -np.inf), np.where(in_b, prob_b, -np.inf))
        wall_prob = best[in_bounds]
        n_wall_total = len(pts_a)
    else:
        wall_pts = wall_sample_points(world)
        wall_prob, in_bounds = _lookup(wall_pts)
        wall_prob = wall_prob[in_bounds]
        n_wall_total = len(wall_pts)
    wall_accuracy = float((wall_prob > 0.5).mean()) if len(wall_prob) else float("nan")

    free_pts = free_points if free_points is not None else WAYPOINTS[:-1]
    free_cells = np.array([grid.world_to_cell(x, y) for x, y in free_pts])
    free_prob = prob[free_cells[:, 1], free_cells[:, 0]]
    free_accuracy = float((free_prob < 0.5).mean())

    return wall_accuracy, free_accuracy, int(in_bounds.sum()), n_wall_total
