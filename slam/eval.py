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


def map_accuracy(grid, world):
    """Returns (wall_accuracy, free_accuracy, n_wall_in_bounds, n_wall_total)."""
    prob = grid.probability_map()

    wall_pts = wall_sample_points(world)
    wall_cells = np.array([grid.world_to_cell(x, y) for x, y in wall_pts])
    in_bounds = ((wall_cells[:, 0] >= 0) & (wall_cells[:, 0] < grid.nx) &
                 (wall_cells[:, 1] >= 0) & (wall_cells[:, 1] < grid.ny))
    wall_prob = prob[wall_cells[in_bounds, 1], wall_cells[in_bounds, 0]]
    wall_accuracy = float((wall_prob > 0.5).mean()) if len(wall_prob) else float("nan")

    free_pts = WAYPOINTS[:-1]   # the 4 corner waypoints themselves are known free space
    free_cells = np.array([grid.world_to_cell(x, y) for x, y in free_pts])
    free_prob = prob[free_cells[:, 1], free_cells[:, 0]]
    free_accuracy = float((free_prob < 0.5).mean())

    return wall_accuracy, free_accuracy, int(in_bounds.sum()), len(wall_pts)
