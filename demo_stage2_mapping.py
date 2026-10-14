"""Stage 2/3: occupancy-grid mapping, from either the TRUE pose (Stage 2 -- isolates the
mapping half of SLAM, perfect localization assumed) or the noisy dead-reckoning ODOMETRY
estimate (Stage 3 -- shows concretely how pose drift corrupts a map, motivating Stage 4's
scan-matching correction). Same script, `--pose-source` selects which.

The lidar always senses from the robot's actual TRUE pose (that's physically correct --
the sensor doesn't care what the mapper believes); only the pose FED TO THE MAPPER differs
between modes. This isolates the comparison to exactly one variable.

Verification before trusting the map: samples points along every true wall (should end up
occupied, probability > 0.5) and the trajectory's own waypoints (known free space, should
end up probability < 0.5) and reports a numeric accuracy, not just a visual impression.

Usage:
    python3 demo_stage2_mapping.py --pose-source true --record media/stage2_mapping_true_pose.gif
    python3 demo_stage2_mapping.py --pose-source odom --record media/stage3_mapping_odom_drift.gif
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slam.world import World
from slam.robot import DiffDriveRobot
from slam.lidar import Lidar2D
from slam.occupancy_grid import OccupancyGrid
from slam.viz import render_frame, save_gif
from slam.trajectory import WAYPOINTS, run_loop


def wall_sample_points(world, n_per_wall=20, inset=0.05):
    """Points just INSIDE each wall segment (by `inset` m, along its own inward normal) --
    the cell a correct map should mark occupied is the wall's own location, not empty space
    half a cell away from it, so sampling exactly on the line (inset=0) would be too
    sensitive to which side of a cell boundary the wall happens to fall on."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-source", choices=["true", "odom"], default="true")
    ap.add_argument("--resolution", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--waypoint-tolerance", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--render-every", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    world = World.loop_corridor()
    robot = DiffDriveRobot(*WAYPOINTS[0], 0.0, rng=rng)
    lidar = Lidar2D(n_beams=360, max_range=8.0, rng=rng)
    grid = OccupancyGrid(world.bounds, resolution=args.resolution)

    frames = []

    def on_step(step, robot):
        pose_for_mapping = robot.true_pose if args.pose_source == "true" else robot.odom_pose
        ranges, hit = lidar.scan(robot.true_pose, world)   # sensor always reads the TRUE pose
        grid.update(pose_for_mapping, ranges, hit, lidar.angles)
        if args.record and step % args.render_every == 0:
            frames.append(render_frame(world, robot.true_pose, occupancy_grid=grid))

    n_steps = run_loop(robot, args.dt, args.max_steps, args.waypoint_tolerance, on_step)
    print(f"Finished after {n_steps} steps ({n_steps * args.dt:.1f}s simulated), "
          f"pose_source={args.pose_source}.")

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

    print(f"Wall-point accuracy (correctly marked occupied): {wall_accuracy:.3f} "
          f"({in_bounds.sum()}/{len(wall_pts)} sample points in-bounds)")
    print(f"Free-point accuracy (correctly marked free): {free_accuracy:.3f} "
          f"({len(free_pts)} sample points)")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        save_gif(frames, args.record, fps=1.0 / (args.dt * args.render_every))
        print(f"Saved {len(frames)} frames to {args.record}")


if __name__ == "__main__":
    main()
