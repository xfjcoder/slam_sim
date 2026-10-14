"""Stage 4: close the basic SLAM loop. Odometry alone (Stage 3) drifts without bound and
corrupts the map; this adds the missing correction step -- at every tick, scan-match the
current lidar scan against the map built SO FAR (ICP, `slam/scan_match.py`) to pull the
pose estimate back toward the truth before using it to extend the map:

    predict (odometry delta) -> correct (ICP against the map) -> update the map
    -> repeat

Maintains a `belief_pose` separate from both `true_pose` (never used by the algorithm,
only for scoring) and the raw `odom_pose` (kept around only as the "what if we never
corrected anything" baseline, i.e. Stage 3 again, for comparison). Early in the lap the map
is too sparse to match against -- icp_2d reports `converged=False` and belief_pose just
follows the odometry prediction, identical to Stage 3, until enough wall has been seen.

Usage:
    python3 demo_stage4_slam.py --record media/stage4_slam_corrected.gif
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
from slam.scan_match import icp_2d, scan_to_local_points
from slam.viz import render_frame, save_gif
from slam.trajectory import WAYPOINTS, run_loop
from slam.eval import map_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--waypoint-tolerance", type=float, default=0.3)
    ap.add_argument("--occ-threshold", type=float, default=0.7)
    ap.add_argument("--max-correspondence-dist", type=float, default=0.5)
    ap.add_argument("--min-correspondences", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--render-every", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    world = World.loop_corridor()
    robot = DiffDriveRobot(*WAYPOINTS[0], 0.0, rng=rng)
    lidar = Lidar2D(n_beams=360, max_range=8.0, rng=rng)
    grid = OccupancyGrid(world.bounds, resolution=args.resolution)

    belief_pose = robot.true_pose.copy()   # known start pose -- standard SLAM assumption
    true_trail, odom_trail, belief_trail = [], [], []
    n_corrections = 0
    frames = []

    def on_step(step, robot):
        nonlocal belief_pose, n_corrections
        v_noisy, omega_noisy = robot.last_noisy_vw
        predicted = DiffDriveRobot._integrate(belief_pose, v_noisy, omega_noisy, args.dt)

        ranges, hit = lidar.scan(robot.true_pose, world)   # sensor always reads the TRUE pose
        local_pts = scan_to_local_points(ranges, hit, lidar.angles)
        map_pts = grid.occupied_points(threshold=args.occ_threshold)
        corrected, converged, _ = icp_2d(
            local_pts, map_pts, predicted,
            max_correspondence_dist=args.max_correspondence_dist,
            min_correspondences=args.min_correspondences)
        belief_pose = corrected if converged else predicted
        if converged:
            n_corrections += 1

        grid.update(belief_pose, ranges, hit, lidar.angles)
        true_trail.append(robot.true_pose[:2].copy())
        odom_trail.append(robot.odom_pose[:2].copy())
        belief_trail.append(belief_pose[:2].copy())
        if args.record and step % args.render_every == 0:
            frames.append(render_frame(
                world, robot.true_pose, occupancy_grid=grid,
                true_trail=np.array(true_trail) if len(true_trail) > 1 else None,
                odom_trail=np.array(odom_trail) if len(odom_trail) > 1 else None,
                belief_trail=np.array(belief_trail) if len(belief_trail) > 1 else None))

    n_steps = run_loop(robot, args.dt, args.max_steps, args.waypoint_tolerance, on_step)
    print(f"Finished after {n_steps} steps ({n_steps * args.dt:.1f}s simulated). "
          f"ICP corrected {n_corrections}/{n_steps} steps "
          f"({100.0 * n_corrections / n_steps:.0f}%).")

    belief_drift = float(np.hypot(*(belief_pose[:2] - robot.true_pose[:2])))
    odom_drift = float(np.hypot(*(robot.odom_pose[:2] - robot.true_pose[:2])))
    print(f"Final position drift: belief (SLAM-corrected) = {belief_drift:.3f}m, "
          f"raw odometry = {odom_drift:.3f}m.")

    wall_accuracy, free_accuracy, n_wall_in_bounds, n_wall_total = map_accuracy(grid, world)
    print(f"Wall-point accuracy (correctly marked occupied): {wall_accuracy:.3f} "
          f"({n_wall_in_bounds}/{n_wall_total} sample points in-bounds)")
    print(f"Free-point accuracy (correctly marked free): {free_accuracy:.3f} (4 sample points)")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        save_gif(frames, args.record, fps=1.0 / (args.dt * args.render_every))
        print(f"Saved {len(frames)} frames to {args.record}")


if __name__ == "__main__":
    main()
