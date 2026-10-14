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
from slam.eval import map_accuracy


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

    wall_accuracy, free_accuracy, n_wall_in_bounds, n_wall_total = map_accuracy(grid, world)
    print(f"Wall-point accuracy (correctly marked occupied): {wall_accuracy:.3f} "
          f"({n_wall_in_bounds}/{n_wall_total} sample points in-bounds)")
    print(f"Free-point accuracy (correctly marked free): {free_accuracy:.3f} "
          f"(4 sample points)")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        save_gif(frames, args.record, fps=1.0 / (args.dt * args.render_every))
        print(f"Saved {len(frames)} frames to {args.record}")


if __name__ == "__main__":
    main()
