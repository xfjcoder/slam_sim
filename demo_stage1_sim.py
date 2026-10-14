"""Stage 1: the simulator core (world + kinematic robot + lidar), verified by direct
visual inspection, not just unit numbers -- same "always generate a GIF for a new behavior"
discipline the quadruped project this grew out of used throughout.

Scripts the robot around a loop (see slam/trajectory.py) once around the island obstacle.
No mapping/localization algorithm yet -- this stage only has to prove the simulator itself
is right: the lidar hits should visibly hug the walls, and the odometry estimate (orange,
dashed) should visibly drift away from the true path (green, solid) over the lap -- that
drift is the whole reason later stages exist.

Usage:
    python3 demo_stage1_sim.py --record media/stage1_sim_and_drift.gif
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slam.world import World
from slam.robot import DiffDriveRobot
from slam.lidar import Lidar2D
from slam.viz import render_frame, save_gif
from slam.trajectory import WAYPOINTS, run_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--waypoint-tolerance", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", type=str, default="media/stage1_sim_and_drift.gif")
    ap.add_argument("--render-every", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    world = World.loop_corridor()
    robot = DiffDriveRobot(*WAYPOINTS[0], 0.0, rng=rng)
    lidar = Lidar2D(n_beams=360, max_range=8.0, rng=rng)

    true_trail, odom_trail = [robot.true_pose[:2].copy()], [robot.odom_pose[:2].copy()]
    frames = []

    def on_step(step, robot):
        true_trail.append(robot.true_pose[:2].copy())
        odom_trail.append(robot.odom_pose[:2].copy())
        if step % args.render_every == 0:
            scan_points = lidar.scan_points_world(robot.true_pose, world)
            frames.append(render_frame(world, robot.true_pose, robot.odom_pose, scan_points,
                                        np.array(true_trail), np.array(odom_trail)))

    n_steps = run_loop(robot, args.dt, args.max_steps, args.waypoint_tolerance, on_step)

    drift = float(np.hypot(*(robot.true_pose[:2] - robot.odom_pose[:2])))
    print(f"Finished after {n_steps} steps ({n_steps * args.dt:.1f}s simulated). "
          f"Final true pose: ({robot.true_pose[0]:.2f}, {robot.true_pose[1]:.2f}). "
          f"Odometry drift at finish: {drift:.2f} m.")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        save_gif(frames, args.record, fps=1.0 / (args.dt * args.render_every))
        print(f"Saved {len(frames)} frames to {args.record}")


if __name__ == "__main__":
    main()
