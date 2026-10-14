"""Stage 5 (stretch): loop closure via pose-graph optimization.

Deliberately isolated from Stage 4's continuous scan-to-map correction -- this demo's
belief trajectory is raw, uncorrected ODOMETRY (dead reckoning only), so that open-loop
drift has a chance to actually accumulate before anything fixes it. That's the point: a
sequential corrector (Stage 4, or nothing at all) can only ever act on the most recent
pose. Loop closure instead recognizes a REVISIT -- using ICP to match the current scan
against the very first keyframe's scan once the odometry estimate drifts back near the
start -- and adds that single relative-pose measurement as a graph edge alongside the
whole chain of odometry edges. A batch least-squares solve (`slam/pose_graph.py`) then
redistributes the loop's accumulated error across every pose in between, not just the
last one.

Odometry noise is turned up from Stage 1-4's default (0.02, 0.02, 0.01) -- at that
default, one lap's open-loop drift is already only ~0.15m (see Stage 3/4's own numbers),
too small for loop closure's correction to be visually or numerically dramatic. This demo
isn't claiming real robots are this noisy; it's deliberately exercising the technique
somewhere it actually matters.

Usage:
    python3 demo_stage5_loop_closure.py --record media/stage5_loop_closure.gif
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slam.world import World
from slam.robot import DiffDriveRobot
from slam.lidar import Lidar2D
from slam.scan_match import icp_2d, scan_to_local_points, transform_points
from slam.pose_graph import PoseGraph, relative_pose
from slam.viz import render_frame, save_gif
from slam.trajectory import WAYPOINTS, run_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--waypoint-tolerance", type=float, default=0.3)
    ap.add_argument("--odom-noise", type=float, nargs=3, default=[0.08, 0.08, 0.04],
                     help="(sigma_v_frac, sigma_omega_frac, sigma_omega_from_v) -- "
                          "stronger than Stage 1-4's default so open-loop drift is large "
                          "enough, by the time the loop closes, for the correction to be "
                          "dramatically visible.")
    ap.add_argument("--closure-radius", type=float, default=1.5,
                     help="Trigger a loop-closure attempt once the ODOMETRY estimate (the "
                          "only thing the robot has) comes back within this many meters of "
                          "the start keyframe's own odometry pose.")
    ap.add_argument("--min-closure-step", type=int, default=100)
    ap.add_argument("--max-correspondence-dist", type=float, default=0.5)
    ap.add_argument("--min-correspondences", type=int, default=20)
    ap.add_argument("--odom-edge-weight", type=float, default=1.0)
    ap.add_argument("--loop-edge-weight", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--render-every", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    world = World.loop_corridor()
    robot = DiffDriveRobot(*WAYPOINTS[0], 0.0, odom_noise=tuple(args.odom_noise), rng=rng)
    lidar = Lidar2D(n_beams=360, max_range=8.0, rng=rng)

    ranges0, hit0 = lidar.scan(robot.true_pose, world)
    local_pts0 = scan_to_local_points(ranges0, hit0, lidar.angles)

    true_poses = [robot.true_pose.copy()]
    odom_poses = [robot.odom_pose.copy()]
    loop_closure_edge = None
    frames = []

    def on_step(step, robot):
        nonlocal loop_closure_edge
        ranges, hit = lidar.scan(robot.true_pose, world)   # sensor always reads the TRUE pose
        true_poses.append(robot.true_pose.copy())
        odom_poses.append(robot.odom_pose.copy())
        idx = len(true_poses) - 1

        if loop_closure_edge is None and idx >= args.min_closure_step:
            d = float(np.hypot(*(robot.odom_pose[:2] - odom_poses[0][:2])))
            if d < args.closure_radius:
                local_pts = scan_to_local_points(ranges, hit, lidar.angles)
                map_pts0 = transform_points(local_pts0, odom_poses[0])
                corrected, converged, n = icp_2d(
                    local_pts, map_pts0, robot.odom_pose,
                    max_correspondence_dist=args.max_correspondence_dist,
                    min_correspondences=args.min_correspondences)
                if converged:
                    rel = relative_pose(odom_poses[0], corrected)
                    loop_closure_edge = (0, idx, rel)

        if args.record and step % args.render_every == 0:
            frames.append(render_frame(
                world, robot.true_pose,
                true_trail=np.array(true_poses)[:, :2],
                odom_trail=np.array(odom_poses)[:, :2]))

    n_steps = run_loop(robot, args.dt, args.max_steps, args.waypoint_tolerance, on_step)
    true_poses = np.array(true_poses)
    odom_poses = np.array(odom_poses)

    graph = PoseGraph(odom_poses)
    for k in range(len(odom_poses) - 1):
        rel = relative_pose(odom_poses[k], odom_poses[k + 1])
        graph.add_edge(k, k + 1, rel, weight=(args.odom_edge_weight,) * 3)

    if loop_closure_edge is not None:
        i, j, rel = loop_closure_edge
        graph.add_edge(i, j, rel, weight=(args.loop_edge_weight,) * 3)
        print(f"Loop closure detected at step {j}/{n_steps} (odometry distance to start "
              f"{np.hypot(*(odom_poses[j][:2] - odom_poses[0][:2])):.3f}m); added as a "
              f"graph edge.")
    else:
        print("No loop closure detected within this run -- nothing to optimize against "
              "but the odometry chain itself.")

    optimized_poses = graph.optimize().copy()

    def rmse(a, b):
        return float(np.sqrt(np.mean(np.sum((a[:, :2] - b[:, :2]) ** 2, axis=1))))

    odom_rmse = rmse(odom_poses, true_poses)
    optimized_rmse = rmse(optimized_poses, true_poses)
    odom_final_drift = float(np.hypot(*(odom_poses[-1][:2] - true_poses[-1][:2])))
    optimized_final_drift = float(np.hypot(*(optimized_poses[-1][:2] - true_poses[-1][:2])))

    print(f"Finished after {n_steps} steps ({n_steps * args.dt:.1f}s simulated).")
    print(f"Trajectory RMSE vs. ground truth: odometry-only = {odom_rmse:.3f}m, "
          f"pose-graph-optimized = {optimized_rmse:.3f}m.")
    print(f"Final position drift: odometry-only = {odom_final_drift:.3f}m, "
          f"pose-graph-optimized = {optimized_final_drift:.3f}m.")

    if args.record:
        # A short "before -> after" snap: interpolate the trail from the raw odometry
        # trajectory to the pose-graph-optimized one over a few frames, so the correction
        # reads as an actual event, not just a changed number at the end.
        n_snap = 15
        for t in np.linspace(0.0, 1.0, n_snap):
            blended = odom_poses[:, :2] * (1 - t) + optimized_poses[:, :2] * t
            frames.append(render_frame(
                world, true_poses[-1],
                true_trail=true_poses[:, :2], odom_trail=odom_poses[:, :2],
                belief_trail=blended))
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        save_gif(frames, args.record, fps=1.0 / (args.dt * args.render_every))
        print(f"Saved {len(frames)} frames to {args.record}")


if __name__ == "__main__":
    main()
