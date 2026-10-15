"""Stage 7: a multi-room office floorplan (kitchen, two conference rooms, a cubicle area,
a fire-extinguisher check spot) plus a command layer on top of the existing SLAM stack --
give the robot an ordered list of job names and it drives to each one in turn, dwells there
to "perform" a short scripted task, then moves to the next. Every SLAM primitive from
Stages 1-6 (occupancy-grid mapping, ICP scan-to-map correction) is reused completely
unmodified against this richer ground-truth layout; what's new is `slam/office.py` (the
floorplan + named-location/job registry) and this mission-stepping loop.

While dwelling at a job the robot is genuinely stationary (no commanded velocity), but
lidar scanning, ICP correction, and map updates keep running every tick -- same discipline
as every driving tick, just with zero odometry prediction to correct against (nothing new
is pulled into the belief pose from a step that didn't happen).

Usage:
    python3 demo_stage7_office.py --jobs kitchen conference_a fire_extinguisher \
        --record media/stage7_office_tour.gif
    python3 demo_stage7_office.py --jobs kitchen conference_a conference_b cubicles \
        fire_extinguisher --record media/stage7_office_tour.gif
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slam.robot import DiffDriveRobot
from slam.lidar import Lidar2D
from slam.occupancy_grid import OccupancyGrid
from slam.scan_match import icp_2d, scan_to_local_points
from slam.viz import render_frame, save_gif
from slam.trajectory import waypoint_command
from slam.eval import map_accuracy
from slam.office import build_world, build_mission, ROOM_LABELS, JOBS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=str, nargs="+", default=list(JOBS.keys()),
                     help=f"Ordered list of job names to visit. Choices: {sorted(JOBS)}.")
    ap.add_argument("--resolution", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--waypoint-tolerance", type=float, default=0.3)
    ap.add_argument("--occ-threshold", type=float, default=0.7)
    ap.add_argument("--max-correspondence-dist", type=float, default=0.5)
    ap.add_argument("--min-correspondences", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--render-every", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    world = build_world()
    waypoints, job_at_index = build_mission(args.jobs)

    robot = DiffDriveRobot(*waypoints[0], 0.0, rng=rng)
    lidar = Lidar2D(n_beams=360, max_range=8.0, rng=rng)
    grid = OccupancyGrid(world.bounds, resolution=args.resolution)

    belief_pose = robot.true_pose.copy()
    true_trail, odom_trail, belief_trail = [], [], []
    n_corrections, n_steps_taken = 0, 0
    frames = []
    task_log = []

    wp_idx = 1
    dwell_remaining = 0
    status_text = None
    step = 0
    while step < args.max_steps:
        if dwell_remaining == 0 and wp_idx >= len(waypoints):
            break   # mission complete: every job visited, back at START

        did_step = False
        if dwell_remaining > 0:
            dwell_remaining -= 1
            if dwell_remaining == 0:
                task_log.append("  ...done.")
                status_text = None
        else:
            goal = waypoints[wp_idx]
            v, omega, dist = waypoint_command(robot.true_pose, goal)
            if dist <= args.waypoint_tolerance:
                if wp_idx in job_at_index:
                    name, task, dwell = job_at_index[wp_idx]
                    dwell_remaining = dwell
                    status_text = f"{task}..."
                    task_log.append(f"Reached {name} -- {task}...")
                wp_idx += 1
                continue   # re-evaluate next loop iteration without burning a sim step
            robot.step(v, omega, args.dt)
            did_step = True

        # Sense + SLAM-correct every tick, driving or dwelling -- a dwell tick contributes
        # zero odometry prediction (the robot really isn't moving), so belief_pose only
        # ever refines in place against the current scan, never drifts from a phantom step.
        v_noisy, omega_noisy = robot.last_noisy_vw if did_step else (0.0, 0.0)
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
        n_steps_taken += 1

        if args.record and step % args.render_every == 0:
            frames.append(render_frame(
                world, robot.true_pose, occupancy_grid=grid,
                true_trail=np.array(true_trail) if len(true_trail) > 1 else None,
                odom_trail=np.array(odom_trail) if len(odom_trail) > 1 else None,
                belief_trail=np.array(belief_trail) if len(belief_trail) > 1 else None,
                labels=ROOM_LABELS, status_text=status_text))
        step += 1

    print(f"Finished after {n_steps_taken} ticks ({n_steps_taken * args.dt:.1f}s simulated), "
          f"visiting {len(args.jobs)} job(s): {', '.join(args.jobs)}.")
    for line in task_log:
        print(line)
    print(f"ICP corrected {n_corrections}/{n_steps_taken} ticks "
          f"({100.0 * n_corrections / n_steps_taken:.0f}%).")

    belief_drift = float(np.hypot(*(belief_pose[:2] - robot.true_pose[:2])))
    odom_drift = float(np.hypot(*(robot.odom_pose[:2] - robot.true_pose[:2])))
    print(f"Final position drift: belief (SLAM-corrected) = {belief_drift:.3f}m, "
          f"raw odometry = {odom_drift:.3f}m.")

    free_points = np.unique(waypoints, axis=0)   # every mission waypoint is known free space
    wall_accuracy, free_accuracy, n_wall_in_bounds, n_wall_total = map_accuracy(
        grid, world, free_points=free_points, wall_both_sides=True)
    print(f"Wall-point accuracy (correctly marked occupied): {wall_accuracy:.3f} "
          f"({n_wall_in_bounds}/{n_wall_total} sample points in-bounds)")
    print(f"Free-point accuracy (correctly marked free): {free_accuracy:.3f} "
          f"({len(free_points)} sample points)")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        save_gif(frames, args.record, fps=1.0 / (args.dt * args.render_every))
        print(f"Saved {len(frames)} frames to {args.record}")


if __name__ == "__main__":
    main()
