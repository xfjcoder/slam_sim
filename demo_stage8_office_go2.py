"""Stage 8: Stage 7's office floorplan + named-location job commands, run on the REAL
Unitree Go2 quadruped from the sibling `go1_rl_walk` project (real MuJoCo rigid-body
dynamics + classical convex-MPC controller) -- the same combination Stage 6 did for the
plain loop-corridor world, now pointed at the office instead. Every piece is reused
completely unmodified from its own stage: `slam/office.py`'s world/mission (Stage 7),
`slam/mj_lidar.py`'s real ray-cast lidar and `demo_stage6_go2_world.py`'s wall-injection
XML plumbing (Stage 6), and the occupancy-grid + ICP pipeline (Stage 2/4). What's new here
is just the mission-stepping loop itself, driving the real robot instead of the kinematic
one, and converting each job's dwell time (specified in `slam/office.py` as Stage 7's own
0.1s "ticks") into real physics steps.

While dwelling the commanded velocity is zero, but `env.step()` keeps running every
physics tick -- the real balance/footstep controller has no "pause" button, so holding
still costs just as much compute as driving.

Usage (needs go1_rl_walk/mpc's own venv for mujoco/casadi/pinocchio):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 demo_stage8_office_go2.py \
        --jobs kitchen conference_a conference_b cubicles fire_extinguisher \
        --record-map media/stage8_office_go2_slam_map.gif \
        --record-robot media/stage8_office_go2_quadruped.gif
"""
import argparse
import sys
import tempfile
from pathlib import Path

import mujoco as mj
import numpy as np

GO1_RL_WALK = Path(__file__).resolve().parent.parent / "go1_rl_walk"
sys.path.insert(0, str(GO1_RL_WALK / "mpc"))
sys.path.insert(0, str(GO1_RL_WALK))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpc_residual_env import MPCResidualEnv                                # noqa: E402
from navigate_mpc import compute_nav_command_turn                          # noqa: E402

from demo_stage6_go2_world import build_env_xml                            # noqa: E402
from slam.robot import DiffDriveRobot                                      # noqa: E402
from slam.occupancy_grid import OccupancyGrid                              # noqa: E402
from slam.scan_match import icp_2d, scan_to_local_points                   # noqa: E402
from slam.mj_lidar import mj_lidar_scan                                    # noqa: E402
from slam.viz import render_frame, save_gif                                # noqa: E402
from slam.eval import map_accuracy                                         # noqa: E402
from slam.office import build_world, build_mission, ROOM_LABELS, JOBS      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=str, nargs="+", default=list(JOBS.keys()),
                     help=f"Ordered list of job names to visit. Choices: {sorted(JOBS)}.")
    ap.add_argument("--seconds", type=float, default=220.0)
    ap.add_argument("--waypoint-tolerance", type=float, default=0.3)
    ap.add_argument("--kp-fwd", type=float, default=1.0)
    ap.add_argument("--kp-yaw", type=float, default=2.0)
    ap.add_argument("--speed-range", type=float, nargs=2, default=[0.0, 0.7])
    ap.add_argument("--yaw-rate-range", type=float, nargs=2, default=[-1.5, 1.5])
    ap.add_argument("--max-accel", type=float, default=1.5)
    ap.add_argument("--max-yaw-accel", type=float, default=3.0)
    ap.add_argument("--odom-noise", type=float, nargs=3, default=[0.03, 0.03, 0.015])
    ap.add_argument("--lidar-beams", type=int, default=180)
    ap.add_argument("--lidar-every", type=int, default=10, help="control steps between lidar scans")
    ap.add_argument("--map-render-every", type=int, default=2, help="lidar scans between map-GIF frames")
    ap.add_argument("--robot-render-every", type=int, default=16, help="control steps between quadruped-GIF frames")
    ap.add_argument("--occ-threshold", type=float, default=0.7)
    ap.add_argument("--max-correspondence-dist", type=float, default=0.5)
    ap.add_argument("--min-correspondences", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record-map", type=str, default=None)
    ap.add_argument("--record-robot", type=str, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    world = build_world()
    waypoints, job_at_index = build_mission(args.jobs)

    tmp_dir = Path(tempfile.mkdtemp(prefix="slam_sim_office_go2_"))
    xml_path = build_env_xml(world.walls, tmp_dir)

    dt = 1.0 / 200.0

    env = MPCResidualEnv(
        xml_path=str(xml_path), episode_seconds=args.seconds,
        speed_range=(0.0, 0.0), lateral_range=(0.0, 0.0), yaw_rate_range=(0.0, 0.0),
        seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    env.mujoco_go2.data.qpos[0:2] = waypoints[0]
    mj.mj_forward(env.mujoco_go2.model, env.mujoco_go2.data)

    speed_lo, speed_hi = args.speed_range
    yaw_lo, yaw_hi = args.yaw_rate_range
    max_cmd_delta = args.max_accel * dt
    max_yaw_delta = args.max_yaw_accel * dt

    grid = OccupancyGrid(world.bounds, resolution=0.1)
    belief_pose = np.array([waypoints[0][0], waypoints[0][1], 0.0])
    odom_pose = belief_pose.copy()
    true_trail, odom_trail, belief_trail = [], [], []
    n_lidar_scans, n_corrections = 0, 0
    map_frames, robot_frames = [], []
    task_log = []

    robot_renderer = None
    robot_cam = None
    robot_scene_opt = None
    if args.record_robot:
        robot_cam = mj.MjvCamera()
        robot_cam.type = mj.mjtCamera.mjCAMERA_FREE
        robot_cam.lookat[:] = [7.0, 5.0, 0.0]
        robot_cam.distance = 17.0
        robot_cam.elevation = -55
        robot_cam.azimuth = 90
        robot_renderer = mj.Renderer(env.mujoco_go2.model, height=480, width=640)
        robot_scene_opt = mj.MjvOption()
        robot_scene_opt.geomgroup[:] = 1

    cur_fwd, cur_yaw = 0.0, 0.0
    wp_idx = 1
    dwell_steps_remaining = 0
    status_text = None
    x, y, yaw = float(waypoints[0][0]), float(waypoints[0][1]), 0.0
    max_steps = int(args.seconds * 200)
    step = 0
    for step in range(1, max_steps + 1):
        if dwell_steps_remaining == 0 and wp_idx >= len(waypoints):
            break   # mission complete: every job visited, back at the entrance

        if dwell_steps_remaining > 0:
            target_fwd, target_yaw = 0.0, 0.0
            dwell_steps_remaining -= 1
            if dwell_steps_remaining == 0:
                task_log.append("  ...done.")
                status_text = None
        else:
            goal = waypoints[wp_idx]
            dist = float(np.hypot(goal[0] - x, goal[1] - y))
            if dist <= args.waypoint_tolerance:
                if wp_idx in job_at_index:
                    name, task, dwell_ticks = job_at_index[wp_idx]
                    # job_at_index's dwell is in Stage 7's own 0.1s "ticks" -- convert to
                    # real physics steps at this script's own dt.
                    dwell_steps_remaining = int(round(dwell_ticks * 0.1 / dt))
                    status_text = f"{task}..."
                    task_log.append(f"Reached {name} -- {task}...")
                wp_idx += 1
                continue
            target_fwd, target_yaw = compute_nav_command_turn(
                x, y, yaw, goal[0], goal[1], args.kp_fwd, args.kp_yaw,
                speed_lo, speed_hi, yaw_lo, yaw_hi)

        cur_fwd += np.clip(target_fwd - cur_fwd, -max_cmd_delta, max_cmd_delta)
        cur_yaw += np.clip(target_yaw - cur_yaw, -max_yaw_delta, max_yaw_delta)
        env.set_command(cur_fwd, 0.0, cur_yaw)

        obs, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        x, y = float(info["base_pos"][0]), float(info["base_pos"][1])
        yaw = float(info["yaw"])
        true_trail.append([x, y])

        v_body, w_body = info["v_body"], info["w_body"]
        sv, so, sov = args.odom_noise
        v_noisy = float(v_body[0]) + rng.normal(0.0, sv * abs(v_body[0]) + 1e-4)
        w_noisy = float(w_body[2]) + rng.normal(0.0, so * abs(w_body[2]) + sov * abs(v_body[0]) + 1e-4)
        odom_pose = DiffDriveRobot._integrate(odom_pose, v_noisy, w_noisy, dt)
        belief_pose = DiffDriveRobot._integrate(belief_pose, v_noisy, w_noisy, dt)
        odom_trail.append(odom_pose[:2].copy())

        if terminated:
            print(f"Terminated early at step {step} (fell or QP failed).")
            break

        if step % args.lidar_every == 0:
            ranges, hit, angles = mj_lidar_scan(
                env.mujoco_go2.model, env.mujoco_go2.data, (x, y), yaw,
                n_beams=args.lidar_beams)
            local_pts = scan_to_local_points(ranges, hit, angles)
            map_pts = grid.occupied_points(threshold=args.occ_threshold)
            corrected, converged, _ = icp_2d(
                local_pts, map_pts, belief_pose,
                max_correspondence_dist=args.max_correspondence_dist,
                min_correspondences=args.min_correspondences)
            belief_pose = corrected if converged else belief_pose
            if converged:
                n_corrections += 1
            grid.update(belief_pose, ranges, hit, angles)
            n_lidar_scans += 1
            belief_trail.append(belief_pose[:2].copy())

            if args.record_map and n_lidar_scans % args.map_render_every == 0:
                map_frames.append(render_frame(
                    world, (x, y, yaw), occupancy_grid=grid,
                    true_trail=np.array(true_trail) if len(true_trail) > 1 else None,
                    odom_trail=np.array(odom_trail) if len(odom_trail) > 1 else None,
                    belief_trail=np.array(belief_trail) if len(belief_trail) > 1 else None,
                    labels=ROOM_LABELS, status_text=status_text))

        if args.record_robot and step % args.robot_render_every == 0:
            robot_renderer.update_scene(env.mujoco_go2.data, camera=robot_cam,
                                        scene_option=robot_scene_opt)
            robot_frames.append(robot_renderer.render().copy())

        if truncated:
            print(f"Truncated (ran out of episode time) at step {step}.")
            break

    n_steps = step
    true_trail = np.array(true_trail)
    odom_trail = np.array(odom_trail)
    print(f"Finished after {n_steps} steps ({n_steps * dt:.1f}s simulated), "
          f"visiting {len(args.jobs)} job(s): {', '.join(args.jobs)}.")
    for line in task_log:
        print(line)
    print(f"{n_lidar_scans} lidar scans, ICP corrected {n_corrections}/{n_lidar_scans}.")

    belief_drift = float(np.hypot(*(belief_pose[:2] - np.array([x, y]))))
    odom_drift = float(np.hypot(*(odom_pose[:2] - np.array([x, y]))))
    print(f"Final position drift: belief (SLAM-corrected) = {belief_drift:.3f}m, "
          f"raw odometry = {odom_drift:.3f}m.")

    free_points = np.unique(waypoints, axis=0)
    wall_accuracy, free_accuracy, n_wall_in_bounds, n_wall_total = map_accuracy(
        grid, world, free_points=free_points, wall_both_sides=True)
    print(f"Wall-point accuracy (correctly marked occupied): {wall_accuracy:.3f} "
          f"({n_wall_in_bounds}/{n_wall_total} sample points in-bounds)")
    print(f"Free-point accuracy (correctly marked free): {free_accuracy:.3f} "
          f"({len(free_points)} sample points)")

    if robot_renderer is not None:
        robot_renderer.close()
    if args.record_map and map_frames:
        Path(args.record_map).parent.mkdir(parents=True, exist_ok=True)
        save_gif(map_frames, args.record_map,
                  fps=1.0 / (dt * args.lidar_every * args.map_render_every))
        print(f"Saved {len(map_frames)} frames to {args.record_map}")
    if args.record_robot and robot_frames:
        Path(args.record_robot).parent.mkdir(parents=True, exist_ok=True)
        save_gif(robot_frames, args.record_robot, fps=1.0 / (dt * args.robot_render_every))
        print(f"Saved {len(robot_frames)} frames to {args.record_robot}")


if __name__ == "__main__":
    main()
