"""Stage 6: run the full SLAM stack (Stages 1-4's occupancy-grid mapping + ICP
scan-matching correction) on the REAL Unitree Go2 quadruped -- actual MuJoCo rigid-body
dynamics plus the classical convex-MPC balance/footstep controller from the sibling
`go1_rl_walk` project -- instead of this project's own simplified kinematic
unicycle/differential-drive stand-in (Stages 1-5). Every SLAM primitive
(`slam/occupancy_grid.py`, `slam/scan_match.py`) is reused completely UNMODIFIED: it only
ever consumed a generic (ranges, hit, angles) scan and a (x, y, theta) pose, which a real
legged robot can supply just as well as a toy unicycle.

What's genuinely different from Stages 1-5, not just re-skinned:
  - The lidar is a REAL MuJoCo ray-cast (`slam/mj_lidar.py`, `mujoco.mj_ray`) against
    actual 3D wall geometry, not a ray-vs-line-segment intersection against an abstract
    2D map.
  - "Odometry" is synthesized from the robot's own ACTUAL realized body-frame velocity
    (`info["v_body"]`/`info["w_body"]`, i.e. what leg odometry would estimate) corrupted
    by the same noise model `slam/robot.py` uses -- not a perfect unicycle integration,
    so real stance-phase/slip dynamics contribute to the drift this stage corrects, not
    just an injected random walk.
  - The world is a richer layout than Stages 1-5's single rectangle-with-an-island: the
    same outer-boundary + island loop corridor, PLUS a free-standing pillar obstacle
    placed well clear of the scripted route (so no new obstacle-avoidance/path-planning
    logic is needed -- the point of this stage is exercising real dynamics, not adding
    navigation capability this project has already deliberately scoped out).

Produces two GIFs: a third-person MuJoCo render of the actual quadruped walking the route
(media/stage6_go2_quadruped.gif), and the usual 2D occupancy-grid/trajectory view
(media/stage6_go2_slam_map.gif) built from its real sensor stream.

Usage (run with go1_rl_walk/mpc's own venv -- casadi/pinocchio/convex_mpc live there, not
in this project's requirements.txt):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 demo_stage6_go2_world.py \
        --record-map media/stage6_go2_slam_map.gif --record-robot media/stage6_go2_quadruped.gif
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco as mj
import numpy as np

GO1_RL_WALK = Path(__file__).resolve().parent.parent / "go1_rl_walk"
sys.path.insert(0, str(GO1_RL_WALK / "mpc"))
sys.path.insert(0, str(GO1_RL_WALK))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpc_residual_env import MPCResidualEnv, DEFAULT_XML, _fix_meshdir_absolute   # noqa: E402
from navigate_mpc import compute_nav_command_turn                                 # noqa: E402

from slam.world import World                                                      # noqa: E402
from slam.robot import DiffDriveRobot, wrap_to_pi                                  # noqa: E402
from slam.occupancy_grid import OccupancyGrid                                     # noqa: E402
from slam.scan_match import icp_2d, scan_to_local_points                         # noqa: E402
from slam.mj_lidar import mj_lidar_scan                                           # noqa: E402
from slam.viz import render_frame, save_gif                                       # noqa: E402
from slam.eval import map_accuracy                                                # noqa: E402

WAYPOINTS = np.array([
    [1.5, 1.5], [10.5, 1.5], [10.5, 6.5], [1.5, 6.5], [1.5, 1.5],
])
# A free-standing pillar, well clear of every WAYPOINTS segment above (verified: closest
# approach is >1.3m from any route leg) -- a second mapped feature beyond Stages 1-5's
# single island, with zero risk to the scripted route (no obstacle-avoidance needed).
PILLAR = np.array([
    [2.5, 4.5, 3.5, 4.5], [3.5, 4.5, 3.5, 5.5], [3.5, 5.5, 2.5, 5.5], [2.5, 5.5, 2.5, 4.5],
])


def build_world() -> World:
    base = World.loop_corridor()
    return World(np.concatenate([base.walls, PILLAR], axis=0), base.bounds)


def inject_walls(xml_text: str, walls: np.ndarray, height: float = 0.6,
                  thickness: float = 0.05, group: int = 4) -> str:
    """Append one static box geom per wall segment, just before `</worldbody>` -- same
    simple string-injection technique `mpc_residual_env.py`'s own `_inject_terrain_xml`
    uses, kept here (not imported) so this project adds zero code to the sibling repo."""
    geoms = []
    for i, (x1, y1, x2, y2) in enumerate(walls):
        length = float(np.hypot(x2 - x1, y2 - y1))
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        angle = float(np.arctan2(y2 - y1, x2 - x1))
        geoms.append(
            f'<geom name="wall_{i}" type="box" group="{group}" contype="1" conaffinity="1" '
            f'rgba="0.55 0.55 0.6 1" size="{length / 2:.4f} {thickness / 2:.4f} {height / 2:.4f}" '
            f'pos="{mx:.4f} {my:.4f} {height / 2:.4f}" euler="0 0 {angle:.6f}"/>')
    block = "\n    ".join(geoms)
    assert xml_text.count("</worldbody>") == 1, "expected exactly one </worldbody>"
    return xml_text.replace("</worldbody>", f"    {block}\n  </worldbody>")


def build_env_xml(walls: np.ndarray, tmp_dir: Path) -> Path:
    xml_text = Path(DEFAULT_XML).read_text()
    xml_text = _fix_meshdir_absolute(xml_text, Path(DEFAULT_XML).resolve().parent)
    xml_text = inject_walls(xml_text, walls)
    out_path = tmp_dir / "go2_world_slam.xml"
    out_path.write_text(xml_text)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=70.0)
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

    import tempfile
    rng = np.random.default_rng(args.seed)
    world = build_world()
    tmp_dir = Path(tempfile.mkdtemp(prefix="slam_sim_go2_"))
    xml_path = build_env_xml(world.walls, tmp_dir)

    env = MPCResidualEnv(
        xml_path=str(xml_path), episode_seconds=args.seconds,
        speed_range=(0.0, 0.0), lateral_range=(0.0, 0.0), yaw_rate_range=(0.0, 0.0),
        seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    env.mujoco_go2.data.qpos[0:2] = WAYPOINTS[0]
    mj.mj_forward(env.mujoco_go2.model, env.mujoco_go2.data)

    speed_lo, speed_hi = args.speed_range
    yaw_lo, yaw_hi = args.yaw_rate_range
    max_cmd_delta = args.max_accel / 200.0
    max_yaw_delta = args.max_yaw_accel / 200.0
    dt = 1.0 / 200.0

    grid = OccupancyGrid(world.bounds, resolution=0.1)
    belief_pose = np.array([WAYPOINTS[0][0], WAYPOINTS[0][1], 0.0])
    odom_pose = belief_pose.copy()
    true_trail, odom_trail, belief_trail = [], [], []
    n_lidar_scans, n_corrections = 0, 0
    map_frames, robot_frames = [], []

    robot_renderer = None
    robot_cam = None
    robot_scene_opt = None
    if args.record_robot:
        robot_cam = mj.MjvCamera()
        robot_cam.type = mj.mjtCamera.mjCAMERA_FREE
        robot_cam.lookat[:] = [6.0, 4.0, 0.0]
        robot_cam.distance = 16.0
        robot_cam.elevation = -60
        robot_cam.azimuth = 90
        robot_renderer = mj.Renderer(env.mujoco_go2.model, height=480, width=640)
        # Default MjvOption only enables a subset of geom groups -- our wall geoms are
        # group 4 (see inject_walls) specifically so the lidar's geomgroup ray-filter can
        # select them cleanly; without this they're invisible in the MuJoCo render too.
        robot_scene_opt = mj.MjvOption()
        robot_scene_opt.geomgroup[:] = 1

    cur_fwd, cur_yaw = 0.0, 0.0
    wp_idx = 1
    x, y, yaw = float(WAYPOINTS[0][0]), float(WAYPOINTS[0][1]), 0.0
    max_steps = int(args.seconds * 200)
    step = 0
    for step in range(1, max_steps + 1):
        goal = WAYPOINTS[wp_idx]
        dist = float(np.hypot(goal[0] - x, goal[1] - y))
        if dist <= args.waypoint_tolerance:
            wp_idx += 1
            if wp_idx >= len(WAYPOINTS):
                break
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
                    belief_trail=np.array(belief_trail) if len(belief_trail) > 1 else None))

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
          f"{n_lidar_scans} lidar scans, ICP corrected {n_corrections}/{n_lidar_scans}.")

    belief_drift = float(np.hypot(*(belief_pose[:2] - np.array([x, y]))))
    odom_drift = float(np.hypot(*(odom_pose[:2] - np.array([x, y]))))
    print(f"Final position drift: belief (SLAM-corrected) = {belief_drift:.3f}m, "
          f"raw odometry = {odom_drift:.3f}m.")

    wall_accuracy, free_accuracy, n_wall_in_bounds, n_wall_total = map_accuracy(grid, world)
    print(f"Wall-point accuracy (correctly marked occupied): {wall_accuracy:.3f} "
          f"({n_wall_in_bounds}/{n_wall_total} sample points in-bounds)")
    print(f"Free-point accuracy (correctly marked free): {free_accuracy:.3f} (4 sample points)")

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
