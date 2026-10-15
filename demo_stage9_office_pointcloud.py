"""Stage 9a: a 3D lidar point-cloud map of the office -- every stage so far only ever built
a flat 2D occupancy grid (even Stage 6/8's "real" lidar only ever cast a single horizontal
plane of rays, `slam/mj_lidar.py`). This casts a multi-ring sensor (`slam/mj_lidar_3d.py`,
several elevation angles per heading) against the real 3D MuJoCo wall geometry instead, so
the resulting point cloud actually shows wall height (face vs. top edge), not just a
footprint.

Deliberately the FAST way to get a full-building point cloud: no quadruped physics at all,
just Stage 7's own kinematic trajectory (`DiffDriveRobot` + `waypoint_command` +
`slam.office.build_mission`) walked through a STATIC MuJoCo scene (one `mj_forward`, no
stepping -- same trick Stage 7b's flythrough uses) with the 3D sensor sampled at intervals
along the way. See `demo_stage8_office_go2.py --record-pointcloud3d` for the heavier,
"from the real Go2's actual run" version of the same idea.

Usage (needs go1_rl_walk/mpc's own venv for mujoco):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 demo_stage9_office_pointcloud.py \
        --record media/stage9_office_pointcloud.gif \
        --record-still media/stage9_office_pointcloud_views.png
"""
import argparse
import sys
import tempfile
from pathlib import Path

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_stage6_go2_world import build_env_xml                            # noqa: E402
from slam.robot import DiffDriveRobot                                      # noqa: E402
from slam.trajectory import waypoint_command                               # noqa: E402
from slam.mj_lidar_3d import mj_lidar_scan_3d                              # noqa: E402
from slam.viz import render_pointcloud_3d, render_pointcloud_3d_panel, save_gif, save_image  # noqa: E402
from slam.office import build_world, build_mission, JOBS                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=str, nargs="+", default=list(JOBS.keys()),
                     help=f"Ordered list of job names to tour. Choices: {sorted(JOBS)}.")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--waypoint-tolerance", type=float, default=0.3)
    ap.add_argument("--sensor-height", type=float, default=0.30)
    ap.add_argument("--scan-every", type=int, default=6, help="ticks between 3D ring scans")
    ap.add_argument("--n-beams", type=int, default=90)
    ap.add_argument("--max-range", type=float, default=8.0)
    ap.add_argument("--n-frames", type=int, default=48)
    ap.add_argument("--elevation", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", type=str, default=None,
                     help="rotating-azimuth GIF output path")
    ap.add_argument("--record-still", type=str, default=None,
                     help="static multi-angle panel PNG output path (see "
                          "slam/viz.py's render_pointcloud_3d_panel)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    world = build_world()
    waypoints, job_at_index = build_mission(args.jobs)

    tmp_dir = Path(tempfile.mkdtemp(prefix="slam_sim_office_pc_"))
    xml_path = build_env_xml(world.walls, tmp_dir)
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)   # resolves static wall geom poses; nothing is stepped

    robot = DiffDriveRobot(*waypoints[0], 0.0, rng=rng)

    wp_idx = 1
    dwell_remaining = 0
    step = 0
    n_scans = 0
    all_points = []
    while step < args.max_steps:
        if dwell_remaining == 0 and wp_idx >= len(waypoints):
            break
        if dwell_remaining > 0:
            dwell_remaining -= 1
        else:
            goal = waypoints[wp_idx]
            v, omega, dist = waypoint_command(robot.true_pose, goal)
            if dist <= args.waypoint_tolerance:
                if wp_idx in job_at_index:
                    _, _, dwell = job_at_index[wp_idx]
                    dwell_remaining = dwell
                wp_idx += 1
                continue
            robot.step(v, omega, args.dt)

        if step % args.scan_every == 0:
            x, y, theta = robot.true_pose
            pts = mj_lidar_scan_3d(
                model, data, (x, y, args.sensor_height), theta,
                n_beams=args.n_beams, max_range=args.max_range)
            if len(pts):
                all_points.append(pts)
            n_scans += 1
        step += 1

    points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3))
    print(f"Finished after {step} ticks ({step * args.dt:.1f}s of trajectory), "
          f"visiting {len(args.jobs)} job(s): {', '.join(args.jobs)}.")
    print(f"{n_scans} 3D ring scans, {len(points)} total hit points accumulated.")

    if args.record:
        frames = render_pointcloud_3d(points, world.bounds, n_frames=args.n_frames,
                                       elevation=args.elevation)
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        save_gif(frames, args.record, fps=12.0)
        print(f"Saved {len(frames)} frames to {args.record}")

    if args.record_still:
        panel = render_pointcloud_3d_panel(points, world.bounds)
        Path(args.record_still).parent.mkdir(parents=True, exist_ok=True)
        save_image(panel, args.record_still)
        print(f"Saved a static multi-angle panel to {args.record_still}")


if __name__ == "__main__":
    main()
