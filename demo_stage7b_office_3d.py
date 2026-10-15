"""Stage 7b: a static 3D view of the office floorplan -- no SLAM, no physics stepping, just
the same real MuJoCo wall geometry Stage 6/7's real-Go2 pipeline senses against, rendered
from a slowly orbiting camera so the layout (kitchen, two conference rooms, the cubicle
area, the fire-extinguisher cabinet, all the doorway gaps) can actually be looked at in 3D
before spending the time on a full physics-driven run.

Reuses Stage 6's own `inject_walls`/`build_env_xml` unmodified (both already generic over
any wall array, not specific to Stage 6's own layout) against `slam/office.py`'s floorplan
instead of the loop-corridor one. The Go2 quadruped is placed at the office entrance purely
for scale/visual reference -- no controller runs, `mj.mj_forward` just resolves the one
static pose for rendering.

Usage (needs go1_rl_walk/mpc's own venv for the `mujoco` package):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 demo_stage7b_office_3d.py \
        --record media/stage7b_office_3d_flythrough.gif
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

from demo_stage6_go2_world import build_env_xml                            # noqa: E402
from slam.office import build_world, START                                 # noqa: E402
from slam.viz import save_gif                                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=72)
    ap.add_argument("--distance", type=float, default=17.0)
    ap.add_argument("--elevation", type=float, default=-50.0)
    ap.add_argument("--azimuth-start", type=float, default=0.0)
    ap.add_argument("--azimuth-sweep", type=float, default=360.0)
    ap.add_argument("--record", type=str, default=None)
    args = ap.parse_args()

    world = build_world()
    tmp_dir = Path(tempfile.mkdtemp(prefix="slam_sim_office3d_"))
    xml_path = build_env_xml(world.walls, tmp_dir)

    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    data.qpos[0:2] = START
    mj.mj_forward(model, data)

    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [7.0, 5.0, 0.0]
    cam.distance = args.distance
    cam.elevation = args.elevation

    scene_opt = mj.MjvOption()
    scene_opt.geomgroup[:] = 1   # group 4 (injected walls) is off by default -- see Stage 6

    renderer = mj.Renderer(model, height=540, width=720)
    frames = []
    azimuths = np.linspace(args.azimuth_start, args.azimuth_start + args.azimuth_sweep,
                            args.n_frames, endpoint=False)
    for az in azimuths:
        cam.azimuth = float(az)
        renderer.update_scene(data, camera=cam, scene_option=scene_opt)
        frames.append(renderer.render().copy())
    renderer.close()

    print(f"Rendered {len(frames)} frames of the office floorplan "
          f"(distance={args.distance}, elevation={args.elevation}).")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        save_gif(frames, args.record, fps=12.0)
        print(f"Saved {len(frames)} frames to {args.record}")


if __name__ == "__main__":
    main()
