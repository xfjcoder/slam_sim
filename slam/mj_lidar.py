"""Stage 6: a REAL 2D lidar scan cast against an actual MuJoCo scene (mujoco's own
`mj_ray`), rather than the toy ray-vs-line-segment intersection `slam/lidar.py` uses
against the flat 2D `World`. Used when the "robot" is the real Go2 quadruped (MuJoCo +
classical MPC, from the sibling go1_rl_walk project) instead of this project's own
simplified kinematic model.

Returns the exact same (ranges, hit, angles) shape `slam/lidar.py` does, so every
downstream SLAM primitive (`slam/occupancy_grid.py`, `slam/scan_match.py`,
`slam/pose_graph.py`) is reusable completely unmodified.
"""
import numpy as np
import mujoco as mj


def mj_lidar_scan(model, data, pos2d, yaw: float, n_beams: int = 360, max_range: float = 8.0,
                   height: float = 0.30, wall_group: int = 4):
    """Cast `n_beams` horizontal rays from (pos2d[0], pos2d[1], height) around `yaw`,
    against ONLY geoms in MuJoCo geom group `wall_group` -- excludes the robot's own
    body/floor regardless of which MuJoCo body they're attached to, which `bodyexclude`
    alone could not do cleanly (the robot's legs are separate bodies from its trunk)."""
    angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
    beam_theta = yaw + angles
    pnt = np.array([pos2d[0], pos2d[1], height], dtype=np.float64)
    geomgroup = np.zeros(6, dtype=np.uint8)
    geomgroup[wall_group] = 1
    geomid = np.zeros(1, dtype=np.int32)

    ranges = np.full(n_beams, max_range, dtype=np.float64)
    hit = np.zeros(n_beams, dtype=bool)
    for i in range(n_beams):
        vec = np.array([np.cos(beam_theta[i]), np.sin(beam_theta[i]), 0.0], dtype=np.float64)
        dist = mj.mj_ray(model, data, pnt, vec, geomgroup, 1, -1, geomid)
        if 0.0 <= dist <= max_range:
            ranges[i] = dist
            hit[i] = True
    return ranges, hit, angles
