"""A multi-ring (multi-elevation) extension of `slam.mj_lidar`'s single-plane 2D scan --
casts a ring of beams at each of several elevation angles from one sensor origin, against
the SAME real 3D MuJoCo wall geometry (`mujoco.mj_ray`, filtered to `wall_group` exactly
like `mj_lidar_scan`). Returns WORLD-frame 3D hit points directly (not a (ranges, hit,
angles) tuple) -- unlike the 2D scan, nothing downstream consumes these in sensor frame; the
only use is point-cloud accumulation for visualization.

Every wall in this project has a flat top (`slam/office.py`/Stage 6's walls are 0.6m-tall
boxes, no ceiling), so hits naturally cluster into two bands per wall -- the face and the
top edge -- the same thing a real multi-ring lidar (e.g. a Velodyne) would show pointed at a
wall from a few meters away.
"""
import numpy as np
import mujoco as mj


def mj_lidar_scan_3d(model, data, pos3d, yaw: float, n_beams: int = 90,
                      elevations_deg=(-20.0, -10.0, -3.0, 3.0, 10.0, 20.0, 30.0),
                      max_range: float = 8.0, wall_group: int = 4) -> np.ndarray:
    """pos3d: (x, y, z) sensor origin in world frame. Returns (K, 3) world-frame hit points,
    K <= n_beams * len(elevations_deg) (fewer if some beams miss)."""
    angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
    beam_theta = yaw + angles
    cos_t, sin_t = np.cos(beam_theta), np.sin(beam_theta)

    geomgroup = np.zeros(6, dtype=np.uint8)
    geomgroup[wall_group] = 1
    geomid = np.zeros(1, dtype=np.int32)

    origin = np.array(pos3d, dtype=np.float64)
    pts = []
    for elev_deg in elevations_deg:
        elev = np.radians(elev_deg)
        cos_e, sin_e = np.cos(elev), np.sin(elev)
        vx, vy, vz = cos_e * cos_t, cos_e * sin_t, np.full(n_beams, sin_e)
        for i in range(n_beams):
            vec = np.array([vx[i], vy[i], vz[i]], dtype=np.float64)
            dist = mj.mj_ray(model, data, origin, vec, geomgroup, 1, -1, geomid)
            if 0.0 <= dist <= max_range:
                pts.append(origin + dist * vec)
    return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3))
