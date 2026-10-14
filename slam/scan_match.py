"""Stage 4: point-to-point ICP scan matching -- correct a predicted pose (odometry's best
guess) by aligning the CURRENT lidar scan against the occupied cells of the map built so
far. This is "scan-to-map" matching, the same style real SLAM front-ends use (Hector SLAM,
Cartographer's local matcher): matching each new scan against the accumulated map, not just
against the previous scan, is what lets the correction cancel drift instead of merely
smoothing it (scan-to-scan matching still accumulates error step by step, since each match
only ever disagrees with one prior frame, which may itself already be off).

Closes the basic SLAM loop with Stage 2/3's occupancy grid: predict (odometry) -> correct
(this module, against the map) -> update the map with the corrected pose -> repeat.
"""
import numpy as np
from scipy.spatial import cKDTree


def best_rigid_transform(src: np.ndarray, dst: np.ndarray):
    """Kabsch algorithm: the rotation R and translation t minimizing sum ||R@src_i + t -
    dst_i||^2, given src/dst already in 1:1 correspondence (both (N, 2))."""
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    src0 = src - src_c
    dst0 = dst - dst_c
    H = src0.T @ dst0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        # reflection, not a rotation -- flip the smallest-singular-value axis (the
        # standard Kabsch fix) to get back a proper rotation.
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_c - R @ src_c
    return R, t


def icp_2d(scan_points: np.ndarray, map_points: np.ndarray, init_pose,
           max_iters: int = 20, tol: float = 1e-5,
           max_correspondence_dist: float = 0.5, min_correspondences: int = 20):
    """Align `scan_points` (N, 2 -- ROBOT-FRAME hit points, i.e. before any pose is
    applied) to `map_points` (M, 2 -- world-frame occupied cell centers), starting the
    search from `init_pose` (x, y, theta).

    Returns (corrected_pose, converged, n_correspondences). `converged=False` (with
    corrected_pose == init_pose, unchanged) whenever there isn't enough map yet to match
    against -- callers can then just fall back to the raw odometry prediction without
    any special-casing of their own, exactly what happens early in a lap before enough
    wall has been mapped to correct against.
    """
    init_pose = np.asarray(init_pose, dtype=np.float64)
    if len(map_points) < min_correspondences or len(scan_points) < min_correspondences:
        return init_pose, False, 0

    tree = cKDTree(map_points)
    x, y, theta = init_pose
    R_cur = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t_cur = np.array([x, y])

    n_valid = 0
    prev_error = None
    for _ in range(max_iters):
        pts = scan_points @ R_cur.T + t_cur
        dists, idx = tree.query(pts)
        valid = dists < max_correspondence_dist
        n_valid = int(valid.sum())
        if n_valid < min_correspondences:
            return init_pose, False, n_valid

        R_cur, t_cur = best_rigid_transform(scan_points[valid], map_points[idx[valid]])
        error = float(dists[valid].mean())
        if prev_error is not None and abs(prev_error - error) < tol:
            break
        prev_error = error

    theta_corrected = float(np.arctan2(R_cur[1, 0], R_cur[0, 0]))
    corrected_pose = np.array([t_cur[0], t_cur[1], theta_corrected])
    return corrected_pose, True, n_valid


def scan_to_local_points(ranges: np.ndarray, hit: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Lidar (ranges, hit, angles) -> (N, 2) hit points in the ROBOT's own local frame
    (angle measured from the robot's heading, not world-frame) -- what icp_2d expects as
    `scan_points`, since it applies the full candidate pose (rotation + translation)
    itself during alignment."""
    local_x = ranges[hit] * np.cos(angles[hit])
    local_y = ranges[hit] * np.sin(angles[hit])
    return np.stack([local_x, local_y], axis=1)
