"""Stage 5 (stretch): a simplified 2D pose-graph optimizer. Where Stage 4 corrects the
belief pose continuously against the map (scan-to-map, every tick), this closes a
longer-range constraint instead: once the robot recognizes it has returned to an
already-visited place (loop closure), a single relative-pose measurement between two
far-apart points in the trajectory gets added as a graph EDGE alongside the chain of
odometry edges, and a batch least-squares solve redistributes the loop's accumulated
error across every pose in between -- not just the most recent one, which is exactly what
a purely sequential correction (Stage 4, or nothing at all) cannot do.
"""
import numpy as np
from scipy.optimize import least_squares


def wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def pose_to_matrix(pose) -> np.ndarray:
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def relative_pose(pose_a, pose_b) -> np.ndarray:
    """The pose of b as seen from a's own frame: T_a^-1 @ T_b -- what an odometry step or
    an ICP loop-closure match actually measures (a relative transform), not an absolute
    position neither sensor has access to."""
    Ta, Tb = pose_to_matrix(pose_a), pose_to_matrix(pose_b)
    return matrix_to_pose(np.linalg.inv(Ta) @ Tb)


def compose(pose_a, rel) -> np.ndarray:
    """a's pose composed with a relative transform expressed in a's own frame."""
    Ta, Tr = pose_to_matrix(pose_a), pose_to_matrix(rel)
    return matrix_to_pose(Ta @ Tr)


class PoseGraph:
    """Poses: a flat (N, 3) array of [x, y, theta] estimates, one per keyframe. Pose 0 is
    held FIXED during optimization -- there's no absolute ground truth available to a real
    robot, only "where it started", so fixing one pose anchors the solve's otherwise-free
    global rotation/translation (every SLAM back-end needs exactly this anchor)."""

    def __init__(self, initial_poses: np.ndarray):
        self.poses = np.asarray(initial_poses, dtype=np.float64).copy()
        self.edges = []   # each: (i, j, rel_pose (3,), weight (3,))

    def add_edge(self, i: int, j: int, rel_pose, weight=(1.0, 1.0, 1.0)) -> None:
        self.edges.append((i, j, np.asarray(rel_pose, dtype=np.float64),
                            np.asarray(weight, dtype=np.float64)))

    def _residuals(self, free_vars: np.ndarray) -> np.ndarray:
        poses = self.poses.copy()
        poses[1:] = free_vars.reshape(-1, 3)
        res = []
        for i, j, rel_meas, weight in self.edges:
            rel_pred = relative_pose(poses[i], poses[j])
            e = rel_pred - rel_meas
            e[2] = wrap_to_pi(e[2])
            res.append(e * weight)
        return np.concatenate(res)

    def optimize(self, max_nfev: int = 300) -> np.ndarray:
        """Batch least-squares over every free pose at once (Levenberg-Marquardt) --
        simple and fine at this problem size (hundreds of poses), though a real-time
        SLAM back-end would use a sparse solver exploiting the graph's structure."""
        free0 = self.poses[1:].reshape(-1)
        result = least_squares(self._residuals, free0, method="lm", max_nfev=max_nfev)
        self.poses[1:] = result.x.reshape(-1, 3)
        return self.poses
