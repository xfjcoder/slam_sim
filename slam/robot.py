"""A simplified kinematic (unicycle/differential-drive) robot -- no mass, no dynamics, no
falling. Deliberately chosen (per the project's own scoping decision) over the full rigid-body
quadruped sim this grew out of: for developing/validating a SLAM algorithm, locomotion noise
and balance failures are a confound, not a feature.

Maintains TWO separate pose estimates, the core idea every SLAM demo in this project builds
on:
  - `true_pose`: the ground-truth pose, advanced by the EXACT commanded velocity each step.
    Only ever used by the simulator itself (to place the robot for lidar ray-casting) and by
    evaluation/plotting code that needs to know "what actually happened" -- never fed to any
    mapping/localization algorithm, which would be cheating.
  - `odom_pose`: dead-reckoning estimate, advanced by the commanded velocity CORRUPTED by
    Gaussian noise each step, independent of true_pose. This is what a real robot would have
    (wheel encoders drift) -- it's meant to visibly diverge from true_pose over time, and
    that divergence is exactly the problem later SLAM stages exist to correct.
"""
import numpy as np


def wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


class DiffDriveRobot:
    def __init__(self, x0: float, y0: float, theta0: float,
                 odom_noise: tuple[float, float, float] = (0.02, 0.02, 0.01),
                 rng: np.random.Generator | None = None):
        """odom_noise: (sigma_v_frac, sigma_omega_frac, sigma_omega_from_v) -- standard
        odometry noise model (e.g. Thrun/Burgard/Fox): linear-velocity noise scales with
        |v|, angular-velocity noise scales with |omega| PLUS a cross term scaling with |v|
        (straight-line driving still accumulates a little heading drift, matching real
        wheel-encoder behavior, not just "noise when turning")."""
        self.true_pose = np.array([x0, y0, theta0], dtype=np.float64)
        self.odom_pose = self.true_pose.copy()
        self.odom_noise = odom_noise
        self._rng = rng if rng is not None else np.random.default_rng()

    def step(self, v: float, omega: float, dt: float) -> None:
        """Advance true_pose exactly, odom_pose with noise added to the SAME commanded
        (v, omega) -- both integrated with the same unicycle model, differing only in
        whether the input was corrupted first."""
        self.true_pose = self._integrate(self.true_pose, v, omega, dt)

        sv, so, sov = self.odom_noise
        v_noisy = v + self._rng.normal(0.0, sv * abs(v) + 1e-4)
        omega_noisy = omega + self._rng.normal(0.0, so * abs(omega) + sov * abs(v) + 1e-4)
        self.odom_pose = self._integrate(self.odom_pose, v_noisy, omega_noisy, dt)
        # Exposed so a caller tracking its own belief pose (Stage 4 scan-matching) can
        # apply the IDENTICAL noisy incremental motion as odom_pose got, rather than
        # jumping to the cumulative odom_pose itself (which would throw away any
        # correction already folded into the belief pose from a prior step).
        self.last_noisy_vw = (v_noisy, omega_noisy)

    @staticmethod
    def _integrate(pose, v, omega, dt):
        x, y, theta = pose
        x = x + v * np.cos(theta) * dt
        y = y + v * np.sin(theta) * dt
        theta = wrap_to_pi(theta + omega * dt)
        return np.array([x, y, theta], dtype=np.float64)
