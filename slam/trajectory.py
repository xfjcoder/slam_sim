"""Shared scripted trajectory: one lap around the island obstacle in World.loop_corridor(),
via a turn-to-face-then-drive proportional waypoint controller (the same steering-law shape
as go1_rl_walk/mpc/navigate_mpc.py's compute_nav_command_turn, which already proved itself
there). Factored out of demo_stage1_sim.py so later stage demos reuse the identical
trajectory instead of each re-deriving their own -- the point of every stage past Stage 1 is
to test a mapping/localization algorithm against a KNOWN, already-verified motion, not to
re-verify the simulator again each time.
"""
import numpy as np

from .robot import wrap_to_pi

WAYPOINTS = np.array([
    [1.5, 1.5], [10.5, 1.5], [10.5, 6.5], [1.5, 6.5], [1.5, 1.5],
])


def waypoint_command(pose, goal, kp_v=1.0, kp_w=2.0, v_max=1.2, w_max=1.5):
    x, y, theta = pose
    dx, dy = goal[0] - x, goal[1] - y
    dist = float(np.hypot(dx, dy))
    heading_err = wrap_to_pi(float(np.arctan2(dy, dx)) - theta)
    omega = float(np.clip(kp_w * heading_err, -w_max, w_max))
    v = float(np.clip(kp_v * dist, 0.0, v_max) * max(0.0, np.cos(heading_err)))
    return v, omega, dist


def run_loop(robot, dt=0.1, max_steps=1200, waypoint_tolerance=0.3, on_step=None):
    """Drives `robot` once around WAYPOINTS, calling on_step(step, robot) after every
    robot.step() (if given) -- e.g. to take a lidar scan and update a map. Returns the
    number of steps actually taken."""
    wp_idx = 1
    step = 0
    for step in range(max_steps):
        goal = WAYPOINTS[wp_idx]
        v, omega, dist = waypoint_command(robot.true_pose, goal)
        if dist <= waypoint_tolerance:
            wp_idx += 1
            if wp_idx >= len(WAYPOINTS):
                break
            continue
        robot.step(v, omega, dt)
        if on_step is not None:
            on_step(step, robot)
    return step + 1
