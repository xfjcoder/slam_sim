# 2D SLAM Sandbox

A from-scratch SLAM (Simultaneous Localization and Mapping) project: build the classical
occupancy-grid-mapping + scan-matching-localization pipeline from first principles, staged
and verified at each step.

A separate project (and repo) from `go1_rl_walk` (a sibling directory) — that project
trained a Unitree Go1/Go2 quadruped to walk and navigate (MuJoCo + PPO, then a classical MPC
controller) and ended with a working point-to-goal navigation stack, but every "obstacle
avoidance" mechanism there used privileged ground-truth or a short-range 9-ray near-field
fan built for footstep planning, not real mapping or localization. This project starts
over on the actual SLAM problem, deliberately away from that project's quadruped/MuJoCo
specifics:

- **Simplified kinematic robot, not the full quadruped.** A unicycle/differential-drive
  model with no mass, dynamics, or balance to fail — locomotion noise and falls would be a
  confounding variable while developing and debugging a mapping/localization algorithm, not
  a useful feature of this sandbox.
- **A real 2D lidar scan (360 beams), not a sparse near-field fan.** This is the standard
  sensor classical SLAM algorithms (GMapping, Hector SLAM, Cartographer) are built around —
  "richer sensing" than the quadruped project's own 9-ray fan was a deliberate choice, not
  an oversight carried over.
- **A world with a loop in it from the start** (a rectangular island obstacle inside an
  outer boundary, leaving a loop-shaped corridor around it) — loop closure is the
  signature hard problem in SLAM, so the sandbox is built to support it from Stage 1,
  rather than retrofitted later.

## Staged plan

1. **Simulator core** (world, kinematic robot, lidar ray-casting) — DONE. Verified
   visually, not just by unit numbers: lidar hits should hug every wall tightly, and the
   dead-reckoning odometry estimate should visibly drift away from the true path over a
   lap — that drift is the whole reason the later stages exist.
2. Occupancy-grid mapping, given the TRUE pose (perfect localization) — isolates the
   mapping half of the problem first, before localization noise is in the picture at all.
3. Mapping from the NOISY odometry estimate instead — shows concretely how dead-reckoning
   drift corrupts a map over time, motivating scan-matching.
4. Scan-matching (e.g. a simple point-to-point ICP) to correct the pose estimate against
   the map being built, closing the basic SLAM loop.
5. (Stretch) Loop closure / pose-graph correction, exploiting the deliberately-built loop
   in the world.

### Stage 1 demo

<p align="center">
  <img src="media/stage1_sim_and_drift.gif" width="520" alt="Simplified kinematic robot driving a loop around the island obstacle; lidar hits (red) hug every wall, true path (green) vs. dead-reckoning odometry estimate (orange, dashed) visibly diverge">
</p>

Scripted waypoint-following (a turn-to-face-then-drive proportional controller, reused from
`go1_rl_walk/mpc/navigate_mpc.py`'s own steering law, which already proved itself there)
drives one full lap around the island. Lidar hits (red) hug every wall exactly as expected;
the true path (green, solid) and the dead-reckoning odometry estimate (orange, dashed)
start overlapping and visibly diverge over the lap — real, not simulated-for-effect, drift
from the odometry noise model in `slam/robot.py`.

```bash
pip install -r requirements.txt
python3 demo_stage1_sim.py --record media/stage1_sim_and_drift.gif
```

## Repository layout

```
slam_sim/
├── slam/
│   ├── world.py     # ground-truth 2D map: wall line segments, free-space check
│   ├── robot.py      # unicycle kinematics, true pose vs. noisy dead-reckoning odometry
│   ├── lidar.py       # 360-beam ray-casting sensor against the world's walls
│   └── viz.py          # render a frame (world + poses + scan) -> GIF
├── demo_stage1_sim.py # Stage 1 demo: scripted loop, verifies the simulator itself
├── media/              # demo GIFs, one per stage
├── requirements.txt
└── README.md
```
