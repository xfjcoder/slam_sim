"""Stage 7: an office floorplan, plus a small named-location "mission" layer on top of the
existing waypoint-chasing steering law (`slam.trajectory.waypoint_command`) -- give the
robot an ordered list of job names (kitchen, conference_a, conference_b, cubicles,
fire_extinguisher) and it visits each one in turn via a shared HUB waypoint in the open
cubicle/corridor area, dwelling there to "perform" a short scripted task before moving to
the next. Every SLAM primitive from Stages 1-6 is reused completely unmodified; what's new
here is just a richer, multi-room ground-truth layout (three separate rooms behind
doorways, two cubicle-divider clusters, and a small fire-extinguisher cabinet) and this
named-location/job registry.

No path planning or obstacle avoidance: every HUB<->job path below is hand-placed so the
straight-line segments the existing turn-to-face controller drives never cross a wall --
same deliberate scope decision as Stage 6's pillar placement.
"""
import numpy as np

from .world import World

START = (7.0, 0.5)   # the office "entrance", where the mission begins and ends
HUB = (7.0, 5.0)      # shared waypoint every job path starts and ends from

ROOM_LABELS = [
    (2.0, 9.3, "Kitchen"),
    (6.5, 9.3, "Conference A"),
    (11.5, 9.3, "Conference B"),
    (2.0, 3.6, "Cubicles"),
    (13.0, 1.6, "Fire ext."),
]

# "path": waypoints from HUB (exclusive) to the job spot (inclusive) -- the return trip is
# just this reversed, through HUB again, so a mission can chain jobs in any order.
JOBS = {
    "kitchen": {
        "path": [(2.0, 5.0), (2.0, 6.5), (2.0, 8.0)],
        "task": "Making coffee",
        "dwell": 20,
    },
    "conference_a": {
        "path": [(6.5, 5.0), (6.5, 6.5), (6.5, 8.0)],
        "task": "Attending a meeting",
        "dwell": 25,
    },
    "conference_b": {
        "path": [(11.5, 5.0), (11.5, 6.5), (11.5, 8.0)],
        "task": "Attending a meeting",
        "dwell": 25,
    },
    "cubicles": {
        "path": [(4.0, 5.0), (4.0, 2.0)],
        "task": "Working at a desk",
        "dwell": 20,
    },
    "fire_extinguisher": {
        "path": [(13.0, 5.0), (13.0, 1.8)],
        "task": "Checking the fire extinguisher",
        "dwell": 15,
    },
}


def build_world() -> World:
    w, h = 14.0, 10.0
    walls = [(0, 0, w, 0), (w, 0, w, h), (w, h, 0, h), (0, h, 0, 0)]  # outer boundary

    # Divider between the top three rooms and the bottom open cubicle/corridor area, with
    # one doorway gap per room, each gap centered under that room's HUB-facing approach lane.
    divider_y = 6.0
    walls += [
        (0.0, divider_y, 1.5, divider_y),    # ... gap 1.5-2.5 (kitchen door) ...
        (2.5, divider_y, 6.0, divider_y),    # ... gap 6.0-7.0 (conference A door) ...
        (7.0, divider_y, 11.0, divider_y),   # ... gap 11.0-12.0 (conference B door) ...
        (12.0, divider_y, w, divider_y),
    ]
    # Vertical dividers splitting the top strip into Kitchen / Conference A / Conference B.
    walls += [(4.0, divider_y, 4.0, h), (9.0, divider_y, 9.0, h)]

    # Two L-shaped cubicle-divider clusters in the open bottom area, well clear of every
    # HUB<->job straight-line path (see module docstring).
    walls += [(1.0, 1.0, 1.0, 3.0), (1.0, 3.0, 3.0, 3.0)]
    walls += [(10.0, 1.0, 10.0, 3.0), (10.0, 3.0, 12.0, 3.0)]

    # A small fire-extinguisher cabinet -- a mapped feature, like Stage 6's pillar.
    walls += [(12.8, 0.6, 13.2, 0.6), (13.2, 0.6, 13.2, 1.0),
              (13.2, 1.0, 12.8, 1.0), (12.8, 1.0, 12.8, 0.6)]

    return World(np.array(walls, dtype=np.float64), bounds=(0.0, w, 0.0, h))


def build_mission(job_names: list[str]):
    """Compose an ordered list of job names into one waypoint path (always starting and
    ending at START, routed through HUB between jobs), and a {waypoint_index: (name, task,
    dwell_steps)} map marking where each job's dwell happens."""
    for name in job_names:
        if name not in JOBS:
            raise ValueError(f"Unknown job {name!r}; choose from {sorted(JOBS)}")

    points = [START, HUB]
    job_at_index = {}
    for name in job_names:
        spec = JOBS[name]
        path = spec["path"]
        points.extend(path)
        job_at_index[len(points) - 1] = (name, spec["task"], spec["dwell"])
        points.extend(reversed(path[:-1]))
        points.append(HUB)
    points.append(START)
    return np.array(points, dtype=np.float64), job_at_index
