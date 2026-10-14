"""Rendering: world + robot pose(s) + lidar hits -> an RGB frame, and frames -> a GIF.

Matplotlib with the Agg backend (no display needed, same as the quadruped project's own
offscreen-rendering convention) -- set MPLBACKEND=Agg before importing pyplot, done once
here so every caller gets it automatically.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def render_frame(world, true_pose, odom_pose=None, scan_points=None,
                  true_trail=None, odom_trail=None, map_points=None,
                  figsize=(6, 4.5), dpi=100) -> np.ndarray:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    xmin, xmax, ymin, ymax = world.bounds
    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#f0f0f0")

    for x1, y1, x2, y2 in world.walls:
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)

    if map_points is not None and len(map_points):
        ax.scatter(map_points[:, 0], map_points[:, 1], s=1, c="#4477aa", alpha=0.5,
                   label="mapped points")

    if scan_points is not None and len(scan_points):
        ax.scatter(scan_points[:, 0], scan_points[:, 1], s=4, c="#dd3344",
                   label="current lidar hits")

    if true_trail is not None and len(true_trail) > 1:
        ax.plot(true_trail[:, 0], true_trail[:, 1], color="#2a9d4a", linewidth=1.2,
                label="true path")
    if odom_trail is not None and len(odom_trail) > 1:
        ax.plot(odom_trail[:, 0], odom_trail[:, 1], color="#e08b2a", linewidth=1.2,
                linestyle="--", label="odometry estimate")

    _draw_robot(ax, true_pose, color="#2a9d4a")
    if odom_pose is not None:
        _draw_robot(ax, odom_pose, color="#e08b2a")

    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def _draw_robot(ax, pose, color, size=0.3):
    x, y, theta = pose
    ax.plot(x, y, marker="o", color=color, markersize=6)
    ax.plot([x, x + size * np.cos(theta)], [y, y + size * np.sin(theta)],
            color=color, linewidth=2)


def save_gif(frames, path, fps=10):
    # optimize=True deliberately NOT used -- matches go1_rl_walk's own established finding
    # that it corrupts/bloats multi-frame GIF output on this Pillow version.
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
