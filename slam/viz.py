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
                  true_trail=None, odom_trail=None, belief_trail=None, map_points=None,
                  occupancy_grid=None, show_ground_truth_walls=True,
                  labels=None, status_text=None,
                  figsize=(6, 4.5), dpi=100) -> np.ndarray:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    xmin, xmax, ymin, ymax = world.bounds
    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#f0f0f0")

    if occupancy_grid is not None:
        # imshow's own pixel convention is row 0 = top, but grid row 0 = ymin (bottom) --
        # origin="lower" matches them up so the map isn't rendered upside down.
        prob = occupancy_grid.probability_map()
        gxmin, gymin = occupancy_grid.xmin, occupancy_grid.ymin
        gxmax = gxmin + occupancy_grid.nx * occupancy_grid.resolution
        gymax = gymin + occupancy_grid.ny * occupancy_grid.resolution
        # 1 - prob: occupied (prob high) -> dark, free (prob low) -> light, unknown (0.5) -> mid gray.
        ax.imshow(1.0 - prob, cmap="gray", vmin=0.0, vmax=1.0, origin="lower",
                  extent=(gxmin, gxmax, gymin, gymax), zorder=0)

    if show_ground_truth_walls:
        for x1, y1, x2, y2 in world.walls:
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=1,
                    linestyle=":" if occupancy_grid is not None else "-",
                    alpha=0.6 if occupancy_grid is not None else 1.0)

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
    if belief_trail is not None and len(belief_trail) > 1:
        ax.plot(belief_trail[:, 0], belief_trail[:, 1], color="#8844cc", linewidth=1.2,
                linestyle="-.", label="SLAM belief (corrected)")

    _draw_robot(ax, true_pose, color="#2a9d4a")
    if odom_pose is not None:
        _draw_robot(ax, odom_pose, color="#e08b2a")

    if labels is not None:
        for lx, ly, text in labels:
            ax.text(lx, ly, text, fontsize=7, color="#444", ha="center", va="center",
                     style="italic", zorder=5)

    if status_text is not None:
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes, fontsize=9,
                 va="top", ha="left", color="#222",
                 bbox=dict(boxstyle="round", facecolor="#ffe08a", alpha=0.9))

    if ax.get_legend_handles_labels()[0]:
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


def render_pointcloud_3d(points: np.ndarray, bounds, n_frames: int = 36,
                          elevation: float = 25.0, azimuth_start: float = 0.0,
                          azimuth_sweep: float = 360.0, point_size: float = 2.0,
                          figsize=(6.4, 5.4), dpi=100) -> list:
    """Render an accumulated (N, 3) world-frame point cloud as a rotating-azimuth GIF's
    worth of frames, colored by height (z) -- the one piece of structure a flat 2D
    occupancy-grid view can't show at all (wall faces vs. wall tops)."""
    xmin, xmax, ymin, ymax = bounds
    frames = []
    azimuths = np.linspace(azimuth_start, azimuth_start + azimuth_sweep, n_frames,
                            endpoint=False)
    for az in azimuths:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(projection="3d")
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2],
                       cmap="viridis", s=point_size, depthshade=True)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(0.0, max(1.0, float(points[:, 2].max()) if len(points) else 1.0))
        ax.set_box_aspect((xmax - xmin, ymax - ymin,
                           max(1.0, (float(points[:, 2].max()) if len(points) else 1.0))))
        ax.view_init(elev=elevation, azim=float(az))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        frames.append(frame)
    return frames


def render_pointcloud_3d_panel(points: np.ndarray, bounds, views=None,
                                point_size: float = 1.5, figsize=(15, 9),
                                dpi=110) -> np.ndarray:
    """A single STATIC image: the same point cloud rendered from several fixed angles at
    once (top-down, two side elevations, and a few 3/4 isometric views) -- lets you "look
    from a different angle" by looking at a different panel, with no rotating GIF and no
    extra viewer/dependency needed."""
    if views is None:
        views = [
            ("top-down", 90.0, -90.0),
            ("front (looking +y)", 0.0, -90.0),
            ("side (looking +x)", 0.0, 0.0),
            ("isometric NE", 25.0, 45.0),
            ("isometric NW", 25.0, 135.0),
            ("steep oblique", 55.0, -45.0),
        ]
    xmin, xmax, ymin, ymax = bounds
    zmax = max(1.0, float(points[:, 2].max()) if len(points) else 1.0)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ncols = 3
    nrows = (len(views) + ncols - 1) // ncols
    for i, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2],
                       cmap="viridis", s=point_size, depthshade=True)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(0.0, zmax)
        ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def save_pointcloud_interactive_html(points: np.ndarray, bounds, path, point_size: float = 2.0,
                                      start_inside: bool = True):
    """A genuinely interactive point-cloud viewer: a self-contained HTML file (Plotly,
    embedded, no internet needed to open it) the user can drag to orbit, scroll to zoom in
    or out, and double-click to recenter -- actually being "inside" the cloud and looking
    in a chosen direction, not just picking from a few pre-rendered angles.

    Imports plotly lazily (not at module level) since most of this project's demos never
    need it and run with a plain python3 that doesn't have it installed -- only the Stage
    8/9 scripts (run with go1_rl_walk/mpc's venv, which also has mujoco) use this."""
    import plotly.graph_objects as go

    xmin, xmax, ymin, ymax = bounds
    zmax = max(1.0, float(points[:, 2].max()) if len(points) else 1.0)

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=point_size, color=points[:, 2], colorscale="Viridis", opacity=0.85),
    )])
    # Plotly's camera "eye"/"center" are in a normalized scene-relative frame (roughly a
    # unit cube around the data regardless of actual data units) -- small eye coordinates
    # place the initial view close to the scene's own center, i.e. inside the building,
    # rather than the far-away default that only ever looks at it from outside.
    eye = dict(x=0.12, y=0.12, z=0.09) if start_inside else dict(x=1.25, y=1.25, z=1.0)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[xmin, xmax], title="x"),
            yaxis=dict(range=[ymin, ymax], title="y"),
            zaxis=dict(range=[0.0, zmax], title="z"),
            aspectmode="data",   # true relative proportions, not auto-stretched to a cube
            camera=dict(eye=eye, center=dict(x=0, y=0, z=0)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title="3D lidar point cloud -- drag to orbit, scroll to zoom, double-click to reset",
    )
    fig.write_html(str(path), include_plotlyjs=True)


def save_image(frame: np.ndarray, path):
    Image.fromarray(frame).save(path)


def save_gif(frames, path, fps=10):
    # optimize=True deliberately NOT used -- matches go1_rl_walk's own established finding
    # that it corrupts/bloats multi-frame GIF output on this Pillow version.
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
