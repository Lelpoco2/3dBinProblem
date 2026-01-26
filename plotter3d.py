"""
3D visualization helpers for packing_3d.BoxInstance and PlacedItem.

Features:
- 3D box drawing with items as cuboids.
- Item labels on top of each cuboid.
- Previous/Next buttons to navigate between boxes.
- Text summary (box + unpacked items) inside the window.

Requires:
    matplotlib
"""

from typing import List
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from BinCore import BoxInstance, ItemType

def set_axis_equal(ax):
    """
    Set 3D plot axes to equal scale.
    """
    
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    
    x_middle = (x_limits[1] + x_limits[0]) / 2
    y_middle = (y_limits[1] + y_limits[0]) / 2
    z_middle = (z_limits[1] + z_limits[0]) / 2
    
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def _cuboid_vertices(origin, size):
    """
    Return list of 6 faces (each face is 4 vertices) for a cuboid.
    origin: (x0, y0, z0)
    size: (dx, dy, dz)
    """
    x0, y0, z0 = origin
    dx, dy, dz = size

    x = [x0, x0 + dx]
    y = [y0, y0 + dy]
    z = [z0, z0 + dz]

    faces = [
        # bottom
        [(x[0], y[0], z[0]), (x[1], y[0], z[0]),
         (x[1], y[1], z[0]), (x[0], y[1], z[0])],
        # top
        [(x[0], y[0], z[1]), (x[1], y[0], z[1]),
         (x[1], y[1], z[1]), (x[0], y[1], z[1])],
        # front (y = y[0])
        [(x[0], y[0], z[0]), (x[1], y[0], z[0]),
         (x[1], y[0], z[1]), (x[0], y[0], z[1])],
        # back (y = y[1])
        [(x[0], y[1], z[0]), (x[1], y[1], z[0]),
         (x[1], y[1], z[1]), (x[0], y[1], z[1])],
        # left (x = x[0])
        [(x[0], y[0], z[0]), (x[0], y[1], z[0]),
         (x[0], y[1], z[1]), (x[0], y[0], z[1])],
        # right (x = x[1])
        [(x[1], y[0], z[0]), (x[1], y[1], z[0]),
         (x[1], y[1], z[1]), (x[1], y[0], z[1])],
    ]
    return faces


def draw_box(ax, L, W, H, color='lightgray', alpha=0.1):
    """
    Draw the outer box as a translucent cuboid.
    """
    faces = _cuboid_vertices((0, 0, 0), (L, W, H))
    box = Poly3DCollection(faces, facecolors=color,
                           linewidths=1, edgecolors='k', alpha=alpha)
    ax.add_collection3d(box)


def draw_item(ax, position, dims, label: str,
              color='tab:blue', alpha=0.6):
    """
    Draw one item as a colored cuboid and write its label on top.
    """
    faces = _cuboid_vertices(position, dims)
    cuboid = Poly3DCollection(faces, facecolors=color,
                              linewidths=0.5, edgecolors='k', alpha=alpha)
    ax.add_collection3d(cuboid)

    # Text on top center
    x0, y0, z0 = position
    l, w, h = dims
    tx = x0 + l / 2.0
    ty = y0 + w / 2.0
    tz = z0 + h

    base_label = label.split("#")[0]
    ax.text(tx, ty, tz + 0.5, base_label,
            ha='center', va='bottom', fontsize=8, color='black')


def _box_summary_text(box_instance: BoxInstance) -> str:
    """
    Multi-line text summary for a single box.
    """
    bt = box_instance.box_type
    counter = Counter(it.item_id.split("#")[0] for it in box_instance.items)

    lines = []
    lines.append(f"Box {bt.id}-{box_instance.instance_index}")
    lines.append(f"Size: {bt.inner_length}x{bt.inner_width}x{bt.inner_height}")
    lines.append("Items:")
    if counter:
        for item_id, qty in counter.items():
            lines.append(f"  - {item_id}: {qty}")
    else:
        lines.append("  (empty)")
    return "\n".join(lines)


def _unassigned_summary_text(unassigned_items: List[ItemType]) -> str:
    """
    Multi-line summary for items that could not be packed.
    """
    if not unassigned_items:
        return "Unpacked items:\n  (none)"

    counter = Counter(it.id.split("#")[0] for it in unassigned_items)
    lines = ["Unpacked items:"]
    for item_id, qty in counter.items():
        lines.append(f"  - {item_id}: {qty}")
    return "\n".join(lines)


def visualize_boxes_with_buttons(
    boxes: List[BoxInstance],
    unassigned_items: List[ItemType],
):
    """
    Show a single window with 'Previous' and 'Next' buttons
    to switch between boxes interactively, plus a text summary
    (current box + global unpacked items).
    """
    if not boxes:
        print("No boxes to visualize.")
        return

    state = {"i": 0}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # keep a reference to the text artist so we can update it
    text_box = {"artist": None}

    def redraw():
        ax.clear()
        box_instance = boxes[state["i"]]

        L = box_instance.box_type.inner_length
        W = box_instance.box_type.inner_width
        H = box_instance.box_type.inner_height

        # Draw outer box
        draw_box(ax, L, W, H, color="lightgray", alpha=0.1)

        colors = [
            "tab:blue", "tab:orange", "tab:green", "tab:red",
            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        ]

        for idx, it in enumerate(box_instance.items):
            color = colors[idx % len(colors)]
            draw_item(ax, it.position, it.rotation, it.item_id,
                      color=color, alpha=0.6)

        ax.set_xlim(0, L)
        ax.set_ylim(0, W)
        ax.set_zlim(0, H)
        ax.set_xlabel("X (length)")
        ax.set_ylabel("Y (width)")
        ax.set_zlabel("Z (height)")
        ax.set_title(f"Box {box_instance.box_type.id}-{box_instance.instance_index} "
                     f"({state['i']+1}/{len(boxes)})")
        
        # Set equal scaling
        set_axis_equal(ax)

        # --- Text summary in the figure (bottom-left) ---
        box_summary = _box_summary_text(box_instance)
        unassigned_summary = _unassigned_summary_text(unassigned_items)
        full_summary = box_summary + "\n\n" + unassigned_summary

        if text_box["artist"] is not None:
            text_box["artist"].remove()

        text_artist = fig.text(
            0.01, 0.01, full_summary,
            fontsize=8,
            va="bottom", ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
        )
        text_box["artist"] = text_artist

        plt.draw()

    class Index:
        def next(self, event):
            state["i"] = (state["i"] + 1) % len(boxes)
            redraw()

        def prev(self, event):
            state["i"] = (state["i"] - 1) % len(boxes)
            redraw()

    callback = Index()

    # Buttons under the plot
    axprev = fig.add_axes([0.3, 0.02, 0.1, 0.05])
    axnext = fig.add_axes([0.6, 0.02, 0.1, 0.05])

    bprev = Button(axprev, "Previous")
    bprev.on_clicked(callback.prev)

    bnext = Button(axnext, "Next")
    bnext.on_clicked(callback.next)

    redraw()
    plt.show()
