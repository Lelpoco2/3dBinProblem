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

import math
from typing import List
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

    ax.text(tx, ty, tz + 0.5, label,
            ha='center', va='bottom', fontsize=11,
            color='#1a1a2e', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))


def _input_panel_text(box_types, items) -> str:
    """
    Format input data (box types + items) into a readable multi-line string.
    """
    lines = ["DATI DI INPUT", ""]

    lines.append("Box disponibili:")
    for bt in box_types:
        max_str = f"max {bt.max_boxes}" if bt.max_boxes is not None else "illimitati"
        cost_str = f"  |  €{bt.cost:.2f}" if bt.cost > 0 else ""
        lines.append(
            f"  • {bt.name}   "
            f"{bt.inner_length}×{bt.inner_width}×{bt.inner_height} cm   "
            f"{max_str}{cost_str}"
        )

    lines.append("")
    lines.append("Prodotti da imballare:")
    for it in items:
        lines.append(
            f"  • {it.name}   "
            f"{it.length}×{it.width}×{it.height} cm   "
            f"qty: {it.quantity}"
        )

    return "\n".join(lines)


_ITEM_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]


def _build_legend(ax, box_instance, colors=_ITEM_COLORS):
    """
    Add an info legend to ax showing box dimensions, volume utilisation,
    and a coloured patch per item.
    """
    bt = box_instance.box_type
    used_vol = box_instance.used_volume()
    total_vol = bt.volume
    util_pct = (used_vol / total_vol * 100) if total_vol > 0 else 0.0

    handles = [
        mpatches.Patch(color='none',
                       label=f"Dim: {bt.inner_length}×{bt.inner_width}×{bt.inner_height} cm"),
        mpatches.Patch(color='none',
                       label=f"Volume: {util_pct:.1f}% utilizzato"),
    ]
    for idx, it in enumerate(box_instance.items):
        handles.append(
            mpatches.Patch(facecolor=colors[idx % len(colors)],
                           edgecolor='k', linewidth=0.5,
                           label=it.item_name, alpha=0.8)
        )

    ax.legend(
        handles=handles,
        loc='upper left',
        fontsize=10,
        title="Informazioni",
        title_fontsize=11,
        framealpha=0.85,
        edgecolor='gray',
    )


def _box_summary_text(box_instance: BoxInstance) -> str:
    """
    Multi-line text summary for a single box.
    """
    bt = box_instance.box_type
    counter = Counter(it.item_name for it in box_instance.items)

    lines = []
    lines.append(f"Box: {bt.name}-{box_instance.instance_index}")
    lines.append(f"Size: {bt.inner_length}x{bt.inner_width}x{bt.inner_height}")
    lines.append("Items:")
    if counter:
        for item_name, qty in counter.items():
            lines.append(f"  - {item_name}: {qty}")
    else:
        lines.append("  (empty)")
    return "\n".join(lines)


def _unassigned_summary_text(unassigned_items: List[ItemType]) -> str:
    """
    Multi-line summary for items that could not be packed.
    """
    if not unassigned_items:
        return "Pacchi non assegnati:\n  (none)"

    counter = Counter(it.name for it in unassigned_items)
    lines = ["Pacchi non assegnati:"]
    for item_name, qty in counter.items():
        lines.append(f"  - {item_name}: {qty}")
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
        draw_box(ax, L, W, H, color="wheat", alpha=0.25)

        # Box name floating above the top center
        ax.text(L / 2, W / 2, H + max(L, W, H) * 0.08,
                f"{box_instance.box_type.name}",
                ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='saddlebrown')

        for idx, it in enumerate(box_instance.items):
            draw_item(ax, it.position, it.rotation, it.item_name,
                      color=_ITEM_COLORS[idx % len(_ITEM_COLORS)], alpha=1)

        ax.set_xlim(0, L)
        ax.set_ylim(0, W)
        ax.set_zlim(0, H)
        ax.set_xlabel("X (length)")
        ax.set_ylabel("Y (width)")
        ax.set_zlabel("Z (height)")
        ax.set_title(f"Box: {box_instance.box_type.id}-{box_instance.instance_index} "
                     f"({state['i']+1}/{len(boxes)})")

        _build_legend(ax, box_instance)
        
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


def visualize_all_boxes(
    boxes: List[BoxInstance],
    unassigned_items: List[ItemType],
    input_box_types=None,
    input_items=None,
):
    """
    Show all boxes in a single figure arranged in a grid of 3D subplots.
    Unpacked items are listed in a banner at the bottom of the figure.
    """
    if not boxes:
        print("No boxes to visualize.")
        return

    n = len(boxes)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig = plt.figure(figsize=(9 * cols, 8 * rows))

    for i, box_instance in enumerate(boxes):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        L = box_instance.box_type.inner_length
        W = box_instance.box_type.inner_width
        H = box_instance.box_type.inner_height

        draw_box(ax, L, W, H, color="wheat", alpha=0.25)

        # Box name floating above the top face
        ax.text(L / 2, W / 2, H + max(L, W, H) * 0.08,
                box_instance.box_type.name,
                ha='center', va='bottom', fontsize=15,
                fontweight='bold', color='saddlebrown')

        for idx, it in enumerate(box_instance.items):
            draw_item(ax, it.position, it.rotation, it.item_name,
                      color=_ITEM_COLORS[idx % len(_ITEM_COLORS)], alpha=0.6)

        ax.set_xlim(0, L)
        ax.set_ylim(0, W)
        ax.set_zlim(0, H)
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_zlabel("Z", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_title(
            f"{box_instance.box_type.name} #{box_instance.instance_index}  "
            f"({len(box_instance.items)} item{'s' if len(box_instance.items) != 1 else ''})",
            fontsize=14, fontweight='bold', pad=12
        )
        set_axis_equal(ax)
        _build_legend(ax, box_instance)

    # Unassigned items banner at the bottom
    unassigned_text = _unassigned_summary_text(unassigned_items)
    fig.text(
        0.5, 0.005, unassigned_text,
        ha='center', va='bottom', fontsize=11,
        bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='darkorange')
    )

    # Input data panel — bottom-left, boxes shifted right to avoid overlap
    left_margin = 0.0
    if input_box_types is not None and input_items is not None:
        left_margin = 0.20
        fig.text(
            0.01, 0.01, _input_panel_text(input_box_types, input_items),
            ha='left', va='bottom', fontsize=10, family='monospace',
            bbox=dict(facecolor='#eef4fb', alpha=0.95, edgecolor='steelblue', pad=8)
        )

    plt.tight_layout(rect=[left_margin, 0.06, 1, 1])
    plt.show()
