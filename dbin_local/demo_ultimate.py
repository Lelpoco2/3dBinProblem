from BinCore import (
    ItemType,
    BoxType,
    pack_order,
    pack_single_sku_order,
    print_packing_summary,
    print_box_stock_usage,
)
from plotter3d import visualize_boxes_with_buttons, visualize_all_boxes


def smart_pack(items, box_types, grid_resolution=0.5):
    # Normalize to list
    if isinstance(items, ItemType):
        items_list = [items]
    else:
        items_list = items

    # 1) If only ONE ItemType in the list -> use precise single-SKU planner
    if len(items_list) == 1:
        sku = items_list[0]
        print("Using single-SKU optimizer for:", sku.id)
        boxes, unassigned, planned_box_types, mix = pack_single_sku_order(
            sku, box_types, grid_resolution=grid_resolution
        )
        print("Planned box mix:", mix)
        return boxes, unassigned, planned_box_types

    # 2) If multiple different ItemTypes -> use general multi-SKU packer
    print("Using multi-SKU packer for", len(items_list), "item types")
    boxes, unassigned = pack_order(items_list, box_types, grid_resolution=grid_resolution)
    return boxes, unassigned, box_types


if __name__ == "__main__":
    # Three box types of increasing size
    box_types = [
        BoxType(id="BOX_S", name="Pacco-S", inner_length=15, inner_width=12, inner_height=10, cost=1.0, max_boxes=2, container_type="BOX"),
        BoxType(id="BOX_M", name="Pacco-M", inner_length=25, inner_width=20, inner_height=15, cost=2.0, max_boxes=2, container_type="BOX"),
        BoxType(id="BOX_L", name="Pacco-L", inner_length=35, inner_width=28, inner_height=20, cost=3.5, max_boxes=2, container_type="BOX"),
    ]

    # Items sized so each lands in a different box.
    # "Colosso" is intentionally too large for any box → unassigned.
    items = [
        ItemType(id="ITEM001", name="Gadget",   length=14, width=11, height=9,  quantity=1),  # → Pacco-S
        ItemType(id="ITEM002", name="Giacca",   length=24, width=19, height=14, quantity=1),  # → Pacco-M
        ItemType(id="ITEM003", name="Monitor",  length=34, width=27, height=19, quantity=1),  # → Pacco-L
        ItemType(id="ITEM004", name="Colosso",  length=50, width=50, height=50, quantity=1),  # → unassigned
    ]

    boxes, unassigned, used_box_types = smart_pack(items, box_types, grid_resolution=0.5)

    total_cost = sum(b.box_type.cost for b in boxes)
    print(f"Total boxes used: {len(boxes)}")
    print(f"Estimated packaging cost: €{total_cost:.2f}\n")

    print_box_stock_usage(used_box_types, boxes)
    print_packing_summary(boxes, unassigned)
    visualize_all_boxes(boxes, unassigned, input_box_types=box_types, input_items=items)
