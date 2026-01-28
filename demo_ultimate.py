from BinCore import (
    ItemType,
    BoxType,
    pack_order,
    pack_single_sku_order,
    print_packing_summary,
    print_box_stock_usage,
)
from plotter3d import visualize_boxes_with_buttons


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
    box_types = [
        BoxType("Pacco-S", 12.5, 10, 26.5, 0.5, max_boxes=5),
        BoxType("Pacco-M", 11, 20, 27, 1.0, max_boxes=5),
        # BoxType("Pacco-XL", 30, 30, 30, 1.5, max_boxes=3),
        # BoxType("Pacco-XXL", 40, 40, 40, 2.0, max_boxes=3),
    ]

    # Example A: single SKU (will use pack_single_sku_order)
    # items = [
    #     ItemType("CableSep", 10.5, 4, 19, quantity=10),
    # ]

    # Example B: multi-SKU (uncomment to test multi-SKU logic)
    # items = [
    #     ItemType("CableSep", 10.5, 4, 19, quantity=10),
    #     ItemType("Adapter:Sigma", 8, 9, 20, quantity=5),
    # ]

    boxes, unassigned, used_box_types = smart_pack(items, box_types, grid_resolution=0.5)

    total_cost = sum(b.box_type.cost for b in boxes)
    print(f"Total boxes used: {len(boxes)}")
    print(f"Estimated packaging cost: {total_cost}\n")

    print_box_stock_usage(used_box_types, boxes)
    print_packing_summary(boxes, unassigned)
    visualize_boxes_with_buttons(boxes, unassigned)
