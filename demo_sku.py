from BinCore import (
    ItemType,
    BoxType,
    pack_single_sku_order,
    print_packing_summary,
    print_box_stock_usage,
)
from plotter3d import visualize_boxes_with_buttons

if __name__ == "__main__":
    box_types = [
        BoxType("Pacco-S", 12.5, 10, 26.5, 0.5, max_boxes=100),
        BoxType("Pacco-M", 11, 20, 27, 1.0, max_boxes=100),
        # BoxType("Pacco-XL", 30,   30, 30, 1.0, max_boxes=100),
        # BoxType("Pacco-XXL", 40,   40, 40, 1.0, max_boxes=100),
    ]

    itemsku = ItemType("CableSep", 10.5, 4, 19, quantity=10)

    boxes, unassigned, planned_box_types, mix = pack_single_sku_order(
        itemsku,
        box_types,
        grid_resolution=0.5,
    )

    print("Planned box mix:", mix)
    total_cost = sum(b.box_type.cost for b in boxes)
    print(f"Total boxes used: {len(boxes)}")
    print(f"Estimated packaging cost: {total_cost}\n")

    print_box_stock_usage(planned_box_types, boxes)
    print_packing_summary(boxes, unassigned)
    visualize_boxes_with_buttons(boxes, unassigned)