from BinCore import (
    ItemType,
    BoxType,
    pack_order,
    print_packing_summary,
    print_box_stock_usage,
)
from plotter3d import visualize_boxes_with_buttons

if __name__ == "__main__":
    # Define available box types
    box_types = [
        BoxType("Pacco-S", 12.5, 10, 26.5, 0.5, max_boxes=5),
        BoxType("Pacco-M", 11, 20, 27, 1.0, max_boxes=5),
        # Uncomment these if you want to test XL/XXL
        # BoxType("Pacco-XL", 30, 30, 30, 1.5, max_boxes=3),
        # BoxType("Pacco-XXL", 40, 40, 40, 2.0, max_boxes=3),
    ]

    # Define a LIST of different items to pack
    items_to_pack = [
        ItemType("MyPOS:Go2", 11, 18.5, 8.5, quantity=7),
        ItemType("Phone", 15, 8, 1.5, quantity=3),
        ItemType("Charger", 5, 5, 2, quantity=16),
    ]

    # Pack all items
    boxes, unassigned = pack_order(
        items_to_pack,
        box_types,
        grid_resolution=0.5,
    )

    # Print summary
    print("=" * 50)
    total_cost = sum(b.box_type.cost for b in boxes)
    print(f"Total boxes used: {len(boxes)}")
    print(f"Estimated packaging cost: {total_cost}\n")

    print_box_stock_usage(box_types, boxes)
    print_packing_summary(boxes, unassigned)

    # Visualize
    visualize_boxes_with_buttons(boxes, unassigned)
