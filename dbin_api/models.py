# dbin_api/models.py
"""
Core datamodels for dbin_api.

This module provides:
- Dataclass-based core models used by the packing engine (geometry-focused).
- Pydantic models used for API input/output (serialization & validation).
- Small conversion helpers between dataclasses and pydantic models.

Keep dataclasses free of framework-specific dependencies so they can be used
directly by the packing algorithm. Pydantic models are thin wrappers for
validation/IO when exposing the functionality through FastAPI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, validator

# ----------------------------
# Dataclass core models
# ----------------------------


@dataclass
class ItemType:
    """
    Geometry-only representation of an item type.

    Attributes:
    - id: internal unique identifier (string)
    - length, width, height: float dimensions (same units as boxes)
    - quantity: number of identical items (for packing flows that accept aggregated SKUs)
    - can_rotate: whether 90-degree rotations are allowed
    - name: optional human readable name
    """

    id: str
    length: float
    width: float
    height: float
    quantity: int = 0
    can_rotate: bool = True
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.id

    @property
    def volume(self) -> float:
        """Geometric volume (length * width * height)."""
        return self.length * self.width * self.height


@dataclass
class BoxType:
    """
    Geometry-only representation of a box/container type.

    Attributes:
    - id: internal unique identifier
    - inner_length, inner_width, inner_height: usable interior dimensions
    - cost: economic cost per box (optional metric used by planners)
    - max_boxes: optional stock limit (None means unlimited)
    - effective_volume: optional override for usable volume (useful for bags or irregular boxes)
    - container_type: "BOX" (rigid) or "BAG" (flexible)
    - name: optional human-readable name
    """

    id: str
    inner_length: float
    inner_width: float
    inner_height: float
    cost: float
    max_boxes: Optional[int] = None
    effective_volume: Optional[float] = None
    container_type: Literal["BOX", "BAG"] = "BOX"
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.id
        if self.container_type == "BAG" and self.effective_volume is None:
            # Keep behavior consistent with original algorithm:
            raise ValueError("BAG container types must specify effective_volume")

    @property
    def volume(self) -> float:
        """Return effective usable volume: effective_volume override or geometric interior volume."""
        if self.effective_volume is not None:
            return self.effective_volume
        return self.inner_length * self.inner_width * self.inner_height


@dataclass
class PlacedItem:
    """
    Concrete placed item inside a concrete box instance.

    - item_id: id of the item instance (may include suffixes for expanded items)
    - box_id: string identifier of the box instance (for example: "<box_type>-<index>")
    - length, width, height: original item dimensions
    - position: (x, y, z) coordinates of the item's minimum corner inside the box
    - rotation: (l, w, h) tuple representing dimensions after rotation (axis-aligned)
    - item_name: optional human-readable name
    """

    item_id: str
    box_id: str
    length: float
    width: float
    height: float
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    item_name: str = ""


@dataclass
class BoxInstance:
    """
    Concrete instance of a box we are filling.

    - box_type: BoxType used for this instance
    - instance_index: 1-based index of this instance for human-readable ids
    - items: list of PlacedItem objects
    """

    box_type: BoxType
    instance_index: int
    items: List[PlacedItem] = field(default_factory=list)

    def used_volume(self) -> float:
        """Sum of volumes of placed items."""
        return sum(p.rotation[0] * p.rotation[1] * p.rotation[2] for p in self.items)


# ----------------------------
# Pydantic models for API surface
# ----------------------------

# Input models (Create / Request)


class ItemCreate(BaseModel):
    id: str = Field(..., description="Unique id for the item SKU")
    length: float = Field(..., gt=0)
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    quantity: int = Field(1, ge=0, description="Number of identical items")
    can_rotate: bool = Field(True)
    name: Optional[str] = Field(None)

    class Config:
        schema_extra = {
            "example": {
                "id": "SKU-123",
                "length": 10.0,
                "width": 5.0,
                "height": 2.5,
                "quantity": 3,
                "can_rotate": True,
                "name": "Gadget",
            }
        }


class BoxTypeCreate(BaseModel):
    id: str = Field(..., description="Unique id for the box type")
    inner_length: float = Field(..., gt=0)
    inner_width: float = Field(..., gt=0)
    inner_height: float = Field(..., gt=0)
    cost: float = Field(..., ge=0)
    max_boxes: Optional[int] = Field(None, ge=1)
    effective_volume: Optional[float] = Field(None, gt=0)
    container_type: Literal["BOX", "BAG"] = Field("BOX")
    name: Optional[str] = Field(None)

    @validator("effective_volume")
    def validate_effective_vs_dimensions(
        cls, v: Optional[float], values: Dict[str, Any]
    ) -> Optional[float]:  # type: ignore
        # If provided, it should not exceed geometric volume by an unreasonable amount
        if v is None:
            return v
        inner_length = values.get("inner_length")
        inner_width = values.get("inner_width")
        inner_height = values.get("inner_height")
        if inner_length is None or inner_width is None or inner_height is None:
            return v
        geom = inner_length * inner_width * inner_height
        if v > geom * 1.5:
            # allow some tolerance but guard against accidental huge values
            raise ValueError(
                "effective_volume unusually large compared to geometric inner volume"
            )
        return v

    @validator("container_type")
    def require_effective_for_bag(cls, v: str, values: Dict[str, Any]) -> str:  # type: ignore
        if v == "BAG" and values.get("effective_volume") is None:
            raise ValueError("BAG container_type must provide effective_volume")
        return v

    class Config:
        schema_extra = {
            "example": {
                "id": "BOX-42",
                "inner_length": 40.0,
                "inner_width": 30.0,
                "inner_height": 20.0,
                "cost": 0.5,
                "max_boxes": None,
                "effective_volume": None,
                "container_type": "BOX",
                "name": "Standard Box",
            }
        }


# Output models (Read / Response)


class PlacedItemRead(BaseModel):
    item_id: str
    box_id: str
    length: float
    width: float
    height: float
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    item_name: Optional[str]

    class Config:
        orm_mode = True


class BoxInstanceRead(BaseModel):
    box_type_id: str
    box_type_name: Optional[str]
    instance_index: int
    inner_length: float
    inner_width: float
    inner_height: float
    cost: float
    container_type: str
    items: List[PlacedItemRead]
    used_volume: float
    capacity_volume: float
    utilization: float

    class Config:
        orm_mode = True


class PackingResult(BaseModel):
    boxes: List[BoxInstanceRead]
    unassigned_items: List[ItemCreate]
    summary: Optional[Dict[str, Any]] = None


# ----------------------------
# Conversion helpers
# ----------------------------


def itemcreate_to_dataclass(ic: ItemCreate) -> ItemType:
    """Convert ItemCreate (pydantic) to ItemType dataclass."""
    return ItemType(
        id=ic.id,
        length=ic.length,
        width=ic.width,
        height=ic.height,
        quantity=ic.quantity,
        can_rotate=ic.can_rotate,
        name=ic.name or ic.id,
    )


def boxtypecreate_to_dataclass(bc: BoxTypeCreate) -> BoxType:
    """Convert BoxTypeCreate (pydantic) to BoxType dataclass."""
    return BoxType(
        id=bc.id,
        inner_length=bc.inner_length,
        inner_width=bc.inner_width,
        inner_height=bc.inner_height,
        cost=bc.cost,
        max_boxes=bc.max_boxes,
        effective_volume=bc.effective_volume,
        container_type=bc.container_type,
        name=bc.name or bc.id,
    )


def placeditem_from_dataclass(pi: PlacedItem) -> PlacedItemRead:
    """Convert dataclass PlacedItem to Pydantic PlacedItemRead."""
    return PlacedItemRead(
        item_id=pi.item_id,
        box_id=pi.box_id,
        length=pi.length,
        width=pi.width,
        height=pi.height,
        position=pi.position,
        rotation=pi.rotation,
        item_name=pi.item_name or None,
    )


def boxinstance_from_dataclass(bi: BoxInstance) -> BoxInstanceRead:
    """Convert dataclass BoxInstance to Pydantic BoxInstanceRead."""
    items_read = [placeditem_from_dataclass(pi) for pi in bi.items]
    used_vol = bi.used_volume()
    cap_vol = bi.box_type.volume
    util = (used_vol / cap_vol) * 100.0 if cap_vol > 0 else 0.0
    return BoxInstanceRead(
        box_type_id=bi.box_type.id,
        box_type_name=bi.box_type.name,
        instance_index=bi.instance_index,
        inner_length=bi.box_type.inner_length,
        inner_width=bi.box_type.inner_width,
        inner_height=bi.box_type.inner_height,
        cost=bi.box_type.cost,
        container_type=bi.box_type.container_type,
        items=items_read,
        used_volume=used_vol,
        capacity_volume=cap_vol,
        utilization=util,
    )


def packing_result_from_dataclasses(
    boxes: List[BoxInstance],
    unassigned: List[ItemType],
    summary: Optional[Dict[str, Any]] = None,
) -> PackingResult:
    """Convenience helper to create a Pydantic PackingResult from dataclass outputs."""
    boxes_read = [boxinstance_from_dataclass(b) for b in boxes]
    # Convert unassigned dataclass ItemType -> ItemCreate like dicts for consistent output
    unassigned_reads = [
        ItemCreate(
            id=it.id,
            length=it.length,
            width=it.width,
            height=it.height,
            quantity=it.quantity,
            can_rotate=it.can_rotate,
            name=it.name,
        )
        for it in unassigned
    ]
    return PackingResult(
        boxes=boxes_read, unassigned_items=unassigned_reads, summary=summary or {}
    )


# Expose minimal public API from this module
__all__ = [
    "ItemType",
    "BoxType",
    "PlacedItem",
    "BoxInstance",
    "ItemCreate",
    "BoxTypeCreate",
    "PlacedItemRead",
    "BoxInstanceRead",
    "PackingResult",
    "itemcreate_to_dataclass",
    "boxtypecreate_to_dataclass",
    "packing_result_from_dataclasses",
]
