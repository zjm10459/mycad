import logging
import numpy as np
from pydantic import BaseModel, field_validator, ValidationError
from typing import List, Optional, Tuple, Any, Dict
from decimal import Decimal, ROUND_HALF_UP


class PolylineSegment(BaseModel):
    """多线段中的单个段数据"""
    start_point: List[float]
    end_point: List[float]
    is_arc: bool = False
    center: Optional[List[float]] = None
    radius: float = None
    bulge: float = None

    @field_validator('start_point')
    def validate_start_point(cls, v):
        return [float(Decimal(str(num)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)) for num in v]

    @field_validator('end_point')
    def validate_end_point(cls, v):
        return [float(Decimal(str(num)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)) for num in v]

    @field_validator('center')
    def validate_center(cls, v):
        return [float(Decimal(str(num)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)) for num in v]

    @field_validator('bulge')
    def validate_bulge(cls, v):
        return float(Decimal(str(v)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP))

    @field_validator('radius')
    def validate_radius(cls, v):
        return float(Decimal(str(v)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP))


class CADEntities(BaseModel):
    handle: str
    layer: str
    color: int
    entity_type: Optional[str] = None
    points: List[PolylineSegment]
    child_entities: List = []


class LineWeight(BaseModel):
    lineweight: Optional[int]

    @field_validator('lineweight')
    def validate_lineweight(cls, value):
        if value not in [0, 5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60, 70, 80, 90, 100, 106, 120, 140, 158, 200, 211]:
            logging.warning(f'Lineweight {value} is not a valid lineweight')
            return 0
        return value
