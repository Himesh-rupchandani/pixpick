from __future__ import annotations
import cv2
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class Point:
    """
    Immutable result of a point selection.

    Attributes
    ----------
    points : list[tuple[int, int]]
        Ordered list of (x, y) vertices in absolute pixels.
    image_width, image_height : int
        Dimensions of the source image — needed for normalisation.
    """

    points: list[tuple[int, int]]
    image_width: int
    image_height: int

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self):
        if len(self.points) < 1:
            raise ValueError(
                f"Point needs at least 1 point, got {len(self.points)}"
            )
        for i, pt in enumerate(self.points):
            x, y = pt
            if not (0 <= x <= self.image_width and 0 <= y <= self.image_height):
                raise ValueError(
                    f"Point {i} ({x},{y}) is outside image "
                    f"({self.image_width}x{self.image_height})"
                )


    # ------------------------------------------------------------------ #
    # Core format properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def as_numpy(self) -> np.ndarray:
        """Shape (2, 2) int32 array — [[x1,y1], [x2,y2]]."""
        return np.array(self.points, dtype=np.int32)
