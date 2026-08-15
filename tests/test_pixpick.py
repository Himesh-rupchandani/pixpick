"""
pixpick test suite
------------------
Tests everything that doesn't require a display (no cv2.imshow).
The interactive selectors (BoxSelector, PolygonSelector) are excluded
because CI environments have no display — that logic is covered by the
backend itself which is independently testable.

Run locally:
    pip install pytest
    pytest tests/ -v
"""

import json
import os
import tempfile

import numpy as np
import pytest

from pixpick.core.box import Box, Multibox
from pixpick.core.polygon import Polygon
from pixpick.core.line import Line
from pixpick import load


# ======================================================================== #
# Fixtures                                                                   #
# ======================================================================== #

@pytest.fixture
def make_box():
    def _make_box(**kwargs):
        defaults = dict(
            x1=100,
            y1=50,
            x2=400,
            y2=300,
            image_width=1920,
            image_height=1080,
        )
        defaults.update(kwargs)
        return Box(**defaults)
    return _make_box

@pytest.fixture
def make_multibox():
    def _make_multibox(**kwargs):
        defaults = dict(
            boxes=[
                [100, 50, 400, 300],
                [500, 200, 800, 600]
            ],
            image_width=1920,
            image_height=1080,
        )
        defaults.update(kwargs)
        return Multibox(**defaults)
    return _make_multibox

@pytest.fixture
def make_polygon():
    def _make_polygon(**kwargs):
        defaults = dict(
            points=[(100, 50), (400, 50), (400, 300), (100, 300)],
            image_width=1920,
            image_height=1080,
        )
        defaults.update(kwargs)
        return Polygon(**defaults)
    return _make_polygon

@pytest.fixture
def line():
    return Line(
        points=[(100, 50), (400, 300)],
        image_width=1920, image_height=1080
    )

@pytest.fixture
def sample_image():
    """Synthetic BGR image — no file needed."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# ======================================================================== #
# Box — construction and validation                                          #
# ======================================================================== #

class TestBoxConstruction:

    def test_basic(self, make_box):
        box = make_box()
        assert box.x1 == 100
        assert box.y1 == 50
        assert box.x2 == 400
        assert box.y2 == 300

    def test_auto_sort_coords(self, make_box):
        """Drag direction shouldn't matter — x1 < x2 and y1 < y2 always."""
        box = make_box()
        assert box.x1 == 100 and box.x2 == 400
        assert box.y1 == 50  and box.y2 == 300

    def test_zero_area_raises(self, make_box):
        with pytest.raises(ValueError, match="zero area"):
            make_box(x2=100)

    def test_x_out_of_bounds_raises(self, make_box):
        with pytest.raises(ValueError, match="x coords"):
            make_box(x1=-1)

    def test_y_out_of_bounds_raises(self, make_box):
        with pytest.raises(ValueError, match="y coords"):
            make_box(y2=-1)


# ======================================================================== #
# Box — coordinate properties                                               #
# ======================================================================== #

class TestBoxProperties:

    def test_xyxy(self, make_box):
        box = make_box()
        assert box.xyxy == [100, 50, 400, 300]

    def test_xywh(self, make_box):
        box = make_box()
        assert box.xywh == [100, 50, 300, 250]

    def test_cxcywh(self, make_box):
        box = make_box()
        cx, cy, w, h = box.cxcywh
        assert cx == 250.0
        assert cy == 175.0
        assert w  == 300.0
        assert h  == 250.0

    def test_norm(self, make_box):
        box = make_box()
        n = box.norm
        assert len(n) == 4
        assert all(0.0 <= v <= 1.0 for v in n)
        assert pytest.approx(n[0], abs=1e-4) == 100 / 1920
        assert pytest.approx(n[1], abs=1e-4) == 50  / 1080

    def test_norm_xywh(self, make_box):
        box = make_box()
        n = box.norm_xywh
        assert len(n) == 4
        assert all(0.0 <= v <= 1.0 for v in n)

    def test_center(self, make_box):
        box = make_box()
        assert box.center == (250, 175)

    def test_area(self, make_box):
        box = make_box()
        assert box.area == 300 * 250

    def test_as_numpy_shape_and_dtype(self, make_box):
        box = make_box()
        arr = box.as_numpy
        assert arr.shape == (4,)
        assert arr.dtype == np.int32
        assert arr.tolist() == [100, 50, 400, 300]

    def test_yolo_region(self, make_box):
        box = make_box()
        assert box.yolo_region == [
            (100, 50),
            (400, 50),
            (400, 300),
            (100, 300),
        ]

    def test_yolo_prompt(self, make_box):
        box = make_box()
        np.testing.assert_array_equal(
            box.yolo_prompt,
            np.array([[100, 50, 400, 300]]),
        )

    def test_sam(self, make_box):
        box = make_box()
        np.testing.assert_array_equal(box.sam, np.array([100, 50, 400, 300]))

    def test_raw_keys(self, make_box):
        box = make_box()
        raw = box.raw
        expected = {"xyxy", "xywh", "cxcywh", "normalized", "normalized_xywh", "numpy"}
        assert expected.issubset(raw.keys())

    def test_raw_xyxy_matches(self, make_box):
        box = make_box()
        assert box.raw["xyxy"] == box.xyxy


# ======================================================================== #
# Box — persistence                                                         #
# ======================================================================== #

class TestBoxPersistence:

    def test_save_creates_file(self, make_box):
        box = make_box()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            box.save(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_save_json_schema(self, make_box):
        box = make_box()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            box.save(path)
            data = json.loads(open(path).read())
            assert data["type"] == "box"
            assert "image_size" in data
            assert "coordinates" in data
            assert "xyxy" in data["coordinates"]
        finally:
            os.unlink(path)

    def test_round_trip(self, make_box):
        box = make_box()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            box.save(path)
            reloaded = Box.load(path)
            assert reloaded.xyxy == box.xyxy
            assert reloaded.image_width  == box.image_width
            assert reloaded.image_height == box.image_height
        finally:
            os.unlink(path)

    def test_load_wrong_type_raises(self, make_polygon):
        polygon = make_polygon()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            polygon.save(path)
            with pytest.raises(ValueError, match="Expected type 'box'"):
                Box.load(path)
        finally:
            os.unlink(path)


# ======================================================================== #
# Box — visualize                                                           #
# ======================================================================== #

class TestBoxVisualize:

    def test_returns_same_shape(self, make_box, sample_image):
        box = make_box()
        vis = box.visualize(sample_image)
        assert vis.shape == sample_image.shape

    def test_does_not_mutate_original(self, make_box, sample_image):
        box = make_box()
        original = sample_image.copy()
        box.visualize(sample_image)
        np.testing.assert_array_equal(sample_image, original)



# ======================================================================== #
# Multibox — construction and properties                                    #
# ======================================================================== #

class TestMultiboxConstruction:

    def test_basic(self, make_multibox):
        multibox = make_multibox()

        assert multibox.xyxy == [
            [100, 50, 400, 300],
            [500, 200, 800, 600],
        ]

    def test_empty_boxes_raises(self, make_multibox):
            with pytest.raises(ValueError, match="Multibox must contain at least two Box."):
                make_multibox(boxes=[])
                
    def test_invalid_box_raises(self, make_multibox):
        with pytest.raises(ValueError):
            make_multibox(boxes=[[100, 50, 100, 300]])


# ======================================================================== #
# Multibox — coordinate properties                                               #
# ======================================================================== #

class TestMultiboxProperties:

    def test_xyxy(self, make_multibox):
        multibox = make_multibox()
        assert multibox.xyxy == [
            [100, 50, 400, 300],
            [500, 200, 800, 600],
        ]

    def test_xywh(self, make_multibox):
        multibox = make_multibox()
        assert multibox.xywh == [
            [100, 50, 300, 250],
            [500, 200, 300, 400],
        ]

    def test_cxcywh(self, make_multibox):
        multibox = make_multibox()

        assert multibox.cxcywh == [
            [250.0, 175.0, 300.0, 250.0],
            [650.0, 400.0, 300.0, 400.0],
        ]

    def test_norm(self, make_multibox):
        multibox = make_multibox()

        norm = multibox.norm
        assert len(norm) == 2
        assert all(all(0 <= v <= 1 for v in box) for box in norm)

    def test_norm_xywh(self, make_multibox):
        multibox = make_multibox()

        norm = multibox.norm_xywh
        assert len(norm) == 2

    def test_center(self, make_multibox):
        multibox = make_multibox()
        assert multibox.center == [
            [250, 175],
            [650, 400],
        ]

    def test_area(self, make_multibox):
        multibox = make_multibox()
        assert multibox.area == [
            300 * 250,
            300 * 400,
        ]

    def test_as_numpy(self, make_multibox):
        multibox = make_multibox()

        arr = multibox.as_numpy
        assert arr.shape == (2, 4)
        assert arr.dtype == np.int32

    def test_yolo_region(self, make_multibox):
        multibox = make_multibox()
        assert len(multibox.yolo_region) == 2

    def test_yolo_prompt(self, make_multibox):
        multibox = make_multibox()
        np.testing.assert_array_equal(
            multibox.yolo_prompt,
            np.array([
                [100, 50, 400, 300],
                [500, 200, 800, 600],
            ]),
        )

    def test_sam(self, make_multibox):
        multibox = make_multibox()

        np.testing.assert_array_equal(
            multibox.sam,
            np.array([
                [100, 50, 400, 300],
                [500, 200, 800, 600],
            ]),
        )

    def test_raw_keys(self, make_multibox):
        multibox = make_multibox()

        raw = multibox.raw
        expected = {
            "xyxy",
            "xywh",
            "cxcywh",
            "normalized",
            "normalized_xywh",
            "numpy",
        }

        assert expected.issubset(raw.keys())

# ======================================================================== #
# Multibox — persistence                                                   #
# ======================================================================== #

class TestMultiboxPersistence:

    def test_round_trip(self, make_multibox):
        multibox = make_multibox()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            multibox.save(path)

            reloaded = Multibox.load(path)

            assert reloaded.xyxy == multibox.xyxy
            assert reloaded.image_width == multibox.image_width
            assert reloaded.image_height == multibox.image_height
        finally:
            os.unlink(path)


# ======================================================================== #
# Multibox — visualize                                                     #
# ======================================================================== #

class TestMultiboxVisualize:

    def test_returns_same_shape(self, make_multibox, sample_image):
        multibox = make_multibox()
        vis = multibox.visualize(sample_image)
        assert vis.shape == sample_image.shape

    def test_does_not_mutate_original(self, make_multibox, sample_image):
        multibox = make_multibox()
        original = sample_image.copy()
        multibox.visualize(sample_image)
        np.testing.assert_array_equal(sample_image, original)



# ======================================================================== #
# Polygon — construction and validation                                     #
# ======================================================================== #

class TestPolygonConstruction:

    def test_basic(self, make_polygon):
        polygon = make_polygon()
        assert polygon.npoints == 4

    def test_too_few_points_raises(self, make_polygon):
        with pytest.raises(ValueError, match="at least 3"):
            make_polygon(
                points=[(0, 0), (100, 100)],
                image_width=1920,
                image_height=1080,)

    def test_point_out_of_bounds_raises(self, make_polygon):
        with pytest.raises(ValueError, match="outside image"):
            make_polygon(
                points=[(0, 0), (100, 100), (2000, 500)],
                image_width=1920,
                image_height=1080,
            )


# ======================================================================== #
# Polygon — properties                                                      #
# ======================================================================== #

class TestPolygonProperties:

    def test_as_numpy_shape(self, make_polygon):
        polygon = make_polygon()
        arr = polygon.as_numpy
        assert arr.shape == (4, 2)
        assert arr.dtype == np.int32

    def test_norm_range(self, make_polygon):
        polygon = make_polygon()
        for x, y in polygon.norm:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_norm_numpy_shape(self, make_polygon):
        polygon = make_polygon()
        arr = polygon.norm_numpy
        assert arr.shape == (4, 2)
        assert arr.dtype == np.float32

    def test_bbox_type(self, make_polygon):
        polygon = make_polygon()
        bbox = polygon.bbox
        assert isinstance(bbox, list)

    def test_bbox_values(self, make_polygon):
        polygon = make_polygon()
        # polygon is a rectangle (100,50)→(400,300)
        bbox = polygon.bbox
        assert bbox == [100, 50, 400, 300]

    def test_n_points(self, make_polygon):
        polygon = make_polygon()
        assert polygon.npoints == 4

    def test_to_supervision_key(self, make_polygon):
        polygon = make_polygon()
        result = polygon.supervision
        assert "polygon" in result

    def test_to_supervision_numpy(self, make_polygon):
        polygon = make_polygon()
        arr = polygon.supervision["polygon"]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4, 2)

    def test_raw_keys(self, make_polygon):
        polygon = make_polygon()
        raw = polygon.raw
        expected = {"points", "numpy", "normalized", "normalized_numpy", "bbox_xyxy"}
        assert expected.issubset(raw.keys())


# ======================================================================== #
# Polygon — persistence                                                     #
# ======================================================================== #

class TestPolygonPersistence:

    def test_round_trip(self, make_polygon):
        polygon = make_polygon()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            polygon.save(path)
            reloaded = Polygon.load(path)
            assert reloaded.points == polygon.points
            assert reloaded.image_width  == polygon.image_width
            assert reloaded.image_height == polygon.image_height
        finally:
            os.unlink(path)

    def test_save_json_schema(self, make_polygon):
        polygon = make_polygon()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            polygon.save(path)
            data = json.loads(open(path).read())
            assert data["type"] == "polygon"
            assert "image_size" in data
            assert "points" in data["coordinates"]
        finally:
            os.unlink(path)

    def test_load_wrong_type_raises(self, make_box):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            box = make_box()
            box.save(path)
            with pytest.raises(ValueError, match="Expected type 'polygon'"):
                Polygon.load(path)
        finally:
            os.unlink(path)


# ======================================================================== #
# Polygon — visualize                                                       #
# ======================================================================== #

class TestPolygonVisualize:

    def test_returns_same_shape(self, make_polygon, sample_image):
        polygon = make_polygon()
        vis = polygon.visualize(sample_image)
        assert vis.shape == sample_image.shape

    def test_does_not_mutate_original(self, make_polygon, sample_image):
        polygon = make_polygon()
        original = sample_image.copy()
        polygon.visualize(sample_image)
        np.testing.assert_array_equal(sample_image, original)


# ======================================================================== #
# pixpick.load() dispatcher                                                 #
# ======================================================================== #

class TestLoadDispatcher:

    def test_dispatches_box(self, make_box):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            box = make_box()
            box.save(path)
            result = load(path)
            assert isinstance(result, Box)
            assert result.xyxy == box.xyxy
        finally:
            os.unlink(path)

    def test_dispatches_multibox(self, make_multibox):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            multibox = make_multibox()
            multibox.save(path)
            result = load(path)
            assert isinstance(result, Multibox)
            assert result.xyxy == multibox.xyxy
        finally:
            os.unlink(path)

    def test_dispatches_polygon(self, make_polygon):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            polygon = make_polygon()
            polygon.save(path)
            result = load(path)
            assert isinstance(result, Polygon)
            assert result.points == polygon.points
        finally:
            os.unlink(path)

    def test_dispatches_line(self, line):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            line.save(path)
            result = load(path)
            assert isinstance(result, Line)
            assert result.center == line.center
        finally:
            os.unlink(path)