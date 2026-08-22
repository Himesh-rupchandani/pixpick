# Selectors

A selector opens a UI, captures user input, and returns a typed Selection object. The selection object holds all coordinate math and framework conversion methods.

## Box

```python
region = pixpick.box("video.mp4", frame=10)
```

`pixpick.box()` returns a `Box` when you draw one rectangle, and a `Multibox` when you draw several. For multi-box results, use `region.boxes` for the wrapped `Box` objects and `region.xyxy` for all coordinates.

`pixpick.box()` accepts an image path, a video path, or a BGR numpy array. Use `frame=` to choose the video frame.

### Properties

| Property | Description |
|---|---|
| `xyxy` | [x1, y1, x2, y2] absolute pixels |
| `xywh` | [x, y, w, h] absolute pixels |
| `cxcywh` | [cx, cy, w, h] absolute pixels |
| `norm` | [x1, y1, x2, y2] 0.0 – 1.0 |
| `norm_xywh` | [x, y, w, h] 0.0 – 1.0 (YOLO label format) |
| `center` | (cx, cy) absolute pixels |
| `area` | int pixels² |
| `as_numpy` | np.array shape (4,) int32 |


### Framework properties

| Property | Description |
|---|---|
| `yolo_region` | dict with keys: xywh, norm_xywh, class, confidence |
| `yolo_prompt` | str prompt for LLMs |
| `sam` | dict with keys: mask, bbox, area, confidence, point_coords, point_labels |
| `raw` | dict with all formats in one place |

### Visualise

```python
canvas = region.visualize(image)   # returns BGR array with box drawn
cv2.imshow("result", canvas)
cv2.waitKey(0)
```

### Persistence

```python
region.save("selection.json")
region = pixpick.load("selection.json")   # or Box.load("selection.json")
```

### Multi-box results

```python
region = pixpick.box("image.jpg")

region.boxes        # [Box(...), Box(...), ...]
region.xyxy         # [[x1, y1, x2, y2], ...]
region.as_numpy     # np.array shape (N, 4)
```

properties and methods are the same for `Box` and `Multibox`, except that `Multibox` has a `boxes` property.

---

## Polygon

```python
zone = pixpick.polygon("video.mp4", frame=10)
```

Minimum 3 points required before `Enter` confirms. Vertices are recorded in the order you click them.

`pixpick.polygon()` accepts an image path, a video path, or a BGR numpy array. Use `frame=` to choose the video frame.

### Properties

| Property | Description |
|---|---|
| `points` | list of tuples, absolute pixels |
| `as_numpy` | np.array shape (N, 2) int32 |
| `norm` | list of tuples, 0.0 – 1.0 |
| `norm_numpy` | np.array shape (N, 2) float32 |
| `npoints` | int |
| `bbox` | Box — tight axis-aligned bbox around the polygon |
| `image_width` | int |
| `image_height` | int |

### Framework properties

| Property | Description |
|---|---|
| `yolo_region` | dict with keys: xywh, norm_xywh, class, confidence |
| `supervision` | dict with keys: points, polygon, bbox, area, confidence, point_coords, point_labels |
| `raw` | dict with all formats in one place |

### Visualise

```python
canvas = zone.visualize(image)                        # default green, 15% fill
canvas = zone.visualize(image, color=(0,0,255), fill_alpha=0.3)
```

### Persistence

```python
zone.save("zone.json")
zone = pixpick.load("zone.json")   # or Polygon.load("zone.json")
```


### Multi-polygon results

```python
zones = pixpick.polygon("frame.jpg")
zones.polygons        # [Polygon(...), Polygon(...), ...]
zones.points          # [[(x0,y0), ...], [(x0,y0), ...], ...]
zones.as_numpy        # np.array shape (N, 2)      int32
```
`pixpick.polygon()` uses the same interactive flow as the OpenCV polygon backend: draw a polygon, press Space to save it, draw another polygon, and press Enter to finish.

---

## Line

```python
line = pixpick.line("image.jpg")
```

`pixpick.line()` returns a `Line` object when you draw a line by clicking two points.

`pixpick.line()` accepts an image path, a video path, or a BGR numpy array. Use `frame=` to choose the video frame.


### Properties

| Property | Description |
|---|---|
| `start` | (x1, y1) absolute pixels |
| `end` | (x2, y2) absolute pixels |
| `center` | (cx, cy) absolute pixels |
| `length` | length of the line|
| `vertical` | True if vertical |
| `horizontal` | True if horizontal |


### Visualise

```python
canvas = line.visualize(image)                        # default green, 15% fill
canvas = line.visualize(image, color=(0,0,255), thickness=2)
```

### Persistence

```python
line.save("line.json")
line = pixpick.load("line.json")   # or Line.load("line.json")
```

### Multi-line results

```python
lines = pixpick.line("frame.jpg")
lines.start          # [(x1,y1), (x1,y1), ...]
lines.end            # [(x2,y2), (x2,y2), ...]
lines.length         # [length1, length2, ...]
lines.vertical       # [True, False, ...]
```
---


## Coming in future releases

| Selector | Interaction | Returns | Release |
|---|---|---|---|
| `pixpick.points()` | click (fg/bg toggle) | `Points` | v0.3.0 |
| `pixpick.perspective()` | 4-corner click | `Perspective` | v0.3.0 |
