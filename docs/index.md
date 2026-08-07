# PixPick

**Interactive coordinate picker for Computer Vision.**

Draw boxes, polygons and lines on images or videos and instantly get coordinates for YOLO, SAM, YOLOE, OpenCV, and your own pipelines.


![Project Overview](./pixpick_main.png)

---
## Why PixPick?

Most computer vision frameworks require coordinates before inference.

Traditionally you have to:

1. Open CVAT or Roboflow
2. Draw a region
3. Copy the coordinates
4. Paste them back into your code

PixPick lets you draw directly from Python and immediately returns framework-ready coordinates.


```python
import pixpick

region = pixpick.box("video.mp4", frame=10)  # drag a box on a specific video frame
zone   = pixpick.polygon("image.jpg")        # click polygon vertices
```

---

## Install

```bash
pip install pixpick
```

---

## Selectors

| Selector | How to use | Returns |
|---|---|---|
| `pixpick.box()` | Left-click + drag | `Box` |
| `pixpick.polygon()` | Click vertices | `Polygon` |
| `pixpick.line()` | Click start → click end | `Line` |

**Box controls** — `drag` to draw · `R` to reset · `Enter` to confirm · `Esc` to cancel

**Polygon controls** — `LMB` add point · `RMB` undo · `Z` clear · `Enter` confirm · `Esc` cancel

---

## Output formats

Every selection object carries all the formats you'll ever need.

```python
# ── Box ──────────────────────────────────────────────────────
region = pixpick.box("frame.jpg")

region.yolo_region       # coordinates in YOLO region format
region.yolo_prompt       # coordinates in YOLOE visual prompt format
region.sam               # coordinates in SAM box prompt format
region.center            # center point of the box (cx, cy)
region.area              # area of the box in pixels²


# ── Polygon ───────────────────────────────────────────────────
zone = pixpick.polygon("frame.jpg")

zone.supervision         # Supervision PolygonZone object
zone.yolo_region         # coordinates in YOLO region format
zone.bbox                # tight bbox around the polygon
zone.npoints             # int
zone.norm                # normalized coordinates [(x0n,y0n), ...]  0.0 – 1.0


# ── Line ─────────────────────────────────────────────────────
line = pixpick.line("frame.jpg")

line.center               # center point of the line (cx, cy)
line.length               # length of the line
line.start                # start point of the line (cx, cy)
line.end                  # end point of the line (cx, cy)
line.vertical             # True if vertical
line.horizontal           # True if horizontal
```
For more details, see [Selectors](selectors.md).

---

## Framework integration

| Framework | Selector | Method |
|---|---|---|
| Ultralytics YOLOE — visual prompt | `Box` | `region.yolo_prompt` |
| Ultralytics YOLO — region | `Box`/`Polygon` | `region.yolo_region` |
| SAM / SAM2 / SAM3 — box prompt | `Box` | `region.sam` |
|| Supervision PolygonZone — polygon | `Polygon` | `region.supervision` |
| Any other format | `Box` / `Polygon` | `region.raw` |