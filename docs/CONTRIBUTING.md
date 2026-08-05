# Contributing
We welcome contributions! Please open a GitHub issue or submit a pull request. For major changes, please open an issue first to discuss what you would like to change then submit a pull request.

Below are some guidelines for contributing to the project.

## Adding a new selector type (e.g. Line)

1. Add `point` dataclass to `core/point.py` with properties and persistence methods.
2. Add `select_point()` to `BaseBackend` and implement it in `CV2Backend`.
3. Create `selectors/point_picker.py` with `PointSelector`.
4. Add `pixpick.point()` to `__init__.py`.

No other files change.

## Adding a new properties or persistence methods

Add a property directly to the relevant file in `core/<class>.py`, for example to add property in class box make changes to `core/box.py` 

```python
# eg. in Box
@property
def xyxy(self) -> list[int]:
    """[x1, y1, x2, y2] — absolute pixels for each box in boxes."""
    return self.boxes

```

That's it.
