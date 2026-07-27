"""
Minimal reader for MATLAB ``.fig`` files (HG2 / MAT v5 serialization).

A ``.fig`` is a MAT-file holding one ``hgS_070000`` struct whose ``children`` form the
handle-graphics tree. Line data hides in ``<obj>.properties.XData/YData``; axis labels and
titles are *not* stored as ``XLabel``/``Title`` properties but appear as ``text`` children of
the axes carrying a ``String`` field -- hence the two separate collectors below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat


@dataclass
class FigLine:
    axes_index: int
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    display_name: str = ""


@dataclass
class FigAxes:
    index: int
    lines: List[FigLine] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)


def read_fig(path: str) -> List[FigAxes]:
    """Return one :class:`FigAxes` per axes object in the figure, in stored order."""
    root = loadmat(path, struct_as_record=False, squeeze_me=True)["hgS_070000"]

    axes_objects = [
        child for child in _as_list(getattr(root, "children", None))
        if getattr(child, "type", "") == "axes"
    ]

    result = []
    for index, axes_object in enumerate(axes_objects):
        axes = FigAxes(index=index)
        _walk(axes_object, axes)
        result.append(axes)
    return result


def _walk(node: Any, axes: FigAxes) -> None:
    properties = getattr(node, "properties", None)
    fields = getattr(properties, "_fieldnames", []) if properties is not None else []

    if "XData" in fields:
        axes.lines.append(FigLine(
            axes_index=axes.index,
            x=_vector(getattr(properties, "XData")),
            y=_vector(getattr(properties, "YData")),
            display_name=str(getattr(properties, "DisplayName", "") or ""),
        ))
    if "String" in fields:
        axes.texts.append(str(getattr(properties, "String")))

    children = getattr(properties, "Children", None) if properties is not None else None
    if children is None:
        children = getattr(node, "children", None)
    for child in _as_list(children):
        if hasattr(child, "_fieldnames") or hasattr(child, "properties"):
            _walk(child, axes)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return [item for item in np.atleast_1d(value).ravel()]


def _vector(value: Any) -> NDArray[np.float64]:
    return np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
