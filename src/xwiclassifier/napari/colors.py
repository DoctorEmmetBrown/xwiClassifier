"""Palette des classes (valeur de label -> couleur RGBA 0..1)."""
from __future__ import annotations

BACKGROUND_LABEL = 1
_PALETTE = [
    (0.60, 0.60, 0.60), (0.90, 0.10, 0.10), (0.10, 0.60, 0.90), (0.20, 0.80, 0.20),
    (0.95, 0.75, 0.10), (0.70, 0.20, 0.80), (1.00, 0.50, 0.00), (0.00, 0.80, 0.80),
    (0.95, 0.40, 0.70), (0.55, 0.35, 0.15),
]


def default_color(value: int) -> tuple:
    r, g, b = _PALETTE[(int(value) - 1) % len(_PALETTE)]
    return (r, g, b, 1.0)


def to_rgba01(c) -> tuple:
    c = tuple(float(x) for x in c)
    if max(c) > 1.0:
        c = tuple(x / 255.0 for x in c)
    return c + (1.0,) if len(c) == 3 else c


def palette(values, overrides: dict | None = None) -> dict:
    overrides = overrides or {}
    return {int(v): to_rgba01(overrides[v]) if v in overrides else default_color(v) for v in values}


def apply_to_layer(layer, colors: dict) -> None:
    """Applique des couleurs fixes a un layer Labels (napari >= 0.5 puis < 0.5)."""
    try:
        from napari.utils.colormaps import DirectLabelColormap

        cd = {int(k): list(v) for k, v in colors.items()}
        cd[0] = [0.0, 0.0, 0.0, 0.0]
        cd[None] = [0.0, 0.0, 0.0, 0.0]
        layer.colormap = DirectLabelColormap(color_dict=cd)
        return
    except Exception:
        pass
    try:
        layer.color_mode = "direct"
        layer.color = colors
    except Exception:
        pass
