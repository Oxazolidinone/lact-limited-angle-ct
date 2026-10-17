"""Colour palette and matplotlib style shared by every thesis figure.

Categorical colours are assigned to architectures FOR GOOD (never by rank); the slot order of the reference palette
was checked for colour-vision deficiency (validate_palette.js: the first 4 slots PASS; aqua/yellow < 3:1 on light
backgrounds => always paired with direct labels or a legend + table). FBP/TV baselines use neutral ink, no slot.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# One table for every learned model: (regularizer, config name, family, layout, short label, colour, marker).
# Colours follow the architecture FAMILY; the "RegNet layout" variant of each family shares its colour, star marker.
_BLUE, _BROWN, _ORANGE, _VIOLET = "#2a78d6", "#8a5a3c", "#eb6834", "#4a3aa7"
_RED, _MAGENTA, _AQUA, _YELLOW = "#e34948", "#c2459b", "#1baf7a", "#eda100"
METHOD_TABLE = [
    ("cnn3", "learn_cnn", "cnn", "plain", "CNN", _BLUE, "o"),
    ("unet_ed", "learn_unet", "unet", "uniform", "UNet", _BROWN, "h"),
    ("unet_regnet", "learn_unet_regnet", "unet", "regnet", "UNet–RegNet", _BROWN, "*"),
    ("resnet_ed", "learn_resnet", "resnet", "uniform", "ResNet", _ORANGE, "s"),
    ("resnet_regnet", "learn_resnet_regnet", "resnet", "regnet", "ResNet–RegNet", _ORANGE, "*"),
    ("resnext_ed", "learn_resnext", "xblock", "uniform", "ResNeXt", _VIOLET, "X"),
    ("regnet_ed", "learn_regnet", "xblock", "regnet", "RegNet", _VIOLET, "*"),
    ("densenet_ed", "learn_densenet", "densenet", "uniform", "DenseNet", _RED, "p"),
    ("densenet_regnet", "learn_densenet_regnet", "densenet", "regnet", "DenseNet–RegNet", _RED, "*"),
    ("mbconv_ed", "learn_efficientnet", "efficientnet", "uniform", "EfficientNet", _MAGENTA, "d"),
    ("mbconv_regnet", "learn_efficientnet_regnet", "efficientnet", "regnet", "EfficientNet–RegNet", _MAGENTA, "*"),
    ("convnext_ed", "learn_convnext", "convnext", "uniform", "ConvNeXt", _AQUA, "D"),
    ("convnext_regnet", "learn_convnext_regnet", "convnext", "regnet", "ConvNeXt–RegNet", _AQUA, "*"),
    ("vit", "learn_vit", "vit", "uniform", "ViT", _YELLOW, "^"),
    ("vit_regnet", "learn_vit_regnet", "vit", "regnet", "ViT–RegNet", _YELLOW, "*"),
]
ARCH_COLORS = {r: c for r, _, _, _, _, c, _ in METHOD_TABLE}
BASELINE_COLORS = {"fbp": MUTED, "tv": INK_2}
# Learned methods OUTSIDE the LEARN framework (standalone models, not regularizers): name -> (label, colour, marker).
# They come after the LEARN models in every table/figure and stay out of METHOD_TABLE so the E0-E6 comparison pairs
# do not change.
OTHER_METHODS = {"regformer": ("RegFormer", "#0e7c86", "8"),
                 # methods of the comparison table with the CG-GLORE paper (configs/experiment/cmp_*.yaml)
                 "fbpconvnet": ("FBPConvNet", "#8a6d3b", "X"), "lpd": ("Learned Primal-Dual", "#5b6c8f", "h"),
                 "learn_orig": ("LEARN (bản gốc)", "#7a7a7a", "o"), "dpma": ("DPMA", "#4f7f52", "<"),
                 "cg_glore": ("CG-GLORE", "#b0413e", "H")}
ARCH_MARKERS = {**{r: m for r, _, _, _, _, _, m in METHOD_TABLE}, "fbp": "v", "tv": "P",
                **{r: m for r, (_, _, m) in OTHER_METHODS.items()}}
ARCH_SHORT = {r: s for r, _, _, _, s, _, _ in METHOD_TABLE}
ARCH_LABELS = {**{r: f"LEARN–{s}" for r, s in ARCH_SHORT.items()}, "fbp": "FBP", "tv": "TV",
               **{r: lab for r, (lab, _, _) in OTHER_METHODS.items()}}
ARCH_ORDER = ["fbp", "tv", *ARCH_COLORS, *OTHER_METHODS]
ARCH_TO_REG = {a: r for r, a, _, _, _, _, _ in METHOD_TABLE}      # model config name -> regularizer
# (RegNet-layout variant, uniform-layout variant) of each family; backbones added after the first four thesis models
REGNET_TWINS = [(r, next(u for u, _, f2, lay2, *_ in METHOD_TABLE if f2 == fam and lay2 == "uniform"))
                for r, _, fam, lay, *_ in METHOD_TABLE if lay == "regnet"]
ORIGINAL_ARCHS = ("cnn3", "resnet_ed", "convnext_ed", "vit")
NEW_BACKBONES = [r for r, _, _, lay, *_ in METHOD_TABLE
                 if r not in ORIGINAL_ARCHS and (lay == "uniform" or r == "regnet_ed")]

# sequential single-hue band (blue), light -> dark; the lightest step fades into the background = "near 0"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("lact_seq_blue", SEQ_BLUE)
# diverging blue <-> red with a neutral grey midpoint
CMAP_DIV = LinearSegmentedColormap.from_list("lact_div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32b2a"])


def color_of(key: str) -> str:
    if key in OTHER_METHODS:
        return OTHER_METHODS[key][1]
    return ARCH_COLORS.get(key, BASELINE_COLORS.get(key, INK_2))


def label_of(key: str) -> str:
    return ARCH_LABELS.get(key, key)


def geom_label(name: str) -> str:
    """fan_a120 -> 120°."""
    for tok in name.split("_"):
        if tok.startswith("a") and tok[1:].isdigit():
            return f"{tok[1:]}°"
    return name


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 1.5, "lines.markersize": 6,
        "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
