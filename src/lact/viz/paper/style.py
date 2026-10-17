"""Paper / thesis figure style: journal sizes, embedded fonts, English–Vietnamese labels, fixed method colours.

    from lact.viz import paper as pp
    pp.use("ieee", lang="en")              # or "elsevier", "lncs", "thesis"; lang "vi"
    fig = pp.figure("full", 2.4)            # two columns wide, 2.4 inches tall
    pp.save(fig, out_dir, "fig3_qual")     # -> fig3_qual.pdf (vector, embedded fonts) + .png (300 dpi) + .svg

Rules (identical for every figure):
  * The size equals the column width of the venue => inserted into LaTeX/Word without scaling, text = caption size.
  * White background, no in-figure title (the caption does that), panel labels (a), (b) at the top left.
  * Colour + marker are FIXED per method (never by rank); the palette was checked for colour blindness on white
    (validate_palette.js: PASS; yellow/teal < 3:1 => always a distinct marker + direct label or legend).
  * Non-learned baselines (FBP, TV) use neutral grey.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.transforms import Bbox

from ..style import ARCH_TO_REG, METHOD_TABLE

FONT_DIR = Path(__file__).resolve().parents[1] / "fonts"


# ---------------------------------------------------------------------------- figure sizes
@dataclass(frozen=True)
class Preset:
    name: str
    col: float      # inches: one column
    full: float     # inches: two columns / full text width
    font: float     # pt: main text (axis labels)
    small: float    # pt: tick numbers, legends, small annotations
    panel: str      # panel label pattern
    max_h: float    # inches: maximum figure height


PRESETS = {
    "ieee": Preset("ieee", 3.5, 7.16, 8.0, 7.0, "({})", 8.5),          # IEEE TMI/TCI/TIP
    "elsevier": Preset("elsevier", 3.54, 7.48, 8.0, 7.0, "({})", 9.0),  # MedIA, PMB (90 / 190 mm)
    "lncs": Preset("lncs", 4.8, 4.8, 8.0, 7.0, "({})", 7.4),            # MICCAI (122 mm)
    "thesis": Preset("thesis", 3.9, 6.3, 10.0, 9.0, "({})", 9.0),       # A4 thesis ("col" figures centred)
}


@dataclass
class _Ctx:
    preset: Preset = PRESETS["ieee"]
    lang: str = "en"


CTX = _Ctx()

# ---------------------------------------------------------------------------- colours
INK = "#1a1a1a"
INK_2 = "#4d4d4d"
MUTED = "#8c8c8c"
HAIR = "#d9d9d9"
GRID = "#ececec"
WHITE = "#ffffff"

BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
ROI_COLOR = YELLOW

# method key -> (colour, marker, EN label, VI label)
METHODS: dict[str, tuple[str, str, str, str]] = {
    "gt": (INK, "", "Reference", "Tham chiếu"),
    "fbp": ("#9a9a9a", "v", "FBP", "FBP"),
    "tv": ("#5e5e5e", "P", "TV", "TV"),
    "learn_resnet_pm2m": (ORANGE, "s", "LEARN-ResNet (2M)", "LEARN-ResNet (2M)"),   # E4: matched at ~2M parameters
    "learn_regnet_pm2m": (VIOLET, "*", "LEARN-RegNet (2M)", "LEARN-RegNet (2M)"),
    "learn_vit_pm2m": (YELLOW, "^", "LEARN-ViT (2M)", "LEARN-ViT (2M)"),
}
# learned models: colour by architecture family, star = RegNet layout (same table as lact.viz.style)
METHODS.update({r: (c, m, f"LEARN-{s.replace('–', '-')}", f"LEARN-{s.replace('–', '-')}")
                for r, _, _, _, s, c, m in METHOD_TABLE})
ALIASES = {**ARCH_TO_REG, "resnet_pm2m": "learn_resnet_pm2m", "regnet_pm2m": "learn_regnet_pm2m",
           "vit_pm2m": "learn_vit_pm2m"}

# sequential single-hue band (blue) for magnitudes: white (= 0) -> dark blue
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("paper_seq", SEQ)
# diverging blue <-> red with a neutral grey middle (signed errors)
CMAP_DIV = LinearSegmentedColormap.from_list("paper_div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32b2a"])


def key_of(method: str) -> str:
    return ALIASES.get(method, method)


def color(method: str) -> str:
    return METHODS.get(key_of(method), (INK_2,))[0]


def marker(method: str) -> str:
    return METHODS.get(key_of(method), (INK_2, "o"))[1] or "o"


def label(method: str) -> str:
    k = key_of(method)
    if k not in METHODS:
        return method
    _, _, en, vi = METHODS[k]
    return en if CTX.lang == "en" else vi


def tint(c: str, amount: float) -> tuple[float, float, float]:
    """Blend a colour with white (amount=0 keeps it, 1 = white)."""
    r, g, b = to_rgb(c)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


# ---------------------------------------------------------------------------- text (English / Vietnamese)
TEXT: dict[str, tuple[str, str]] = {
    "psnr": ("PSNR (dB)", "PSNR (dB)"),
    "ssim": ("SSIM", "SSIM"),
    "rmse_hu": ("RMSE (HU)", "RMSE (HU)"),
    "psnr_body": ("PSNR, body (dB)", "PSNR vùng cơ thể (dB)"),
    "angle": ("Angular range", "Khoảng góc quét"),
    "epoch": ("Epoch", "Epoch"),
    "val_psnr": ("Validation PSNR (dB)", "PSNR tập xác thực (dB)"),
    "delta": ("Δ{m} vs. {ref} (dB)", "Δ{m} so với {ref} (dB)"),
    "params": ("Parameters (millions)", "Số tham số (triệu)"),
    "time": ("Inference time (ms / slice)", "Thời gian suy luận (ms / lát)"),
    "stage": ("Stage $k$", "Tầng $k$"),
    "abs_err": ("|Error| (HU)", "|Sai số| (HU)"),
    "visible": ("visible", "nhìn thấy được"),
    "invisible": ("invisible", "không nhìn thấy"),
    "mse_split": ("Mean squared error (×10$^{-3}$)", "Sai số bình phương trung bình (×10$^{-3}$)"),
    "inv_share": ("{p:.0f}% invisible", "{p:.0f}% không thấy"),
    "vis_frac": ("Fraction of visible directions", "Tỉ lệ hướng nhìn thấy"),
    "glyph": ("recoverable edge orientations", "hướng biên khôi phục được"),
    "err_share": ("Share of squared error (%)", "Tỉ lệ sai số bình phương (%)"),
    "source": ("source arc", "cung nguồn"),
    "lr": ("lr", "lr"),
    "ref_short": ("Ref.", "Tham chiếu"),
    "seed_mean": ("mean of seeds", "trung bình các seed"),
    "slices": ("slices", "lát cắt"),
    "ci": ("mean ± 95% CI", "trung bình ± KTC 95%"),
    "ns": ("n.s.", "k.ý.n"),
}


def t(key: str, **kw) -> str:
    en, vi = TEXT.get(key, (key, key))
    s = en if CTX.lang == "en" else vi
    return s.format(**kw) if kw else s


def geom_label(name: str) -> str:
    """fan_a120 -> 120°."""
    for tok in str(name).split("_"):
        if tok.startswith("a") and tok[1:].isdigit():
            return f"{tok[1:]}°"
    return str(name)


def geom_angle(name: str) -> float:
    return float(geom_label(name).rstrip("°"))


# ---------------------------------------------------------------------------- rcParams
def _register_fonts() -> None:
    have = {f.fname for f in font_manager.fontManager.ttflist}
    for f in sorted(FONT_DIR.glob("*.ttf")):
        if str(f) not in have:
            font_manager.fontManager.addfont(str(f))


def use(preset: str = "ieee", lang: str = "en", family: str = "sans") -> Preset:
    """Enable the figure style for the session. family: 'sans' (Arial/Helvetica, most journals' default) or 'serif'."""
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}")
    if lang not in ("en", "vi"):
        raise ValueError("lang must be 'en' or 'vi'")
    _register_fonts()
    p = PRESETS[preset]
    CTX.preset, CTX.lang = p, lang
    face = "Liberation Sans" if family == "sans" else "Liberation Serif"
    mpl.rcParams.update({
        "font.family": "sans-serif" if family == "sans" else "serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.serif": ["Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "custom", "mathtext.rm": face, "mathtext.it": f"{face}:italic",
        "mathtext.bf": f"{face}:bold", "mathtext.sf": face, "mathtext.cal": face, "mathtext.tt": "DejaVu Sans Mono",
        "mathtext.fallback": "stixsans" if family == "sans" else "stix",
        "font.size": p.font, "axes.labelsize": p.font, "axes.titlesize": p.font,
        "xtick.labelsize": p.small, "ytick.labelsize": p.small, "legend.fontsize": p.small,
        "legend.title_fontsize": p.small,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#262626", "axes.linewidth": 0.6,
        "xtick.color": "#262626", "ytick.color": "#262626", "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.8, "ytick.major.size": 2.8, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.5, "axes.titlepad": 3.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "lines.linewidth": 1.1, "lines.markersize": 4.0, "lines.markeredgewidth": 0.6,
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
        "patch.linewidth": 0.5, "hatch.linewidth": 0.5, "errorbar.capsize": 0.0,
        "legend.frameon": False, "legend.handlelength": 1.5, "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.2, "legend.columnspacing": 0.9, "legend.labelspacing": 0.3,
        "figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "pdf.compression": 9,
        "image.cmap": "gray", "axes.unicode_minus": True, "axes.formatter.use_mathtext": True,
        "figure.constrained_layout.use": False,
    })
    return p


def width(w: str | float) -> float:
    if isinstance(w, str):
        return {"col": CTX.preset.col, "full": CTX.preset.full}[w]
    return float(w)


def figure(w: str | float = "col", h: float | None = None, aspect: float = 0.72):
    """Scale a given height with the preset font size (10 pt thesis text needs taller panels than 8 pt papers)."""
    W = width(w)
    scale = max(1.0, CTX.preset.font / 8.0)
    H = min(h * scale if h is not None else W * aspect, CTX.preset.max_h)
    return plt.figure(figsize=(W, H))


def save(fig, out_dir: str | Path, name: str, formats: Iterable[str] = ("pdf", "png"),
         png_dpi: int = 300) -> list[Path]:
    """PDF/SVG: vector with embedded fonts (Type 42), for LaTeX. PNG at 300 dpi, for Word/slides.

    The file width is EXACTLY the chosen width (column / two columns) so it is inserted without scaling: cropped
    tightly vertically, horizontally centred within the figure width. Content wider than that triggers a warning."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    W = fig.get_figwidth()
    pad = 0.03
    if tb.width > W + 0.03:
        warnings.warn(f"{name}: content is {tb.width:.2f} in wide > {W:.2f} in (it will be shrunk when inserted)",
                      stacklevel=2)
        box = Bbox([[tb.x0 - pad, tb.y0 - pad], [tb.x1 + pad, tb.y1 + pad]])
    else:
        x0 = min(max(tb.x0 - (W - tb.width) / 2, tb.x1 - W), tb.x0)
        box = Bbox([[x0, tb.y0 - pad], [x0 + W, tb.y1 + pad]])
    paths = []
    for ext in formats:
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=png_dpi if ext == "png" else mpl.rcParams["savefig.dpi"], bbox_inches=box,
                    metadata={"Creator": "lact.viz.paper"} if ext == "pdf" else None)
        paths.append(p)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------- layout helpers
def panel_label(ax, i: int | str, x: float = -0.02, y: float = 1.0, fig_coords: bool = False) -> None:
    s = CTX.preset.panel.format(chr(ord("a") + i) if isinstance(i, int) else i)
    kw = dict(fontsize=CTX.preset.font + 1, fontweight="bold", va="bottom", ha="right", color=INK)
    if fig_coords:
        ax.figure.text(x, y, s, **kw)
    else:
        ax.text(x, y, s, transform=ax.transAxes, **kw)


def panel_title(ax, i: int | None, text: str = "", loc: str = "left") -> None:
    """Panel title "(a) text": bold panel label + regular text, left-aligned; never overlaps the neighbouring title."""
    lab = CTX.preset.panel.format(chr(ord("a") + i)) if i is not None else ""
    head = rf"$\mathbf{{{lab}}}$" if lab else ""
    ax.set_title(f"{head}  {text}".strip(), loc=loc, fontsize=CTX.preset.font, pad=3.0)


def ygrid(ax, minor: bool = False) -> None:
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.5)
    if minor:
        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", bottom=False)


def stroke(lw: float = 1.6, fg: str = "black") -> list:
    """Text outline that keeps labels readable on CT images."""
    return [pe.withStroke(linewidth=lw, foreground=fg)]


def spread(values: list[float], gap: float) -> list[float]:
    """Spread direct-label positions so they do not overlap (order preserved)."""
    order = np.argsort(values)
    out = np.array(values, float)
    for _ in range(50):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            if out[b] - out[a] < gap:
                mid = (out[a] + out[b]) / 2
                out[a], out[b] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    return out.tolist()


def despine_image(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
