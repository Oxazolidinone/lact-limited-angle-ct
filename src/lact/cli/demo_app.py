"""Gradio UI of the demo application (logic in ``lact.services.demo.DemoBackend``). UI labels are Vietnamese."""
from __future__ import annotations

import numpy as np

from ..viz.style import CMAP_SEQ
from ..viz.windows import WINDOWS, apply_window, to_hu


def _window_u8(x, mu_ref, mu_w, window) -> np.ndarray:
    W, L = WINDOWS[window]
    return (apply_window(to_hu(x, mu_ref, mu_w), W, L) * 255).astype(np.uint8)


def render(r: dict, window: str, err_max: float):
    """uint8 images (sinogram, FBP, reconstruction, reference, error) + the metrics table."""
    s = r["sino"]
    sino = ((s - s.min()) / (np.ptp(s) + 1e-8) * 255).astype(np.uint8)
    err = np.abs(to_hu(r["pred"], r["mu_ref"], r["mu_w"]) - to_hu(r["ref"], r["mu_ref"], r["mu_w"]))
    err_rgb = (CMAP_SEQ(np.clip(err / float(err_max), 0, 1))[..., :3] * 255).astype(np.uint8)
    table = [[k, f"{v:.4f}" if "SSIM" in k or k == "RMSE" else f"{v:.2f}"] for k, v in r["metrics"].items()]
    w = [_window_u8(r[k], r["mu_ref"], r["mu_w"], window) for k in ("fbp", "pred", "ref")]
    return (sino, *w, err_rgb, table)


def build_ui(backend):
    import gradio as gr

    geoms = backend.geometries()
    pids = backend.patients()
    n_max = max((backend.n_slices(p) for p in pids), default=1)

    with gr.Blocks(title="LA-CT deep unrolling") as ui:
        gr.Markdown("### Tái tạo CT góc giới hạn – LEARN với các regularization block")
        with gr.Row():
            geom = gr.Dropdown(geoms, value=geoms[0] if geoms else None, label="Hình học")
            pid = gr.Dropdown(pids, value=pids[0] if pids else None, label="Bệnh nhân")
            idx = gr.Slider(0, n_max - 1, step=1, value=0, label="Lát")
        with gr.Row():
            method = gr.Dropdown([], label="Phương pháp")
            mode = gr.Radio(["cache", "trực tiếp"], value="cache", label="Chế độ")
            window = gr.Radio(list(WINDOWS), value="soft", label="Cửa sổ HU")
            err_max = gr.Slider(50, 500, value=200, step=10, label="Thang sai số (HU)")
        btn = gr.Button("Tái tạo")
        with gr.Row():
            o_sino, o_fbp, o_pred = gr.Image(label="Sinogram"), gr.Image(label="FBP"), gr.Image(label="Ảnh tái tạo")
        with gr.Row():
            o_ref, o_err = gr.Image(label="Ảnh tham chiếu"), gr.Image(label="|Sai số| (HU)")
            o_tab = gr.Dataframe(headers=["Chỉ số", "Giá trị"], label="Chỉ số")

        def upd_methods(g):
            ms = list(backend.methods(g)) if g else []
            return gr.update(choices=ms, value=ms[0] if ms else None)

        def upd_slider(p):
            return gr.update(maximum=backend.n_slices(p) - 1 if p else 0, value=0)

        def run(g, p, i, m, mo, w, e):
            return render(backend.reconstruct(g, p, int(i), m, live=(mo != "cache")), w, e)

        geom.change(upd_methods, geom, method)
        pid.change(upd_slider, pid, idx)
        ui.load(upd_methods, geom, method)
        btn.click(run, [geom, pid, idx, method, mode, window, err_max], [o_sino, o_fbp, o_pred, o_ref, o_err, o_tab])
    return ui
