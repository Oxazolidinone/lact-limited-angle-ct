"""FBP backprojection with a prebuilt sparse matrix (``FanBeamFBP(sparse=True)``) versus the per-view-chunk path."""
def test_sparse_backprojection_matches_chunked():
    """The prebuilt sparse backprojection must match the per-view-chunk path (only float32 summation order differs),
    for the forward pass, the adjoint and the gradient through ``apply``; covers a limited range, a full 360° scan
    (factor 0.5) and a short scan (Parker)."""
    import torch

    from lact.core.imaging.geometry import FanBeamGeometry
    from lact.core.operators.fbp import FanBeamFBP

    torch.manual_seed(0)
    for span, center in ((90.0, 224.95), (360.0, 359.95), (240.0, 224.95)):
        g = FanBeamGeometry(name="t", span_deg=span, step_deg=1.0, center_deg=center, n_det=64, det_spacing_mm=2.4,
                            dso_mm=431.0, dod_mm=98.0, fov_mm=116.5)
        old, new = FanBeamFBP(g, 48, sparse=False), FanBeamFBP(g, 48, sparse=True)
        y = torch.rand(2, 1, g.n_views, g.n_det)
        a, b = old(y), new(y)
        assert float((a - b).abs().max() / a.abs().max()) < 1e-5
        x = torch.randn(2, 1, 48, 48)
        assert float((old.backproject_t(x) - new.backproject_t(x)).abs().max()
                     / old.backproject_t(x).abs().max()) < 1e-5
        grads = []
        for f in (old, new):
            yy = y.clone().requires_grad_(True)
            (f.apply(yy) * x).sum().backward()
            grads.append(yy.grad)
        assert float((grads[0] - grads[1]).abs().max() / grads[0].abs().max()) < 1e-5
