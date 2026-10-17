"""RegFormer and the differentiable FBP used in its data step, on a small operator."""
import numpy as np
import pytest
import torch

from lact.core.analysis import model_cost
from lact.core.imaging.intensity import block_mean
from lact.core.imaging.phantom import shepp_logan
from lact.core.metrics import psnr
from lact.core.models import RegFormer, Unrolled, build_regformer, count_params
from lact.core.models.baselines.regformer import ConvStage, SwinBlock, SwinStage
from lact.core.operators import FanBeamFBP
from lact.infra.operators.operator_store import load_matrix

N = 32
TINY = {"dim": 16, "window": 8, "depth": 2, "heads": 2}


def _data(op_dir, geom):
    A64, _ = load_matrix(geom, 64, op_dir)
    ph = shepp_logan(64)
    x = torch.from_numpy(block_mean(ph, 2))[None, None]
    y = torch.from_numpy((A64 @ ph.ravel()).astype(np.float32)).view(1, 1, geom.n_views, geom.n_det)
    return y, FanBeamFBP(geom, N)(y), x


def test_lbfgs_data_step_reuses_the_trained_blocks(op_dir, small_geom, small_projector):
    """The L-BFGS variant loads the regularization blocks from an FBP-variant checkpoint; alpha is new (= 1), w is
    gone."""
    y, x0, x = _data(op_dir, small_geom)
    plain = build_regformer(TINY, K=3, projector=small_projector)
    qn = build_regformer(TINY, K=3, projector=small_projector, data_step="lbfgs")
    own = qn.state_dict()
    assert {k for k in own if not k.startswith("stages.")} == {"a_qn"} and "w" in plain.state_dict()
    assert qn.lbfgs_memory == 13 and list(qn.stage_params()) == ["alphas"] and torch.allclose(qn.alphas, torch.ones(3))
    report = qn.load_state_dict({k: v for k, v in plain.state_dict().items() if k in own}, strict=False)
    assert report.missing_keys == ["a_qn"]
    with torch.no_grad():
        xs, terms = qn.eval()(y, x0, return_stages=True, return_terms=True)
        # zero-initialised image output: each stage = ReLU(x - alpha * H g); the first stage has no pairs: a gradient
        # step
        g = small_projector.adjoint(small_projector.forward(x0) - small_projector.normalize_data(y))
        assert torch.allclose(xs[1], torch.relu(x0 - g), atol=1e-6)
        # differs from the FBP variant (w = 0 there)
        assert not torch.allclose(xs[-1], plain.eval()(y, x0), atol=1e-4)
    assert set(terms[0]) == {"k", "dc_norm", "reg_norm", "alpha"} and terms[0]["dc_norm"] > 0
    res = [float(small_projector.data_residual(v, y)) for v in xs]
    assert res[-1] < res[0]
    qn.grad_checkpoint = True
    qn.train()
    torch.mean((qn(y, x0) - x) ** 2).backward()
    assert torch.isfinite(qn.a_qn.grad).all() and float(qn.a_qn.grad.abs().max()) > 0


def test_fbp_adjoint_and_differentiable_apply(small_geom):
    # view count not divisible by the chunk: tests the odd chunk
    fbp = FanBeamFBP(small_geom, N, view_chunk=7)
    gen = torch.Generator().manual_seed(0)
    Q = torch.randn(2, small_geom.n_views, small_geom.n_det, generator=gen)
    G = torch.randn(2, 1, N, N, generator=gen)
    lhs = (fbp.backproject(Q) * G).double().sum()
    rhs = (Q * fbp.backproject_t(G)).double().sum()
    assert float(abs(lhs - rhs) / abs(lhs)) < 1e-5                                   # <B q, g> = <q, B^T g>
    y = torch.randn(2, 1, small_geom.n_views, small_geom.n_det, generator=gen, requires_grad=True)
    lean, plain = fbp.apply(y), fbp.backproject(fbp.filter(y))
    # same computation as the plain FBP
    assert torch.equal(lean, plain) and torch.equal(fbp(y.detach()), plain.detach())
    g_lean, = torch.autograd.grad((lean * G).sum(), y)
    g_plain, = torch.autograd.grad((plain * G).sum(), y)
    assert torch.allclose(g_lean, g_plain, rtol=1e-4, atol=1e-6 * float(g_plain.abs().max()))


def test_default_architecture_matches_the_published_one():
    m = RegFormer()
    assert isinstance(m, Unrolled) and m.K == 10 and not m.uses_cg and m.cg_iters_mean() == 0.0
    assert count_params(m) == 2_678_498 and m.regularizer_params() == 2_678_488     # CG-GLORE paper: 10 stages, 2.7M
    assert [type(s) for s in m.stages] == [ConvStage, SwinStage] * 5                # alternating convolution / Swin
    first, mid, last = m.stages[0], m.stages[2], m.stages[9]
    # later stages concatenate 48 passed channels
    assert first.conv1.out_channels == 96 and mid.conv1.out_channels == 48
    assert m.stages[1].embed.out_channels == 48 and mid.to_feat.out_channels == 48 and last.to_feat is None
    blocks = m.stages[1].blocks
    assert [b.shift for b in blocks] == [0, 4] and blocks[0].drop_path.p == 0.0 and blocks[1].drop_path.p == 0.1
    assert float(m.w.abs().max()) == 0.0 and list(m.stage_params()) == ["weights"]
    # at start each stage returns its input
    assert all(float(s.out_layer.weight.abs().max()) == 0.0 for s in m.stages)
    assert float(RegFormer(K=2, zero_out=False).stages[1].to_image.weight.abs().max()) > 0   # original initialisation
    keys = set(m.state_dict())
    assert "w" in keys and not any("mask" in k or "rel_index" in k for k in keys)   # no image-size lookup tables saved
    with pytest.raises(ValueError):
        RegFormer(dim=15)


def test_shifted_window_mask_blocks_wrapped_neighbours():
    blk = SwinBlock(dim=8, heads=2, window=4, shift=2, mlp_ratio=2.0, drop_path=0.0)
    mask = blk._mask(8, 8, torch.zeros(1))
    # windows away from the edge: no masking
    assert mask.shape == (4, 16, 16) and float(mask[0].abs().max()) == 0.0
    assert float(mask[3].min()) == -100.0 and float(mask[3].diagonal().abs().max()) == 0.0
    assert SwinBlock(8, 2, 4, 0, 2.0, 0.0)._mask(8, 8, torch.zeros(1)) is None
    # cyclic shift: top-edge tokens must not see bottom-edge tokens even though they share a window after the roll
    torch.manual_seed(0)
    x = torch.randn(1, 8, 8, 8)
    x2 = x.clone()
    x2[:, 0, 4:6] += 1.0                                # edit the top row (columns 4-5, one region of a shifted window)
    d = (blk.eval()(x2) - blk(x)).abs().sum(-1)[0]
    # bottom row unchanged, the edited spot changes
    assert float(d[7].max()) == 0.0 and float(d[0, 4:6].max()) > 0
    with pytest.raises(ValueError, match="divisible"):
        SwinStage(8, True, True, 2, 2, 4, 2.0, 0.0)(torch.zeros(1, 1, 6, 8), None)


def test_forward_gradients_and_checkpoint(op_dir, small_geom, small_projector):
    torch.manual_seed(0)
    y, x0, x = _data(op_dir, small_geom)
    m = build_regformer(TINY, K=4, projector=small_projector)
    assert m.fbp is not None and m.fbp.geom is small_geom
    with torch.no_grad():
        xs, terms = m.eval()(y, x0, return_stages=True, return_terms=True)
        assert torch.equal(xs[-1], m(y, x0))                                         # eval: no randomness left
    assert len(xs) == 5 and all(float(v.min()) >= 0.0 for v in xs[1:])               # each stage goes through a ReLU
    assert set(terms[0]) == {"k", "dc_norm", "reg_norm", "weight"} and terms[0]["dc_norm"] == 0.0    # w starts at 0
    # start = FBP (negative part removed)
    assert torch.equal(xs[-1], torch.relu(x0)) and terms[1]["reg_norm"] == 0.0
    m.train()
    torch.manual_seed(1)
    torch.mean((m(y, x0) - x) ** 2).backward()
    grads = {k: p.grad.clone() for k, p in m.named_parameters()}
    assert all(torch.isfinite(g).all() for g in grads.values()) and float(grads["w"].abs().min()) > 0
    # zero-initialised image output: only it and w get gradients at the first step; make it non-zero to reach every
    # layer
    assert float(grads["stages.0.conv3.weight"].abs().max()) > 0
    assert float(grads["stages.1.to_image.weight"].abs().max()) > 0
    with torch.no_grad():
        for s in m.stages:
            s.out_layer.weight.normal_(std=1e-2)
    m.zero_grad()
    torch.manual_seed(1)
    torch.mean((m(y, x0) - x) ** 2).backward()
    grads = {k: p.grad.clone() for k, p in m.named_parameters()}
    assert all(torch.isfinite(g).all() and float(g.abs().max()) > 0 for g in grads.values())
    m.zero_grad()
    m.grad_checkpoint = True                             # per-stage checkpointing: same gradient (DropPath included)
    torch.manual_seed(1)
    torch.mean((m(y, x0) - x) ** 2).backward()
    assert all(torch.allclose(p.grad, grads[k], rtol=1e-4, atol=1e-8) for k, p in m.named_parameters())
    with pytest.raises(RuntimeError, match="projector"):
        RegFormer(K=2, **TINY)(y, x0)


def test_data_step_uses_filtered_backprojection(op_dir, small_geom, small_projector):
    """Regularization off, w = 1: one stage = x + FBP(y - A x), i.e. the measured part of x replaced by the FBP of
    the data."""
    y, x0, x = _data(op_dir, small_geom)
    m = build_regformer(TINY, K=1, projector=small_projector).eval()
    with torch.no_grad():
        for p in m.stages.parameters():
            p.zero_()
        m.w.fill_(1.0)
        out = m(y, x0)
        ref = torch.relu(x0 + m.fbp(y - small_projector.raw_forward(x0)))
    assert torch.allclose(out, ref, atol=1e-6)
    # fixed point: the true image barely changes (FBP of the residual ~ 0), unlike the artefact-laden x0
    with torch.no_grad():
        moved_truth = float((m(y, x) - x).abs().mean())
        moved_fbp = float((out - x0).abs().mean())
    assert moved_truth < 0.5 * moved_fbp


def test_cost_counts_every_stage(small_projector):
    m = build_regformer(TINY, K=4, projector=small_projector)
    c = model_cost(m, N)
    assert c["K"] == 4 and c["n_params"] == count_params(m) and c["n_params_regularizer"] == c["n_params"] - 4
    two = model_cost(build_regformer(TINY, K=2), N)["gflops_regularizer_total"]
    # more stages, more FLOPs (stages differ, no K multiplier)
    assert 0 < two < c["gflops_regularizer_total"]
    assert c["gflops_regularizer_per_stage"] == pytest.approx(c["gflops_regularizer_total"] / 4)


def test_overfit_one_slice(op_dir, small_geom, small_projector):
    torch.manual_seed(0)
    y, x0, x = _data(op_dir, small_geom)
    m = build_regformer({**TINY, "drop_path": 0.0}, K=4, projector=small_projector)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    p0 = float(psnr(x0, x))
    for _ in range(80):
        loss = torch.mean((m(y, x0) - x) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        assert float(psnr(m.eval()(y, x0), x)) > p0 + 1.0
    assert float(m.w.abs().max()) > 0                    # the data-step weight moved away from 0
