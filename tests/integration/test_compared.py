"""Methods of the comparison table with the CG-GLORE paper (CG-GLORE, FBPConvNet, Learned Primal-Dual, original
LEARN) on a small operator: published sizes, forward/backward passes, and slice results independent of the batch."""
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lact.core.analysis import model_cost
from lact.core.imaging.intensity import block_mean
from lact.core.imaging.phantom import shepp_logan
from lact.core.models import (
    FBPConvNet,
    LearnedPrimalDual,
    Unrolled,
    build_compared,
    build_unrolled,
    count_params,
)
from lact.core.operators import FanBeamFBP
from lact.infra.operators.operator_store import load_matrix

HAS_CG_GLORE = importlib.util.find_spec("lact.core.models.baselines.cg_glore") is not None
requires_cg_glore = pytest.mark.skipif(not HAS_CG_GLORE, reason="CG-GLORE is not distributed with this copy")

N = 32
TINY = {"cg_glore": {"channels": 4, "landmarks": 16}, "fbpconvnet": {"base": 4}, "lpd": {"channels": 8}}
STAGES = {"cg_glore": 2, "fbpconvnet": 1, "lpd": 2}


def _data(op_dir, geom):
    A64, _ = load_matrix(geom, 64, op_dir)
    ph = shepp_logan(64)
    x = torch.from_numpy(block_mean(ph, 2))[None, None]
    y = torch.from_numpy((A64 @ ph.ravel()).astype(np.float32)).view(1, 1, geom.n_views, geom.n_det)
    return y, FanBeamFBP(geom, N)(y), x


def test_default_architectures_match_the_published_sizes():
    """Default parameter counts versus the CG-GLORE paper's table: 5.5M / 31.1M / 0.251M / 1.8M."""
    unet, lpd = FBPConvNet(), LearnedPrimalDual()
    learn = build_unrolled("learn_orig", {"channels": 48, "kernel": 5}, 256, K=30, share_theta=False)
    assert all(isinstance(m, Unrolled) and not m.uses_cg for m in (unet, lpd, learn))
    assert (unet.K, lpd.K, learn.K) == (1, 10, 30)
    assert count_params(unet) == 31_081_345 and count_params(lpd) == 253_220
    assert count_params(learn) == 1_802_970 and learn.regularizer_params() == 30 * 60_097
    assert len(learn.regs) == 30 and unet.stage_params() == {} == lpd.stage_params()
    with pytest.raises(KeyError):
        build_compared("qn_mixer", {}, K=14)


@requires_cg_glore
def test_cg_glore_matches_the_published_size():
    from lact.core.models.baselines.cg_glore import CGGlore

    cg = CGGlore()
    assert isinstance(cg, Unrolled) and not cg.uses_cg and cg.K == 15 and cg.cg_iters == 10
    assert count_params(cg) == 5_522_940 and cg.regularizer_params() == 5_522_925      # + 15 alpha
    assert torch.allclose(cg.alpha, torch.full((15,), 0.1)) and list(cg.stage_params()) == ["alphas"]


@requires_cg_glore
def test_cg_glore_tokens_and_attention():
    """Attention tokens concatenate the features at one position of the four image patches (like the original
    ``unfold``) and can be folded back."""
    from lact.core.models.baselines.cg_glore import GloreBlock, NystromAttention

    block = GloreBlock(channels=4, landmarks=16).eval()
    z = torch.randn(2, 4, N, N)
    tokens = block.patchify(z)
    half = N // 2
    ref = z.unfold(2, half, half).unfold(3, half, half).contiguous().view(2, 4, 4, half, half)
    ref = ref.view(2, -1, half * half).permute(0, 2, 1)
    assert tokens.shape == (2, half * half, 16) and torch.equal(tokens, ref)
    assert torch.equal(block.unpatchify(tokens, z.shape), z)
    with pytest.raises(ValueError):
        block.patchify(torch.randn(1, 4, 31, 31))
    attn = NystromAttention(16, dim_head=8, heads=2, landmarks=16, pinv_iters=2, dropout=0.0).eval()
    # 250 is not divisible by 16 landmarks: pad at the start, then crop
    odd = torch.randn(2, 250, 16)
    with torch.no_grad():
        both = attn(odd)
        assert both.shape == odd.shape and torch.isfinite(both).all()
        assert torch.allclose(both[:1], attn(odd[:1]), atol=1e-5)       # samples computed independently of the batch
        x = torch.rand(2, 1, N, N)
        assert torch.allclose(block(x)[1:], block(x[1:]), atol=1e-6)


@requires_cg_glore
def test_cg_glore_solves_its_linear_system(op_dir, small_geom, small_projector):
    """With a zero regularization block each stage solves (alpha F A + I) x = alpha F y + x_k: the residual must
    drop."""
    y, x0, _ = _data(op_dir, small_geom)
    m = build_compared("cg_glore", TINY["cg_glore"], K=1, projector=small_projector).eval()
    for p in m.blocks.parameters():
        torch.nn.init.zeros_(p)
    with torch.no_grad():
        fy = m._fbp(y)
        rhs = m.alpha[0] * fy + x0
        out = m(y, x0)
        r0, r1 = (rhs - m._h(x0, 0)).norm(), (rhs - m._h(out, 0)).norm()
        assert torch.allclose(fy, x0, atol=1e-5)                        # the model's F is the FBP that produced x^0
        assert float(r1) < 0.05 * float(r0)


@pytest.mark.parametrize("name", [pytest.param("cg_glore", marks=requires_cg_glore), "fbpconvnet", "lpd"])
def test_forward_backward_and_batch_independence(name, op_dir, small_geom, small_projector):
    y, x0, x = _data(op_dir, small_geom)
    torch.manual_seed(0)
    m = build_compared(name, TINY[name], K=2, projector=small_projector)
    assert isinstance(m, Unrolled) and STAGES[name] == m.K
    m.train()
    loss = torch.mean((m(y, x0) - x) ** 2)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    m.zero_grad()
    m.grad_checkpoint = True                             # the trainer's fallback on GPU out-of-memory
    torch.mean((m(y, x0) - x) ** 2).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        xs, terms = m(y, x0, return_stages=True, return_terms=True)
        assert len(xs) == m.K + 1 and len(terms) == m.K and torch.equal(xs[0], x0)
        assert all(v.shape == x0.shape and torch.isfinite(v).all() for v in xs)
        y2, x02 = torch.cat([y, 0.5 * y.flip(-1)]), torch.cat([x0, 0.5 * x0.flip(-1)])
        assert torch.allclose(m(y2, x02)[:1], xs[-1], atol=1e-5)        # a batch neighbour does not change the result
    cost = model_cost(m, N)
    assert cost["n_params"] == count_params(m) and cost["n_params_regularizer"] == m.regularizer_params()


def test_fbpconvnet_and_lpd_details(op_dir, small_geom, small_projector):
    y, x0, _ = _data(op_dir, small_geom)
    unet = FBPConvNet(base=4).eval()
    torch.nn.init.zeros_(unet.out.weight)
    torch.nn.init.zeros_(unet.out.bias)
    with torch.no_grad():
        assert torch.equal(unet(None, x0), x0)                          # x = x^0 + net(x^0); the sinogram is unused
        with pytest.raises(ValueError):
            unet(None, torch.zeros(1, 1, 24, 24))                       # 24 is not divisible by 2^4
        zero = LearnedPrimalDual(K=2, channels=8, init="zero", projector=small_projector).eval()
        fbp = LearnedPrimalDual(K=2, channels=8, init="fbp", projector=small_projector).eval()
        fbp.load_state_dict(zero.state_dict())
        assert not torch.allclose(zero(y, x0), fbp(y, x0), atol=1e-4)   # the two image initialisations
    assert zero.dual[0][0].in_channels == 7 and zero.primal[0][0].in_channels == 6
    # per-channel PReLU initialised to 0 as in the original code
    assert float(zero.dual[0][1].weight.detach().abs().max()) == 0.0
    with pytest.raises(ValueError):
        LearnedPrimalDual(init="random")
    with pytest.raises(RuntimeError):
        LearnedPrimalDual()(y, x0)


# ---------------------------------------------------------------------------- DPMA
def _dpma(op_dir, geom, projector, **kw):
    from lact.core.models import DPMA
    from lact.infra.operators.operator_store import build_and_save, load_projector

    full = geom.replace(name="t240", span_deg=240.0)
    build_and_save(full, N, op_dir)
    kw = {"width": 4, "chunk_size": 4, "refine_iters": 6, **kw}
    return DPMA(projector=projector, target_projector=load_projector(full, N, op_dir, with_adjoint=False), **kw)


def test_dpma_network_matches_the_original():
    """The DPMA network: parameter count as in the original code (8.06M; the CG-GLORE paper reports 8.1M), a correct
    hand-written KBA derivative, and identical inference results on every call (the original hashes randomly)."""
    import torch.nn.functional as F

    from lact.core.models.baselines.dpma.network import _KBA, DPMANet, LSHAttention

    assert count_params(DPMANet(16)) == 8_056_321
    torch.manual_seed(0)
    B, c, H, W, nset, k, g = 2, 8, 5, 6, 3, 3, 2
    x, att = (torch.randn(B, n, H, W, dtype=torch.double, requires_grad=True) for n in (c, nset))
    b = torch.randn(1, nset, c, dtype=torch.double, requires_grad=True)
    w = torch.randn(1, nset, c * c // g * k * k, dtype=torch.double, requires_grad=True)

    def plain():
        a = att.reshape(B, nset, H * W).transpose(-2, -1)
        attk = (a @ w).reshape(B, H * W, g, c // g, c // g * k * k)
        uf = F.unfold(x, kernel_size=k, padding=k // 2).reshape(B, g, c // g * k * k, H * W).permute(0, 3, 1, 2)
        out = (attk @ uf.unsqueeze(-1)).squeeze(-1).reshape(B, H * W, c) + a @ b
        return out.transpose(-1, -2).reshape(B, c, H, W)

    fast, ref = _KBA.apply(x, att, k, g, b, w), plain()
    G = torch.randn_like(ref)
    for mine, auto in zip(torch.autograd.grad((fast * G).sum(), (x, att, b, w)),
                          torch.autograd.grad((ref * G).sum(), (x, att, b, w))):
        assert torch.allclose(mine, auto, atol=1e-10)
    assert torch.allclose(fast, ref, atol=1e-12)
    net = DPMANet(4, chunk_size=4).eval()
    z = torch.rand(2, 1, 40, 36)                         # not divisible by 16: pad, then crop
    with torch.no_grad():
        out = net(z)
        assert out.shape == z.shape and torch.equal(out, net(z))
        assert torch.allclose(out[:1], net(z[:1]), atol=1e-5)
        with pytest.raises(ValueError):
            LSHAttention(8, chunk_size=144)(torch.rand(1, 8, 8, 8))     # 64 token < 144


def test_tv_chambolle_matches_skimage():
    """The GPU TV implementation runs the same iteration as the skimage function the original DPMA calls (early stop
    included)."""
    sk = pytest.importorskip("skimage.restoration")
    from lact.core.solvers.tv_chambolle import denoise_tv_chambolle

    rng = np.random.default_rng(0)
    imgs = rng.random((3, 40, 37))
    imgs[1] = np.clip(imgs[1] * 0.05 + np.linspace(0, 1, 37)[None], 0, 1)          # smooth image: stops before max_iter
    for weight, iters in ((0.005, 10), (0.1, 200), (0.3, 50)):
        ref = np.stack([sk.denoise_tv_chambolle(im, weight=weight, max_num_iter=iters) for im in imgs])
        got = denoise_tv_chambolle(torch.from_numpy(imgs)[:, None], weight=weight, max_iter=iters)[:, 0].numpy()
        assert np.abs(ref - got).max() < 1e-12
    with pytest.raises(ValueError):
        denoise_tv_chambolle(torch.zeros(4, 4))


def test_dpma_two_stages_and_refinement(op_dir, small_geom, small_projector):
    y, x0, x = _data(op_dir, small_geom)
    torch.manual_seed(0)
    s1 = _dpma(op_dir, small_geom, small_projector, stage=1)
    assert isinstance(s1, Unrolled) and s1.K == 1 and s1.m2 is None and s1.stage_params() == {}
    assert s1.view_offset == 30 and s1.target.geom.n_views == 120              # 120° centred in 240°, 2° steps
    s1.train()
    loss = s1.training_loss(y, x0, x)
    loss.backward()
    grads = [p.grad for p in s1.m1.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads)
    with torch.no_grad():
        p_hat, x_a, x_b = s1.eval()._stage1(y, x0)
        assert p_hat.shape == (1, 1, 120, small_geom.n_det) and x_a.shape == x_b.shape == x0.shape
        assert torch.equal(s1(y, x0), s1.validation_forward(y, x0))          # stage 1 has no refinement
    # stage 2: M1 loaded from stage 1 and frozen, only M2 learns
    s2 = _dpma(op_dir, small_geom, small_projector, stage=2)
    s2.m1.load_state_dict(s1.m1.state_dict())
    assert not any(p.requires_grad for p in s2.m1.parameters()) and count_params(s2) == count_params(s2.m2)
    s2.train()
    assert not s2.m1.training and s2.m2.training
    s2.training_loss(y, x0, x).backward()
    assert all(p.grad is None for p in s2.m1.parameters()) and any(p.grad is not None for p in s2.m2.parameters())
    s2.eval()
    with torch.no_grad():
        net = s2.validation_forward(y, x0)
        dc = s2._data_consistency(x_b, y)
        # data consistency then M2, as in the original test_dma.py
        assert torch.allclose(net, s2.m2(dc), atol=1e-6)
        out = s2(y, x0)
        assert out.shape == x0.shape and torch.isfinite(out).all() and float(out.min()) >= 0.0
        assert not torch.allclose(out, net, atol=1e-5)                       # the 5 refinement iterations ran
        s2.refine_iters = 1
        assert torch.equal(s2(y, x0), net)
        s2.refine_iters = 6
        y2, x02 = torch.cat([y, 0.5 * y.flip(-1)]), torch.cat([x0, 0.5 * x0.flip(-1)])
        # a batch neighbour does not change the result
        assert torch.allclose(s2(y2, x02)[:1], out, atol=1e-5)
        xs, terms = s2(y, x0, return_stages=True, return_terms=True)
        assert len(xs) == 2 and len(terms) == 1 and terms[0]["dc_norm"] > 0
    shifted = SimpleNamespace(geom=small_geom.replace(center_deg=1.0), img_size=N, device=torch.device("cpu"))
    # angular grid offset by half a step: cannot fill
    with pytest.raises(ValueError):
        s2.set_projector(shifted)
    with pytest.raises(ValueError):
        _dpma(op_dir, small_geom, small_projector, stage=3)
