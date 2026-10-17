"""DPMA (Wu et al., Scientific Reports 15:16894, 2025), rebuilt on the project's operators for limited-angle CT.

Original code: github.com/jia-W-w/DPKA. NOT part of the LEARN framework. Three parts, as in the original:

  1. Network M1 (original ``train.py``), ONE set of weights used in both domains:
        p_in  = extended-range sinogram: measured part kept, missing part pre-filled
        p_hat = M1(p_in)          x_a = FBP_extended(p_hat)          x_b = M1(x_a)
        loss  = 0.1 * L1(p_hat, true extended sinogram) + 0.1 * L1(x_a, x) + L1(x_b, x)
  2. Network M2 (original ``train_pic.py``), M1 frozen:  x_c = M2(x_b),  loss = MSE(x_c, x).
     At inference (original ``test_dma.py``) a data-consistency step precedes M2:
        x_dc = x_b - FBP(A x_b - y),   x_c = M2(x_dc)
  3. Non-learned refinement (original ``iter.py``): 300 iterations alternating a preconditioned data step and TV
     denoising of the deviation from x_c (``refine``).

Kept from the original: the M1/M2 architecture (U-Net of MFF blocks: kernel-basis attention (KBA) convolutions + sparse
non-local attention with LSH, width 16), the loss weights, the data-consistency step and every constant of the 300
iterations (step 0.15, lambda 0.1, delta 0.005, 10 TV iterations).

Adapted to limited angle (the original targets sparse view: 64 views interpolated to 192):
  * The "extended sinogram" covers the ``target`` scan range (240°, same angular step, containing the whole measured
    range). Measured views sit at their positions; MISSING views are filled by re-projecting the FBP image x^0
    (instead of the original nearest-neighbour interpolation, which only makes sense when missing views lie between
    measured ones). Its label is the projection of the true image on the reconstruction grid.
  * Sinograms are multiplied by ``sino_scale`` = 76.8 / 400 before entering the network: the original uses a field of
    view of 76.8 units, so the sinogram/image magnitude ratio (and the 0.1 weight of the sinogram loss) matches it.
  * The denominator of the data step in the 300 iterations, FBP(A 1), is clamped at 0 from below: at limited angle it
    is no longer close to 1 and can be negative at a few pixels, flipping the sign of the division.
Different because of the project integration: the project's projector and FBP; Chambolle TV rewritten for the GPU
(same iteration as skimage); LSH hashing at inference uses a fixed generator so an image always gives the same result
(the original reseeds randomly on every call).
"""
from .model import DPMA
from .network import DPMANet, LSHAttention, MFFBlock

__all__ = ["DPMA", "DPMANet", "LSHAttention", "MFFBlock"]
