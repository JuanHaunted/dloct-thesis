# DLOCT — Cold Diffusion for Sub-Nyquist OCT Reconstruction

Recovering laterally undersampled complex OCT tomograms with a cold-diffusion model whose
forward process is the *physical* sampling degradation rather than Gaussian noise.

> **Status: pre-training prototype.** The sampling-theory analysis is substantially complete and
> sound. The diffusion model does not yet run — see [Known issues](#8-known-issues). There is no
> training script, dataset class, or config system. This README is the specification the code
> should be brought in line with.

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Part I — Sampling theory](#2-part-i--sampling-theory)
3. [Part II — Which degradation operator is correct](#3-part-ii--which-degradation-operator-is-correct)
4. [Part III — The target operator](#4-part-iii--the-target-operator)
5. [Part IV — Cold diffusion](#5-part-iv--cold-diffusion)
6. [Part V — Data consistency](#6-part-v--data-consistency)
7. [Notation reference](#7-notation-reference)
8. [Known issues](#8-known-issues)
9. [References](#9-references)

---

## 1. Repository layout

| Path | Role | State |
|---|---|---|
| `src/dloct/sampling_analysis.py` | MPS / Gaussian-fit sampling analysis (numpy) | Sound |
| `src/dloct/models/convnext_unet.py` | ConvNeXt U-Net, linear attention, time embedding | Imports; to be reworked |
| `src/dloct/diffusion/aliasing_diffusion.py` | Cold diffusion, Gaussian mask + sampler | Wrong operator (see §3); to be replaced |
| `configs/` | Experiment configs (YAML) | — |
| `scripts/` | SLURM job scripts | — |
| `tests/` | Operator assertions (§4) | — |
| `notebooks/` | Sampling-theory analysis (MPS, decimation, aliasing) | Exploratory |
| `figures/` | Thesis figures | — |
| `docs/literature_review.md` | Literature review: complex/phase-aware reconstruction | — |
| `docs/dloct_math.pdf` | Math notes | — |
| `data/` | Local data sample (git-ignored, see `data/README.md`) | — |

### Setup

```bash
uv sync          # creates .venv with torch (CUDA 12.8 wheels) and the dloct package
uv run pytest    # operator tests
```

---|---|---|
| `src/lateral_sampling.py` | MPS / Gaussian-fit sampling analysis | Sound, one missing function |
| `src/prepare_dataset.py` | Builds subsampled training pairs | Runs; redundant with on-the-fly degradation |
| `src/diffusion/aliasing_diffusion.py` | Cold diffusion, Gaussian mask + sampler | Does not run |
| `src/models/spectral_diffusion.py` | Cold diffusion, brick-wall mask, no sampler | Trains; cannot reconstruct |
| `src/models/convnext_unet.py` | ConvNeXt U-Net, linear attention, time embedding | Does not import |
| `notebooks/` | Sampling theory, wavelets, Gabor experiments | Exploratory |

---

## 2. Part I — Sampling theory

An OCT tomogram is a **complex** field $T(z,x,y) = A e^{i\phi} \in \mathbb{C}$. Laterally the
system is a focused Gaussian beam, so the lateral PSF is Gaussian and the lateral power spectrum
is Gaussian too:

$$P(\hat f) = A \exp\left(-\frac{\hat f^{2}}{2\sigma^{2}}\right) + c$$

`compute_mps_1d` estimates this per depth slice (FFT along one lateral axis, modulus squared,
average over the orthogonal axis, normalize peak to 1); `fit_gaussian_to_mps` fits it and converts
to a half-width:

$$\mathrm{HWHM} = \sigma\sqrt{2\ln 2} \approx 1.177\,\sigma$$

Everything is in **normalized frequency** $\hat f \in [-\tfrac12, \tfrac12)$, so Nyquist is always
$\pm 0.5$ regardless of physical pixel pitch.

### The key relation

Decimation by $K$ (keep every $K$-th A-line) periodizes the spectrum. With $W$ the original lateral
width, $N' = W/K$, and $X$ the length-$W$ DFT:

$$Y[\kappa] = \frac{1}{K}\sum_{p=0}^{K-1} X[\kappa + pN'], \qquad \kappa = 0,\dots,N'-1$$

The physical bandwidth is unchanged, but the Nyquist window shrank by $K$, so the *normalized*
bandwidth stretches by $K$:

$$\mathrm{HWHM}_K \approx \min\left(K \cdot \mathrm{HWHM}_0,\ 0.5\right)$$

It saturates at $0.5$ because that is all that is observable — beyond it, energy **folds**. Hence
the **critical factor**:

$$\boxed{\,K^{\ast} = \frac{0.5}{\mathrm{HWHM}_0}\,}$$

If $\mathrm{HWHM}_0 < 0.5$ the volume is oversampled and there is $K^{\ast}$ worth of free headroom
before aliasing begins. This is the point of the analysis half: it derives the meaningful
subsampling factor **from the data** instead of picking one arbitrarily.

Two caveats to state in the thesis rather than leave implicit:

- The Gaussian fit is performed on **linear** power over a spectrum with enormous dynamic range, so
  it is dominated by the central peak and blind to the tails. `compute_spectral_halfwidth`
  (threshold-based) is the more honest metric of the two.
- Once $K > K^{\ast}$ the measured spectrum is a **fold**, not a Gaussian, so fitting a Gaussian to
  decimated data is fitting the wrong model. Saturation near $0.5$ is partly real and partly fit
  breakdown.

---

## 3. Part II — Which degradation operator is correct

Three physically distinct degradations get confused with one another. They are **not** the same
operator, and only the third is lateral undersampling.

| | Degradation | k-space operator | Physical cause |
|---|---|---|---|
| **A** | Resolution loss | multiply by Gaussian | reduced NA, defocus, aberration |
| **B** | Bandwidth truncation | multiply by brick-wall | *ideal anti-aliased* downsampling |
| **C** | **Aliasing** | **fold-in, then brick-wall** | **true sub-Nyquist sampling** |

A real OCT system that laterally subsamples does **not** anti-alias filter first — you simply
acquire fewer A-lines. So the thesis is about **C**.

### What each file actually implements

**`aliasing_diffusion.py` — case A.** Gaussian mask, $\sigma(t) = X\rho(t)/3$, with
$\rho(t) = 1 - \tfrac{t}{T}(1 - \tfrac1K)$. Measured at $X = 512$, $K = 4$:

| $t$ | $\sigma$ (bins) | at DC | at Nyquist | at $\pm X/8$ |
|---:|---:|---:|---:|---:|
| 0 | 170.7 | 1.000 | **0.325** | 0.932 |
| 500 | 106.7 | 1.000 | 0.056 | 0.835 |
| 1000 | 42.7 | 1.000 | 0.000 | **0.325** |

Two failures. First, $D(x,0) \neq x$ — measured relative error **0.35**, which breaks the
cold-diffusion contract outright. Second, at $t = T$ the notional $4\times$ band edge $\pm X/8$
still transmits 32%, so the terminal state is not a $4\times$ measurement in any defensible sense.
Both stem from a factor-2 slip: the code sets $3\sigma = X$ (the *full* width) where the intent was
$3\sigma = X/2$ (the *half* width), making the Gaussian twice as wide as designed at every $t$.

**`spectral_diffusion.py` — case B.** Brick-wall mask keeping the central fraction $\rho(t)$, with
$\rho_{\min} = 0.1$. Verified: $D(x,0) = x$ to machine precision, retained bins
$2\lfloor \lfloor W/2\rfloor \rho(t)\rfloor$ match theory exactly (64, 62, 34, 6 out of 64 at
$t = 0, 1, 50, 100$). The mask is correct for what it is. But it has **no sampler at all**
(`hasattr(model, 'sample')` is `False`), so it can be trained and then cannot reconstruct anything.
Its $\rho_{\min}=0.1$ also implies $K = 10$, disagreeing with the other file's $K = 4$.

### Scorecard

| Component | `aliasing_diffusion` | `spectral_diffusion` | Correct? |
|---|---|---|---|
| Framing / intent | sub-Nyquist ✅ | generic low-pass | **aliasing** |
| Mask satisfies $D(x,0)=x$ | ❌ 35% error | ✅ exact | **spectral** |
| Terminal state matches $K$ | ❌ 32% leak | ✅ exact | **spectral** |
| Implements fold-in | ❌ | ❌ | **neither** |
| Reverse sampler | ✅ (buggy) | ❌ absent | **aliasing** |
| Data consistency | ❌ averages all bands | ✅ masked replacement | **spectral** |
| Phase-aware loss | ✅ present | ❌ absent | **aliasing**, but see §5 |
| Complex-field FFT | ✅ | ✅ | both |

**Conclusion.** Keep `aliasing_diffusion.py` as the surviving module — its scope and framing are
right. Replace its degradation with §4, and port `spectral_diffusion`'s masked data-consistency
step (§6) and its mask discipline. Delete `spectral_diffusion.py` afterwards.

---

## 4. Part III — The target operator

### The measurement

Sinc-interpolating a $K$-fold decimated signal back onto the fine $W$-grid gives, in normalized
frequency:

$$\boxed{\;X_{\mathrm{meas}}(\hat f) \;=\; \frac{1}{K}\sum_{p=0}^{K-1} X\!\left(\hat f + \frac{p}{K}\right)\cdot \mathbf{1}\!\left[\,\lvert \hat f\rvert < \tfrac{1}{2K}\,\right]\;}$$

The $\sum_p$ is the entire difference between this and both existing files. Aliasing **misplaces**
energy; a mask **removes** it. Under a mask the passband stays pristine; under real decimation the
passband is contaminated by folded content, and undoing that contamination is the actual problem.

### The continuous family

Cold diffusion needs a smooth path from identity to measurement, but decimation only exists at
integer $K$. Resolve this by ramping the fold-in amplitude instead of the decimation factor. Let
$\mathcal{T}_s$ be a circular shift by $s$ bins (`torch.roll`) and $M_\rho$ the brick-wall window
retaining the central fraction $\rho$. Require $K \mid W$. Then:

$$\boxed{\;D(x_0, t) \;=\; g(t)\cdot\mathcal{F}^{-1}S^{-1}\!\left[\, M_{\rho(t)} \odot \left( \tilde x_0 \;+\; \alpha(t)\sum_{p=1}^{K-1} \mathcal{T}_{pW/K}\,\tilde x_0 \right) \right]\;}$$

with schedules

$$\rho(t) = 1 - \left(1 - \tfrac{1}{K}\right)\tfrac{t}{T}, \qquad \alpha(t) = \tfrac{t}{T}, \qquad g(t) = \frac{1}{1 + \alpha(t)(K-1)}$$

Check the endpoints:

- $t = 0$: $\rho = 1$, $\alpha = 0$, $g = 1$ $\Rightarrow$ $D(x,0) = x$ **exactly**.
- $t = T$: $\rho = 1/K$, $\alpha = 1$, $g = 1/K$ $\Rightarrow$ $D(x,T) = \frac{1}{K}\sum_{p=0}^{K-1}\mathcal{T}_{pW/K}\tilde x_0$ windowed to $W/K$ bins — **exactly the measurement above.**

A shift of $p/K$ in normalized frequency is a shift of $pW/K$ bins, which is why $K \mid W$ is
required. Choose $W$ a power of two and $K \in \{2,4,8,16\}$. The whole thing is about ten lines of
`torch` and is fully differentiable.

**Verified numerically** ($H=8$, $W=64$, $K=4$, $T=100$, float64, random complex input):

| Check | Result |
|---|---|
| $\lVert D(x,0)-x\rVert / \lVert x\rVert$ | $2.6\times 10^{-16}$ — machine precision |
| $D(x,T)$ vs. true decimation + sinc interpolation | $3.4\times 10^{-16}$, best-fit scale $= 1.000000$ |
| Idempotence $\lVert D(D(x,T),T) - D(x,T)\rVert / \lVert D(x,T)\rVert$ | $0.75$ — correctly **not** a projector |

The unit best-fit scale confirms $g(t)$ is correct as written, not merely correct up to a constant.

**Keep these as assertions in the test suite** — they are what make the forward process defensible:

```
assert allclose(D(x, 0), x)                                  # identity contract
assert allclose(D(x, T), subsample_lateral(x, K, interpolate=True))   # matches the numpy reference
```

### What this costs, and why it is worth it

The brick-wall operator in `spectral_diffusion` is an **orthogonal projector** $P_t$: idempotent
($P_t^2 = P_t$) and nested ($P_{t'}P_t = P_{t'}$ for $t' \ge t$). That is a very strong structure,
and it collapses the whole method. Since $P_t$ is linear, the cold-diffusion update telescopes:

$$x_{t-1} = x_t - P_t\hat x_0 + P_{t-1}\hat x_0 = x_t + \left(P_{t-1} - P_t\right)\hat x_0$$

and $P_{t-1}-P_t$ is itself a projector onto the thin annulus of newly revealed frequencies. So
under operator **B**, cold diffusion is *exactly* greedy band-by-band spectral extrapolation, data
consistency is automatic, and the method reduces to a learned Gerchberg–Papoulis iteration.

Adding fold-in destroys idempotence. $D(\cdot,t)$ stays linear but is no longer a projector, the
telescoping collapse is lost, and explicit data consistency becomes genuinely necessary.

**This is an argument in favour of the change, not against it.** Under operator B the iterative
machinery has almost nothing to do that one U-Net pass could not, which is precisely why a
single-shot baseline would be hard to beat. Under operator C the inverse problem is genuinely
harder and iterative refinement has real work to do.

---

## 5. Part IV — Cold diffusion

### Representation

The network cannot consume complex tensors, so the field is carried as two real channels:

$$\mathbf{x}_0 \in \mathbb{R}^{B\times 2\times H\times W}, \qquad \mathbf{x}_0[b,0] = \Re\,x_0,\quad \mathbf{x}_0[b,1] = \Im\,x_0$$

with packing maps $\mathcal{C}(\mathbf{x}) = \mathbf{x}[0] + i\,\mathbf{x}[1]$ and
$\mathcal{R}(x) = (\Re\,x, \Im\,x)$.

$\mathcal{F}$ is the **unitary** DFT along the lateral axis only, applied row-wise in depth:

$$(\mathcal{F}x)[h,k] = \frac{1}{\sqrt{W}}\sum_{w=0}^{W-1} x[h,w]\,e^{-2\pi i k w/W}$$

so $\mathcal{F}^{-1} = \mathcal{F}^{*}$ and Parseval holds. $S$ is `fftshift`; after shifting,
index $j$ carries frequency $\hat f_j = (j - c)/W$ with $c = \lfloor W/2\rfloor$.

Because the OCT lateral field is complex-analytic its spectrum is **not** conjugate-symmetric, so a
full complex FFT (not `rfft`) is required. Both existing files get this right, and it is the main
thing distinguishing this work from intensity-only OCT super-resolution.

### Objective

$$
\begin{aligned}
t &\sim \mathcal{U}\{1,\dots,T\} \\
\mathbf{x}_t &= \mathbf{D}(\mathbf{x}_0, t) \\
\hat{\mathbf{x}}_0 &= R_\theta(\mathbf{x}_t, t)
\end{aligned}
$$

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}_0,\,t}\Big[\,\big\lVert R_\theta(\mathbf{D}(\mathbf{x}_0,t),\,t) - \mathbf{x}_0\big\rVert_1\,\Big]$$

The network predicts $x_0$ directly (not a residual, not a noise estimate).

**On the $\ell_1$ term.** The separable $\ell_1$ over $(\Re, \Im)$ is *not* $\mathbb{E}\lvert\Delta\rvert$;
it is the complex-plane $\ell_1$, which is **anisotropic** — an error of fixed modulus is penalized
more along a diagonal of the complex plane than along the axes, giving the loss a weak unphysical
preference for certain absolute phases. Use a Charbonnier loss on the modulus,
$\mathbb{E}\sqrt{(\Re\Delta)^2 + (\Im\Delta)^2 + \varepsilon}$, which is rotation-invariant.

**On the phase term.** `aliasing_diffusion` adds $\lambda\,\mathbb{E}[1 - \cos(\hat\theta - \theta)]$,
correctly $2\pi$-periodic so wraps are not punished. But it is unweighted across the whole frame,
and OCT speckle phase is essentially uniform-random per pixel — in background regions
$\lvert x\rvert \to 0$, `atan2` is numerically meaningless, and the target is genuinely
unpredictable. Weight by $\lvert x_0\rvert$ or restrict to a tissue mask. Given `data/train/phase/`,
the physically meaningful quantity is likely the **inter-A-line phase difference**, not absolute
per-pixel phase.

### Sampling (TACoS)

$$
\begin{aligned}
&\textbf{input } \mathbf{x}_T = \text{measurement} \\
&\textbf{for } t = T,\dots,1: \\
&\qquad \hat{\mathbf{x}}_0 \leftarrow R_\theta(\mathbf{x}_t, t) \\
&\qquad \hat{\mathbf{x}}_0 \leftarrow \mathrm{DC}(\hat{\mathbf{x}}_0, \mathbf{x}_T; M_T) \quad\text{(optional)} \\
&\qquad \mathbf{x}_{t-1} \leftarrow \mathbf{x}_t - \mathbf{D}(\hat{\mathbf{x}}_0, t) + \mathbf{D}(\hat{\mathbf{x}}_0, t-1) \\
&\textbf{return } \mathbf{x}_0
\end{aligned}
$$

The subtract-and-add form exists so the *systematic* error of an imperfect $R_\theta$ cancels
between consecutive steps. If $R_\theta$ were exact then $D(\hat x_0, t) = x_t$ and the update
lands exactly on $D(x_0, t-1)$. The naive alternative $x_{t-1} = D(R_\theta(x_t), t-1)$ accumulates
that bias instead.

### Schedule / width constraint

Under a brick-wall mask the cutoff $c(t) = \lfloor c\,\rho(t)\rfloor$ takes only
$\lfloor \tfrac{W}{2}(1-\rho_{\min})\rfloor$ distinct integer values. Whenever $c(t-1) = c(t)$ the
reverse step is a **literal no-op** that still costs a full U-Net forward pass. At $W = 64$,
$T = 100$ there are only 29 distinct values, so **71 of 100 reverse steps do nothing**. Require:

$$T \lesssim \tfrac{W}{2}\left(1 - \rho_{\min}\right)$$

At $W = 512$, $K = 4$ this permits $T \approx 190$; at $W = 64$ it permits about $T = 29$.

---

## 6. Part V — Data consistency

Given a network estimate $\hat x$, the measurement $y$, and mask $M$:

$$\mathrm{DC}(\hat x, y; M) = \mathcal{F}^{-1}S^{-1}\Big[(1-M)\odot S\mathcal{F}\hat x \;+\; M\odot S\mathcal{F}y\Big]$$

Keep the network's extrapolated stop-band; overwrite the passband with the hard measurement. In
projector form $\mathrm{DC}(\hat x, y) = (I-P)\hat x + Py$, the orthogonal projection onto the
measurement-consistent set $\{z : Pz = Py\}$. This is the form in `spectral_diffusion`, and it is
correct. `aliasing_diffusion` instead computes $\tfrac12(\hat X + Y)$ over **all** frequencies,
which halves the network's high-frequency output and injects measurement content into bands where
the measurement has none.

**Never apply DC during training.** For any passband component $u \in \operatorname{ran}(P)$,

$$\frac{\partial}{\partial u}\,\mathcal{L}\big(\mathrm{DC}(R_\theta(\cdot), y)\big) = 0$$

because $P$ annihilates the network's contribution there. The network gets no gradient in the
passband and is free to emit garbage in exactly the band it should find easiest, which then
corrupts the features feeding stop-band prediction. The source comment in `spectral_diffusion`
makes this point and it is correct — a subtlety much published work gets wrong.

---

## 7. Notation reference

| Symbol | Meaning |
|---|---|
| $H, W$ | axial (depth) and lateral sample counts |
| $x_0$ | clean complex field, $\mathbb{C}^{H\times W}$ |
| $\mathbf{x}_0$ | two-channel real packing, $\mathbb{R}^{2\times H\times W}$ |
| $\mathcal{F}, S$ | unitary lateral DFT; `fftshift` |
| $\tilde x$ | $S\mathcal{F}x$, shifted lateral spectrum |
| $c$ | $\lfloor W/2\rfloor$, the DC index after shifting |
| $\hat f$ | normalized frequency, $(j-c)/W \in [-\tfrac12,\tfrac12)$ |
| $K$ | terminal subsampling factor |
| $K^{\ast}$ | critical factor $0.5/\mathrm{HWHM}_0$ |
| $T$ | number of diffusion steps |
| $\rho(t)$ | retained spectral fraction |
| $\alpha(t)$ | fold-in amplitude |
| $M_\rho$ | brick-wall window |
| $\mathcal{T}_s$ | circular shift by $s$ bins |
| $D, P_t$ | degradation operator; projector when brick-wall |
| $R_\theta$ | restoration network (U-Net) |

---

## 8. Known issues

### Blocking

1. **`convnext_unet.py:65` does not import.** `fn: nn.Module | callable[..., torch.Tensor]` uses
   the builtin `callable`, not `typing.Callable`; annotations evaluate at class-definition time.
   `TypeError: 'builtin_function_or_method' object is not subscriptable`.
2. **`lateral_sampling.py:481` calls undefined `_fft_interpolate`.** Only `_linear_interpolate_1d`
   exists, so `subsample_lateral(..., interpolate=True)` is a guaranteed `NameError`.
3. **`aliasing_diffusion.py:127`** — `b = x_T.shape` should be `x_T.shape[0]`; `sample()` raises
   `TypeError` immediately.
4. **No training infrastructure.** No `Dataset`, `DataLoader`, optimizer, checkpointing, or config.
   The `EMA` class is defined and never used. `main.py` is a hello-world.

### Design

5. **Neither degradation implements aliasing** (§3–4) — the central issue.
6. **No normalization anywhere.** Raw linear OCT Re/Im spans orders of magnitude, so $\ell_1$ will
   be dominated by the brightest pixels. Needs a **linearity-preserving** scale (e.g. per-volume
   99.9th-percentile amplitude) — log compression would break the linearity the FFT degradation
   depends on. Probably the largest practical obstacle to convergence.
7. **No baseline.** A single-shot U-Net regressing measurement $\to x_0$, same architecture and
   data, must be built and beaten. This is the first question a committee will ask.
8. **Six volumes, no split.** B-scans within a volume are heavily correlated — split by **volume**,
   not by B-scan.
9. **`prepare_dataset.py` is redundant.** Cold diffusion applies `q_sample` on the fly at random
   $t$; it never wants pre-degraded pairs. As written it materializes 5 factors × 2 axes of
   full-size copies — 20+ GB per volume. It should emit clean $x_0$ patches only.
10. **Axis conventions conflict.** `visualize_downsampling.py` assumes `(B-scan, Z, X, 2)`;
    `lateral_sampling.py` assumes `(depth, x, y[, 2])`.
11. **Minor.** `to_complex` silently drops all but the first polarimetric channel (files are named
    `polInt1_polOut2`); `ComplexSafeLayerNorm`'s per-channel gain/bias contradicts its docstring
    promise that Re/Im are scaled by the same scalar; `sample()` unconditionally calls `.train()`
    on exit; `image_size` is stored and never used; odd $W$ breaks the identity contract by one bin.

### Verified working

The U-Net channel arithmetic is correct — all four resolution levels and skip concatenations were
traced by hand.

### Suggested order

1. Fix issues 1–3.
2. Implement §4 in `aliasing_diffusion.py`; add the two assertions. Delete `spectral_diffusion.py`
   after porting its DC step.
3. Normalization + `Dataset` (patches, split by volume).
4. Training script **and** the single-shot baseline together.
5. Metrics: PSNR/SSIM on log-amplitude; complex correlation coefficient for phase; and — most
   relevant to the framing — does the recovered MPS half-width match ground truth? Part I already
   has the tooling for that last one, and it is the metric that most directly tests "did we undo
   the aliasing."

---

## 9. References

> Entries marked ⚠️ are from memory and should be verified against the venue before entering a
> thesis bibliography. Unmarked arXiv identifiers are high-confidence.

### Read these six first, in this order

1. **Bansal et al.**, *Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise*,
   [arXiv:2208.09392](https://arxiv.org/abs/2208.09392) (2022) — the method this repo implements.
2. **Ho, Jain, Abbeel**, *Denoising Diffusion Probabilistic Models*,
   [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) (2020) — the baseline it departs from.
3. **Hoogeboom & Salimans**, *Blurring Diffusion Models*,
   [arXiv:2209.05557](https://arxiv.org/abs/2209.05557) (2022) — diffusion posed **directly in the
   frequency domain**. Closest formal relative of this work; read carefully.
4. **Rissanen, Heinonen, Solin**, *Generative Modelling with Inverse Heat Dissipation*,
   [arXiv:2206.13397](https://arxiv.org/abs/2206.13397) (2022) — heat equation as forward process,
   i.e. a Gaussian low-pass schedule in Fourier. Independent derivation of operator **A**.
5. **Papoulis**, *A new algorithm in spectral analysis and band-limited extrapolation*,
   IEEE Trans. Circuits Syst. **22**(9), 1975 — with Gerchberg (1974), the classical algorithm your
   §4 discussion shows operator **B** reduces to. Non-negotiable citation.
6. **Schlemper et al.**, *A Deep Cascade of CNNs for Dynamic MR Image Reconstruction*,
   [arXiv:1704.02422](https://arxiv.org/abs/1704.02422), IEEE TMI 2018 — origin of the
   data-consistency layer in §6.

### Diffusion foundations

- Song & Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution*,
  [arXiv:1907.05600](https://arxiv.org/abs/1907.05600) (2019)
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*,
  [arXiv:2011.13456](https://arxiv.org/abs/2011.13456) (2020)
- Song, Meng, Ermon, *Denoising Diffusion Implicit Models*,
  [arXiv:2010.02502](https://arxiv.org/abs/2010.02502) (2020)

### Non-noise / deterministic degradations — most relevant cluster

- Daras et al., *Soft Diffusion: Score Matching for General Corruptions*,
  [arXiv:2209.05442](https://arxiv.org/abs/2209.05442) (2022)
- Delbracio & Milanfar, *Inversion by Direct Iteration (InDI)*,
  [arXiv:2303.11435](https://arxiv.org/abs/2303.11435) (2023) — directly addresses "why iterate
  rather than regress once," which is your issue 7
- Liu et al., *I²SB: Image-to-Image Schrödinger Bridge*,
  [arXiv:2302.05872](https://arxiv.org/abs/2302.05872) (2023) ⚠️

### Diffusion for inverse problems

- Kawar, Elad, Ermon, Song, *Denoising Diffusion Restoration Models*,
  [arXiv:2201.11793](https://arxiv.org/abs/2201.11793) (2022)
- Chung et al., *Diffusion Posterior Sampling for General Noisy Inverse Problems*,
  [arXiv:2209.14687](https://arxiv.org/abs/2209.14687) (2022)
- Song, Shen, Xing, Ermon, *Solving Inverse Problems in Medical Imaging with Score-Based Generative
  Models*, [arXiv:2111.08005](https://arxiv.org/abs/2111.08005) (2021)
- Chung & Ye, *Score-based diffusion models for accelerated MRI*,
  [arXiv:2110.05243](https://arxiv.org/abs/2110.05243), Medical Image Analysis 2022

### Undersampled reconstruction and data consistency (the closest methodological analogue)

MRI is the mature field for "reconstruct from deliberately undersampled k-space with a learned
prior + hard data consistency." Nearly every architectural choice here has an MRI antecedent.

- Lustig, Donoho, Pauly, *Sparse MRI: The application of compressed sensing for rapid MR imaging*,
  Magn. Reson. Med. **58**(6), 2007
- Hammernik et al., *Learning a Variational Network for Reconstruction of Accelerated MRI Data*,
  Magn. Reson. Med. **79**(6), 2018 ⚠️
- Sriram et al., *End-to-End Variational Networks for Accelerated MRI Reconstruction*,
  [arXiv:2004.06688](https://arxiv.org/abs/2004.06688), MICCAI 2020 ⚠️
- Jalal et al., *Robust Compressed Sensing MRI with Deep Generative Priors*,
  [arXiv:2108.01368](https://arxiv.org/abs/2108.01368), NeurIPS 2021 ⚠️
- Zbontar et al., *fastMRI: An Open Dataset and Benchmarks*,
  [arXiv:1811.08839](https://arxiv.org/abs/1811.08839) (2018)

### Classical band-limited extrapolation — what operator B reduces to

- Gerchberg, *Super-resolution through error energy reduction*, Optica Acta **21**(9), 1974
- Slepian & Pollak, *Prolate spheroidal wave functions, Fourier analysis and uncertainty — I*,
  Bell Syst. Tech. J. **40**(1), 1961 — the fundamental limits
- Bertero & Boccacci, *Introduction to Inverse Problems in Imaging*, IOP, 1998

### OCT physics and computational OCT

- Huang et al., *Optical Coherence Tomography*, Science **254**(5035), 1991
- Drexler & Fujimoto (eds.), *Optical Coherence Tomography: Technology and Applications*, Springer —
  standard reference; see the Izatt & Choma theory chapter
- Ralston, Marks, Carney, Boppart, *Interferometric synthetic aperture microscopy*,
  Nature Physics **3**, 2007 — **the classical physics-based counterpart to this work**: complex
  field, computational lateral resolution recovery
- Adie, Graf, Ahmad, Carney, Boppart, *Computational adaptive optics for broadband optical
  interferometric tomography of biological tissue*, PNAS **109**(19), 2012
- Kumar, Drexler, Leitgeb, *Subaperture correlation based digital adaptive optics for full field
  OCT*, Optics Express **21**(9), 2013 ⚠️
- Hillmann et al., *Aberration-free volumetric high-speed imaging of in vivo retina*,
  Scientific Reports **6**, 2016 ⚠️
- Goodman, *Introduction to Fourier Optics*, 4th ed. — for the diffraction-limited PSF and pupil
  bandwidth

### Complex-valued networks

- Trabelsi et al., *Deep Complex Networks*,
  [arXiv:1705.09792](https://arxiv.org/abs/1705.09792), ICLR 2018
- Cole et al., *Analysis of deep complex-valued CNNs for MRI reconstruction*,
  [arXiv:2004.01738](https://arxiv.org/abs/2004.01738) ⚠️
- Virtue, Yu, Lustig, *Better than real: Complex-valued neural nets for MRI fingerprinting*,
  ICIP 2017 ⚠️

### Architecture (as used here)

- Ronneberger, Fischer, Brox, *U-Net*,
  [arXiv:1505.04597](https://arxiv.org/abs/1505.04597), MICCAI 2015
- Liu et al., *A ConvNet for the 2020s* (ConvNeXt),
  [arXiv:2201.03545](https://arxiv.org/abs/2201.03545) (2022)
- Shen et al., *Efficient Attention: Attention with Linear Complexities*, WACV 2021

### Searching the OCT deep-learning literature

This subfield is scattered and I am least confident about exact titles here — search rather than
trust a list. Most work lives in **Biomedical Optics Express**, **Optics Letters**,
**Optics Express**, **IEEE TMI**, and **Journal of Biomedical Optics**. Useful query terms:
`OCT lateral undersampling reconstruction`, `sparse A-scan OCT deep learning`,
`OCT compressed sensing complex field`, `phase-sensitive OCT super-resolution`,
`computational OCT deep learning`. Worth checking explicitly whether anyone has already published
**complex-field** (rather than intensity) lateral super-resolution for OCT — that is your novelty
claim and it needs a clean negative result.
