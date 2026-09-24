# Literature review — phase-preserving CNN reconstruction of laterally undersampled complex OCT

Scope: practical review for the DLOCT thesis (see `README.md`). Goal: identify what has worked, what
to copy, and which representation / loss / metrics to use for **complex-field** reconstruction of
laterally sub-Nyquist (aliased) OCT tomograms `T = A·e^{iφ}`.

Verification policy: every entry below was located via web search / fetch in Sept 2026. Entries
marked **[?]** have a detail (author list, exact formula, venue) that could not be fully confirmed
from an accessible source — check before it enters the bibliography.

---

## 0. Executive summary — recommendations for this project

1. **The novelty gap is real.** Every OCT reconstruction paper found either (a) works on intensity
   only [Yuan 2022; Kim 2022; Guo & Zhao 2025; Chen 2026; Wang PnP-DM 2025], (b) *feeds* Re/Im or
   phase as input but supervises/evaluates **amplitude only** [Zhang 2021 LSA; Bellemo 2025], or
   (c) regresses a *derived* phase product (Doppler map) directly [Li/ASSAN 2025]. No paper found
   reconstructs the laterally undersampled **complex field** and quantitatively evaluates phase.
   Closest complex-valued OCT work is *axial* SR [Wang CVSR-Net 2023]. State this as a
   searched-negative result in the thesis.
2. **Representation: 2-channel Re/Im, real-valued U-Net** (your ConvNeXt U-Net). This is what
   E2E-VarNet, DC-CNN, CDiffMR and Zhang 2021 do. Complex-valued nets (Trabelsi 2018; CReLU per
   Cole 2021) give ~+1–2 dB and better phase in MRI, but are an ablation, not the first model.
3. **Do not predict amplitude/phase channels.** Phase is undefined where `|T|→0` and wrapped; Re/Im
   is smooth and linear, matching the linear forward operator. Derive phase only in the loss/metrics
   via `x̂·x̄`, never via `atan2` regression [Liu 2026 circular-phase; Prokudin 2018].
4. **Key identity:** `|x̂−x|² = (|x̂|−|x|)² + 2(|x̂||x| − Re(x̂x̄))`. A complex MSE *already contains*
   a phase term `|x̂||x|(1−cos Δφ)`, weighted by amplitude² — so dim tissue has almost no phase
   supervision. Fix by adding a phase term with *linear* amplitude weight (or tissue mask), not by
   switching to magnitude losses (which is what fastMRI/E2E-VarNet do — they train on magnitude
   SSIM and ignore phase entirely [Sriram 2020]).
5. **Loss (start here):** complex Charbonnier + amplitude-weighted wrapped phase `1−cos`
   + inter-A-line phase-difference (Doppler-style) term + small lateral-spectrum L1 (§6). The
   `⊥-loss` [Terpstra 2022] is a good published alternative (≈4× lower phase error in MRI) but is
   blind to π flips — pair it with a complex L1.
6. **Operator C data consistency is trivial in image space:** lateral decimation is coordinate
   selection, so the exact projection onto the measurement-consistent set is *overwrite the
   measured A-lines* (equivalently: add the passband residual equally to all K alias bins).
   Apply at inference to every model (baseline included) [Schlemper 2018; CDiffMR 2023].
7. **Baseline ladder (build in this order):** complex sinc/linear interpolation → single-shot
   Re/Im U-Net (+ DC at inference) → **amplitude-only-trained U-Net** (to *demonstrate* the phase
   destruction you claim) → 3–5-cascade DC-CNN [Schlemper 2018] → cold diffusion.
8. **Cold diffusion is defensible but secondary.** CDiffMR (cold diffusion with k-space
   undersampling degradation + DC + start-point conditioning) beat D5C5 by 1.4 dB (×8) and 2.5 dB
   (×16) on complex MRI, at 2–3% of the inference cost of noise diffusion. But deterministic cold
   diffusion is effectively an iterative learned regressor, and an unrolled DC cascade is the fairer
   competitor. Get the U-Net + DC result first; cold diffusion reuses the same net with a time input.
9. **Avoid stochastic diffusion (DPS/DDRM/score-MRI) for phase metrics under time pressure.** A
   posterior sample draws *plausible* speckle phase where the data do not constrain it, which is
   penalised by any pixel/Doppler phase metric (perception–distortion trade-off, Blau & Michaeli
   2018). Cite them as related work.
10. **Augment with a random global phase** `x ← x·e^{iθ}` (θ~U[0,2π)) on input and target. Absolute
    phase is arbitrary in OCT; this makes a real Re/Im network approximately rotation-equivariant
    and removes the "preferred phase" bias noted in the README. Also lateral/axial flips.
11. **Normalize linearly** (per-volume 99.9th-percentile `|T|`), never log-compress the network
    input — the forward operator is linear. Compute amplitude metrics in dB afterwards.
12. **Metrics:** PSNR/SSIM on log-amplitude (tissue mask); complex NRMSE; global and local complex
    correlation `|ρ|`; amplitude-weighted circular phase error; **Kasai inter-A-line phase-difference
    error** (Doppler proxy; Zhao 2000, Liu 2012); speckle contrast / Rayleigh fit (Goodman); MPS
    half-width recovery (README Part I). Report phase metrics at several amplitude thresholds.
13. **Mind the physics floor:** inter-A-line phase difference decorrelates with lateral step even
    in fully-sampled data (Vakoc 2009). Report the fully-sampled GT's own phase-difference noise
    as the reference, and evaluate Doppler-type metrics at the *original* A-line pitch.
14. **Closest methodological analogue for "phase is the product": 4D-flow MRI** (FlowVN,
    Vishnevskiy 2020; FlowMoDL 2026 [?]) — velocity is a phase difference, and they explicitly
    add velocity/angular error terms because magnitude-driven training biases phase. Cite this to
    justify your phase-difference loss.

---

## 1. Deep learning for complex / phase-preserving OCT reconstruction

### 1.1 Undersampled OCT (spectral or lateral)

**Zhang Y., Liu T., Singh M., …, Ozcan A. (2021).** *Neural network-based image reconstruction in
swept-source optical coherence tomography using undersampled spectral data.* Light: Science &
Applications 10:155. https://doi.org/10.1038/s41377-021-00594-7 · arXiv:2103.03877
Undersamples the *spectral* (axial) data 2–3×, producing aliased complex images; a residual U-Net
takes **Re and Im of the aliased image as two channels** and is trained with **L1 against the
ground-truth amplitude only**. Metrics PSNR/SSIM on amplitude; output phase is **not evaluated**.
*Relevance:* nearest "aliasing removal from complex OCT" precedent and the direct evidence for the
thesis motivation — Re/Im in, amplitude out, phase discarded. Your ×K lateral problem is the lateral
analogue; copy its 2-channel input, differ by supervising and evaluating the complex output.

**Li Z. et al. (2025).** *Sparse Reconstruction of Optical Doppler Tomography with Alternative State
Space Model and Attention (ASSAN).* MICCAI 2025. arXiv:2404.17484
Keeps every 2nd/4th raw A-scan (exactly your lateral decimation, ×2/×4), feeds magnitude + phase
channels, and regresses the **Doppler (phase-difference) B-scan directly** with pixel L2; PSNR/SSIM
on Doppler images vs SwinIR/HAT/MambaIR baselines. *Relevance:* the closest task match found. It
side-steps complex-field recovery by predicting the derived product; you can argue that recovering
the field is more general (one model → Doppler, OCTA, OCE) and use its ×2/×4 protocol and its
Doppler-image PSNR as an additional metric. Note it uses amplitude/phase channels (not Re/Im).

**Guo Z., Zhao Z. (2025).** *Hybrid attention structure preserving network for reconstruction of
under-sampled OCT images.* Scientific Reports. https://www.nature.com/articles/s41598-024-82812-x ·
arXiv:2406.00279
Super-resolves B-scans undersampled along the fast (lateral) axis using channel/spatial attention
and a high-frequency "texture" branch; retinal DME data. Intensity images. *Relevance:*
intensity-only lateral-undersampling baseline to cite; its high-frequency branch motivates the
lateral-spectrum loss.

**Chen Y., Chen S., Song J., Ma D., Beg M.F., Mammo Z., Ju M.J. (2026).** *Deep-learning–based
optical coherence tomography reconstruction for high-speed and contrast morphology and vasculature
imaging.* J. Biomed. Opt. 31(2):025001. https://doi.org/10.1117/1.JBO.31.2.025001
VM-UNet (VMamba) reconstructs OCT/OCTA from acquisitions with alternate A-scans (2× lateral);
float amplitude data, Huber + SSIM + perceptual losses; CNR and MS-SSIM. No phase. *Relevance:*
recent lateral-2× precedent; shows the field still ignores phase even for OCTA.

**Kim G. et al. (2022).** *Integrated deep learning framework for accelerated optical coherence
tomography angiography.* Scientific Reports 12:1289. https://doi.org/10.1038/s41598-022-05281-0
Two-stage adversarial network restores angiograms from 2–8× laterally downsampled, 2-repeat
acquisitions (claimed 16–256× acceleration). Operates on angiograms, not complex data.
*Relevance:* shows the OCTA community undersamples laterally; your complex approach would let OCTA
be computed downstream (e.g. CDV) instead of learned end-to-end.

**Wang Y., Yu J., Guo W., Sun Y., Kang J.U. (2025).** *Super-Resolution Optical Coherence Tomography
Using Diffusion Model-Based Plug-and-Play Priors.* arXiv:2505.14916
Plug-and-play diffusion prior with MCMC posterior sampling for sparse corneal B-scans, compared with
a 2D U-Net. Intensity B-mode. *Relevance:* the only diffusion + data-consistency OCT SR work found;
intensity-only, so it supports the novelty claim.

**Lebed E., Mackenzie P.J., Sarunic M.V., Beg M.F. (2010).** *Rapid volumetric OCT image acquisition
using compressive sampling.* Optics Express 18(20):21003–21012.
First CS reconstruction of OCT with randomly omitted A-lines/B-scans. *Relevance:* classical
(pre-DL) lateral-undersampling baseline; note that random omission is incoherent, whereas your
regular decimation produces coherent aliasing — the harder case.

### 1.2 Complex-valued / phase-aware OCT networks

**Wang L., Chen S., et al. (2023) [?authors].** *Axial super-resolution optical coherence tomography
via complex-valued network (CVSR-Net).* Physics in Medicine & Biology 68.
https://doi.org/10.1088/1361-6560/ad0997
Complex-valued SR network using amplitude and phase for **axial** SR; beats its real-valued
counterpart in depth resolving. A follow-up (Cv-EDSR, SPIE Proc. 12830, 2024) reports similar gains.
*Relevance:* the closest complex-valued OCT precedent — cite as "axial, not lateral; phase used as
information but not reported as an output fidelity metric" (check the paper for any phase figure).

**Bellemo V., Haindl R., Pramanik M., Liu L., Schmetterer L., Liu X. (2025).** *Complex conjugate
removal in optical coherence tomography using phase aware generative adversarial network.*
J. Biomed. Opt. 30(2):026001. https://doi.org/10.1117/1.JBO.30.2.026001
Adding the phase map as an input channel improves conjugate-artifact removal, but authors state
explicitly "we did not evaluate the reconstruction of phase information". *Relevance:* quotable
evidence that phase helps as input yet is not delivered as output.

**Yuan Z., Yang D., Yang Z., Zhao J., Liang Y. (2022).** *Digital refocusing based on deep learning
in optical coherence tomography.* Biomed. Opt. Express 13(5):3005–3020.
https://doi.org/10.1364/BOE.453326
GAN (RRDB + RFB) refocusing of *en face intensity* images; LPIPS/SNR/MOS, no phase. *Relevance:*
contrasts with physics-based complex refocusing (ISAM/CAO) which needs the phase your model must
preserve.

**Ralston T.S., Marks D.L., Carney P.S., Boppart S.A. (2007).** *Interferometric synthetic aperture
microscopy.* Nature Physics 3:129–134. / **Adie S.G. et al. (2012).** *Computational adaptive optics
for broadband optical interferometric tomography of biological tissue.* PNAS 109(19):7175–7180.
Classical complex-field computational lateral resolution recovery / aberration correction; both
require a phase-stable, adequately sampled complex field. *Relevance:* a strong downstream test —
if your reconstruction preserves phase, CAO/refocusing applied to it should work (stretch goal).

### 1.3 Deep learning for functional (phase-based) OCT

**Jiang Z. et al. (2020).** *Comparative study of deep learning models for optical coherence
tomography angiography.* Biomed. Opt. Express 11(3):1580. Compares single-path, U-shaped, GAN and
multi-path models for OCTA reconstruction; U-shaped and multi-path best; names phase information as
an open improvement direction. *Relevance:* supports U-Net choice; phase is still unused.

**OCE:** *Deep-learning-based approach for strain estimation in phase-sensitive optical coherence
elastography*, Optics Letters 46(23):5914 (2021) [?authors], trains a CNN on simulated wrapped
phase / phase-gradient maps to output strain; *4D deep learning for real-time volumetric OCE*
(2020, PubMed 32997312 / PMC7822782 [? authors and venue — check]) feeds phase-difference volumes to 4D CNNs.
*Relevance:* downstream OCE consumes **phase differences**, reinforcing the inter-A-line / temporal
phase-difference loss and metric.

**Liu G., Lin A.J., Tromberg B.J., Chen Z. (2012).** *A comparison of Doppler optical coherence
tomography methods.* Biomed. Opt. Express 3(10):2669–2680. https://doi.org/10.1364/BOE.3.002669
Compares phase-resolved colour Doppler, phase-resolved Doppler variance and intensity-based Doppler
variance. *Relevance:* defines the Doppler estimators to use as evaluation operators.

### 1.4 Reviews
**Rivenson Y., Wu Y., Ozcan A. (2019).** *Deep learning in holography and coherent imaging.* LSA 8:85.
**Fanous M.J. et al. / Ozcan group (2024) [?authors].** *Neural network-based processing and
reconstruction of compromised biophotonic image data.* LSA 2024, arXiv:2403.14324 — review of
purposeful undersampling in biophotonics (includes Zhang 2021).

---

## 2. Undersampled MRI reconstruction — the closest analogue

**Zbontar J. et al. (2018).** *fastMRI: An Open Dataset and Benchmarks for Accelerated MRI.*
arXiv:1811.08839. Defines the benchmark and the U-Net baseline (image-domain de-aliasing of
zero-filled input). Metrics NMSE/PSNR/SSIM **on magnitude**. *Relevance:* your single-shot U-Net is
exactly this baseline; its reported gap to E2E-VarNet (e.g. ×8 knee: 34.7 vs 36.9 dB) quantifies
what DC/unrolling typically buys.

**Schlemper J., Caballero J., Hajnal J.V., Price A.N., Rueckert D. (2018).** *A Deep Cascade of
Convolutional Neural Networks for Dynamic MR Image Reconstruction.* IEEE TMI 37(2):491–503.
https://doi.org/10.1109/TMI.2017.2760978 · arXiv:1704.02422
Interleaves small CNNs with a closed-form DC layer (hard or noise-weighted replacement of measured
k-space), Re/Im 2-channel, trained end-to-end *with DC inside the loop*. *Relevance:* origin of your
DC step; the D5C5 cascade is the natural "iterative but not diffusion" competitor. Note: unrolled
nets do train through DC successfully — the README's "never DC in training" applies to a *final*
projection on a single-shot net, not to cascades.

**Hammernik K., Klatzer T., Kobler E., Recht M.P., Sodickson D.K., Pock T., Knoll F. (2018).**
*Learning a variational network for reconstruction of accelerated MRI data.* MRM 79(6):3055–3071.
https://doi.org/10.1002/mrm.26977 — unrolled gradient descent with learned regularizer.

**Sriram A., Zbontar J., Murrell T., Defazio A., Zitnick C.L., Yakubova N., Knoll F., Johnson P.
(2020).** *End-to-End Variational Networks for Accelerated MRI Reconstruction.* MICCAI 2020.
arXiv:2004.06688. Cascades of U-Nets with soft k-space DC; complex data as 2 channels; **trained
with SSIM on the magnitude image** — phase is unsupervised. *Relevance:* strongest classical
architecture; also a cautionary example of magnitude-only training.

**Cole E., Cheng J., Pauly J., Vasanawala S. (2021).** *Analysis of deep complex-valued
convolutional neural networks for MRI reconstruction and phase-focused applications.* MRM
86(2):1093–1109. https://doi.org/10.1002/mrm.28733 · arXiv:2004.01738
At equal parameter count, complex-valued unrolled nets beat 2-channel real nets (knee PSNR 36.1 vs
34.2); CReLU best activation, zReLU close, modReLU/cardioid worse; evaluated on **phase tasks**
(fat–water separation, phase-contrast flow peak velocity) where complex nets clearly won. L1 loss.
*Relevance:* the best evidence for a complex-valued ablation; tells you to use CReLU (= ReLU on Re
and Im separately — trivial in your code) if you try it.

**Virtue P., Yu S.X., Lustig M. (2017).** *Better than real: Complex-valued neural nets for MRI
fingerprinting.* ICIP 2017, 3953–3957. arXiv:1707.00070 — introduces the cardioid activation.

**Wang S., Cheng H., Ying L., Xiao T., Ke Z., Liu X., Zheng H., Liang D. (2020).** *DeepcomplexMRI:
Exploiting deep residual network for fast parallel MR imaging with complex convolution.* Magnetic
Resonance Imaging 68:136–147. arXiv:1906.04359 — complex convolutions + repeated DC.

**Terpstra M.L., Maspero M., Sbrizzi A., van den Berg C.A.T. (2022).** *⊥-loss: A symmetric loss
function for magnetic resonance imaging reconstruction and image registration with deep learning.*
Medical Image Analysis 80:102509. https://doi.org/10.1016/j.media.2022.102509
Shows that complex L1/L2 are asymmetric in the magnitude/phase landscape and bias toward
**underestimated magnitude**. Proposes `⊥(x,y) = P(x,y) + ℓ1(|x|,|y|)` with
`P = |Re(x)Im(y) − Im(x)Re(y)| / |·|` — the perpendicular distance from one vector to the line of
the other, i.e. `amplitude·|sin Δφ|` [? which vector normalizes; ISMRM 2021 abstract gives `/|y|`].
Reported ≈4× lower phase error, higher SSIM and ~2× faster convergence than L2. *Relevance:* the
most-cited complex-domain loss; use as an ablation. Caveat: `|sin Δφ|` is zero at Δφ = π, so it
cannot by itself distinguish a sign flip — keep a complex L1/Charbonnier term alongside.

**Vishnevskiy V., Walheim J., Kozerke S. (2020).** *Deep variational network for rapid 4D flow MRI
reconstruction (FlowVN).* Nature Machine Intelligence 2:228–235.
https://doi.org/10.1038/s42256-020-0165-6 · arXiv:2004.09610 — unrolled VN for undersampled
phase-contrast MRI where velocity = phase difference between encodings; validated on velocity.

**Gottwald T. et al. (2026) [preprint].** *FlowMoDL: Model-Based Deep Learning with
Conjugate-Gradient Data Consistency for Highly Accelerated 4D Flow MRI Reconstruction.*
arXiv:2608.25828. Composite loss explicitly penalizing velocity-magnitude and angular errors
(curriculum-scheduled) because generic magnitude-driven training biases phase; metrics SSIM, nRMSE,
relative velocity error, angular error. *Relevance:* direct justification for a phase-difference
loss term. Exact formulas not in abstract [?].

**Chinese-language work [?]:** *Deep parallel MRI reconstruction based on a complex-valued loss
function* (PubMed 36651242) — weighted sum of magnitude MSE and phase MSE. Mention only if needed.

---

## 3. Representations and losses for phase

### 3.1 Complex-valued networks
**Trabelsi C. et al. (2018).** *Deep Complex Networks.* ICLR 2018. arXiv:1705.09792 — complex
convolution, complex batch-norm (2×2 whitening of Re/Im), complex init; CReLU/modReLU/zReLU.
**Arjovsky M., Shah A., Bengio Y. (2016).** *Unitary Evolution Recurrent Neural Networks.* ICML
2016, 1120–1128. arXiv:1511.06464 — origin of modReLU.
*Relevance:* a complex conv is a 2-channel real conv with tied weights `[[A,−B],[B,A]]` — so a
complex network is a *constrained* Re/Im network that is exactly equivariant to global phase
rotation (with modReLU; CReLU is not exactly). Random global-phase augmentation gives a Re/Im net
most of that benefit at zero implementation cost; do the complex net only as an ablation (Cole 2021).

### 3.2 Why not amplitude/phase channels
Wrapped phase regressed with L1/L2 has discontinuities at ±π and is meaningless where `|T|≈0`.
**Liu C.Y., Cheng J., Chen C.-C., Shu S.F. (2026) [preprint].** *Circular Phase Representation and
Geometry-Aware Optimization for Ptychographic Image Reconstruction.* arXiv:2604.26664 — represent
phase as (cos, sin) on the unit circle with a geodesic loss; better mid/high-frequency phase than
raw phase regression. **Prokudin S., Gehler P., Nowozin S. (2018).** *Deep Directional Statistics:
Pose Estimation with Uncertainty Quantification.* ECCV 2018. arXiv:1805.03430 — von Mises
likelihood / cosine losses for angles. *Relevance:* Re/Im already *is* the unit-circle
representation scaled by amplitude; the loss-level analogue is `1 − cos Δφ` computed as
`1 − Re(x̂x̄)/(|x̂||x|)` (von Mises NLL with fixed κ, up to constants).

### 3.3 Frequency-domain losses
**Jiang L., Dai B., Wu W., Loy C.C. (2021).** *Focal Frequency Loss for Image Reconstruction and
Synthesis.* ICCV 2021. arXiv:2012.12821 — spectrum-distance loss adaptively up-weighting hard
frequencies. *Relevance:* your problem is literally "recover the folded lateral band"; a small
lateral-FFT loss directly targets it (use the complex FFT; no conjugate symmetry).

### 3.4 Phase unwrapping and phase retrieval (what transfers)
**Spoorthi G.E., Gorthi S., Gorthi R.K.S.S. (2019).** *PhaseNet: A deep convolutional neural network
for two-dimensional phase unwrapping.* IEEE Signal Process. Lett. 26(1):54–58 — unwrapping as
wrap-count segmentation. **Wang K., Li Y., Kemao Q., Di J., Zhao J. (2019).** *One-step robust deep
learning phase unwrapping.* Optics Express 27(10):15100–15115. https://doi.org/10.1364/OE.27.015100.
**Wu C. et al. (2020) [?authors].** *Phase unwrapping based on a residual en-decoder network for
phase images in Fourier domain Doppler OCT.* Biomed. Opt. Express (PMC7173896).
*Transfer:* little — you should **never unwrap**; stay in the complex domain. Cite only to explain
why the naive "regress phase" route is avoided.

**Rivenson Y., Zhang Y., Günaydın H., Teng D., Ozcan A. (2018).** *Phase recovery and holographic
image reconstruction using deep learning in neural networks.* LSA 7:17141. arXiv:1705.04286 —
CNN outputs Re/Im of the complex field from a back-propagated hologram (twin-image removal).
**Liu T., de Haan K., Rivenson Y., Wei Z., Zeng X., Zhang Y., Ozcan A. (2019).** *Deep learning-based
super-resolution in coherent imaging systems.* Sci. Rep. arXiv:1810.06611 — GAN super-resolution
of complex holographic images (pixel- and diffraction-limited). **Cherukara M.J. et al. (2020).**
*AI-enabled high-resolution scanning coherent diffraction imaging (PtychoNN).* Appl. Phys. Lett.
117:044103 — two-headed encoder–decoder predicting amplitude and phase. **Wang K., …, Lam E.Y.
(2024).** *On the use of deep learning for phase recovery.* LSA 13:4. arXiv:2308.00942 — review.
*Transfer:* holography's standard practice is Re/Im (or amp+phase) channels with pixel losses;
the complex-field SR paper (Liu 2019) is the nearest "coherent SR" analogue to cite.

---

## 4. Cold diffusion and diffusion for inverse problems

**Bansal A., Borgnia E., Chu H.-M., Li J., Kazemi H., Huang F., Goldblum M., Geiping J., Goldstein T.
(2023).** *Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise.* NeurIPS 2023.
arXiv:2208.09392 — deterministic degradations (blur, masking, downsampling) with the TACoS update
`x_{t−1} = x_t − D(x̂0,t) + D(x̂0,t−1)`. *Relevance:* your method.

**Huang J., Aviles-Rivero A.I., Schönlieb C.-B., Yang G. (2023).** *CDiffMR: Can We Replace the
Gaussian Noise with K-Space Undersampling for Fast MRI?* MICCAI 2023. arXiv:2306.14350
**The template to copy.** Cold diffusion whose degradation is k-space undersampling (linear/log
sampling-rate schedules), time-embedded U-Net trained with L2 on **2-channel complex** images;
*Starting-Point Conditioning* (begin the reverse process at the t whose sampling rate matches the
input) and *Data-Consistency Conditioning* (DC at every reverse step). Beats D5C5 (×8: 27.35 vs
25.99 dB; ×16: 25.83 vs 23.35 dB), comparable to DiffuseRecon at 1.6–3.4% of its inference time;
one model serves several acceleration factors. *Relevance:* validates your design, gives you a log
schedule (fewer no-op steps — cf. README §5 width constraint) and a publishable comparison table
format. Your contribution vs CDiffMR: aliasing (fold-in) operator instead of mask, and phase.

**Delbracio M., Milanfar P. (2023).** *Inversion by Direct Iteration: An Alternative to Denoising
Diffusion for Image Restoration (InDI).* TMLR. arXiv:2303.11435 — small-step iterative restoration
avoids regression-to-the-mean; one big step gives best PSNR, many small steps best perceptual
quality. *Relevance:* explains what to expect: cold diffusion may *not* beat the U-Net on PSNR even
if it looks sharper; decide the headline metric in advance.

**Blau Y., Michaeli T. (2018).** *The Perception-Distortion Tradeoff.* CVPR 2018. arXiv:1711.06077
— no estimator can be optimal in both distortion (PSNR/phase error) and realism. *Relevance:* the
formal reason stochastic samplers lose on pixel-wise phase metrics.

**Stochastic diffusion / score priors with DC (related work):**
- **Chung H., Ye J.C. (2022).** *Score-based diffusion models for accelerated MRI.* Medical Image
  Analysis 80:102479. arXiv:2110.05243 — score prior trained on magnitude, applied separately to
  Re and Im with DC; robust to sampling patterns. Code: github.com/hyungjin-chung/score-MRI.
- **Song Y., Shen L., Xing L., Ermon S. (2022).** *Solving Inverse Problems in Medical Imaging with
  Score-Based Generative Models.* ICLR 2022. arXiv:2111.08005.
- **Jalal A., Arvinte M., Daras G., Price E., Dimakis A.G., Tamir J.I. (2021).** *Robust Compressed
  Sensing MRI with Deep Generative Priors.* NeurIPS 2021. arXiv:2108.01368 — Langevin posterior
  sampling; robust to distribution shift.
- **Chung H., Kim J., McCann M.T., Klasky M.L., Ye J.C. (2023).** *Diffusion Posterior Sampling for
  General Noisy Inverse Problems (DPS).* ICLR 2023. arXiv:2209.14687.
- **Kawar B., Elad M., Ermon S., Song J. (2022).** *Denoising Diffusion Restoration Models (DDRM).*
  NeurIPS 2022. arXiv:2201.11793 — SVD-based, natural for your linear fold operator.
- Frequency-domain relatives (from README, not re-verified here): Hoogeboom & Salimans, *Blurring
  Diffusion Models*, arXiv:2209.05557; Rissanen et al., *Inverse Heat Dissipation*,
  arXiv:2206.13397; Mirza et al., *Learning Fourier-Constrained Diffusion Bridges for MRI
  Reconstruction*, arXiv:2308.01096.

**Honest assessment (time pressure).**
- Deterministic cold diffusion + DC is a *learned iterative reconstruction* with a time-conditioned
  shared network; it is conceptually close to an unrolled DC cascade trained without
  backpropagation through iterations. Committees will ask "why not DC-CNN?" — include one.
- Expected gains are modest in distortion metrics (CDiffMR: +1.4–2.5 dB over D5C5 on MRI; InDI:
  iterative helps perceptual more than PSNR). Under operator C, K is fixed per model — CDiffMR's
  multi-rate benefit only helps if you train one model for K∈{2,4,8}.
- Cost: training ≈ same as the U-Net (random t); inference T× slower. Implementation risk is in
  the sampler/DC, not the network. Recommended plan: single-shot U-Net + DC results first (these
  alone are publishable given the phase novelty), then cold diffusion with a log-rate schedule and
  start-point conditioning as the second contribution.
- Avoid stochastic samplers (DPS/DDRM/score) as a main method: they need an unconditional complex
  OCT prior (six volumes is thin), are slow, and their samples hallucinate speckle phase.

---

## 5. Evaluation metrics for phase fidelity

**Speckle statistics.** **Goodman J.W. (2007).** *Speckle Phenomena in Optics.* Roberts & Co. —
fully developed speckle: Rayleigh amplitude, uniform phase, intensity contrast `σ_I/μ_I = 1`.
Smoothed (MMSE) reconstructions drop contrast below 1 — a quick blur/hallucination diagnostic.

**Phase-resolved Doppler.** **Zhao Y., Chen Z., Saxer C., Xiang S., de Boer J.F., Nelson J.S. (2000).**
*Phase-resolved optical coherence tomography and optical Doppler tomography for imaging blood flow
in human skin with fast scanning speed and high velocity sensitivity.* Optics Letters 25(2):114–116.
Doppler = phase difference between adjacent A-lines; Kasai estimator
`Δφ(z,x) = arg Σ_{window} T(z,x+1)·T*(z,x)`.

**Phase-decorrelation floor.** **Vakoc B.J., Tearney G.J., Bouma B.E. (2009).** *Statistical
properties of phase-decorrelation in phase-resolved Doppler optical coherence tomography.* IEEE TMI
28(6):814–821. https://doi.org/10.1109/TMI.2009.2012891 — phase-difference noise from lateral
scanning/SNR. Use to interpret the noise floor of your inter-A-line metric.

**OCTA from complex data.** **Nam A.S., Chico-Calero I., Vakoc B.J. (2014).** *Complex differential
variance algorithm for optical coherence tomography angiography.* Biomed. Opt. Express
5(11):3822–3832 — uses intensity *and* phase changes. **Fingler J., Schwartz D., Yang C., Fraser S.E.
(2007).** *Mobility and transverse flow visualization using phase variance contrast with spectral
domain OCT.* Optics Express 15(20):12636. *Relevance:* if repeated B-scans exist in your data,
CDV/phase-variance computed on reconstructions vs GT is the most convincing functional metric.

**MRI phase metrics** used by Cole 2021 (fat–water, flow velocity), Terpstra 2022 (phase error),
FlowMoDL (relative velocity error, angular error) — adopt the same "task-derived" philosophy.

Definitions to implement (x = GT, x̂ = reconstruction, sums over a tissue mask Ω):

| Metric | Definition |
|---|---|
| Complex NRMSE | `‖x̂−x‖₂ / ‖x‖₂` |
| Global complex correlation | `ρ = Σ x̂ x̄ / sqrt(Σ|x̂|² Σ|x|²)`; report `|ρ|` and `arg ρ` (global phase bias) |
| Local complex correlation | same in 5×5 (or 3×7 axial×lateral) windows; report mean `|ρ_loc|` over Ω |
| Amplitude-weighted phase error | `E_φ = Σ w·|wrap(φ̂−φ)| / Σ w`, `w = |x|²` (and w = mask only); report at amplitude thresholds (e.g. >noise+10 dB) |
| Circular phase RMSE | `sqrt(2(1 − Σ w cos Δφ / Σ w))` |
| Doppler phase-difference error | `Δφ` via Kasai (window 3×3) on x and x̂ at original pitch; `Σ w_d |wrap(Δφ̂−Δφ)| / Σ w_d`, `w_d = |Σ T(x+1)T*(x)|` |
| Phase-difference noise | std of `Δφ` in static tissue; compare to GT's own value (Vakoc floor) |
| Amplitude fidelity | PSNR / SSIM on 20·log10|x| (dB, fixed dynamic range, tissue mask) |
| Speckle | contrast `σ_I/μ_I` in homogeneous ROIs; KS/KL vs Rayleigh; phase histogram uniformity |
| Aliasing undone? | lateral MPS HWHM of x̂ vs x (README Part I tooling); spectral error in the folded band only |
| Data consistency | `‖A x̂ − y‖ / ‖y‖` (≈0 after DC) |

---

## 6. Recommended recipe for first results

**Data / representation**
- Complex tomogram → per-volume scale `s = P99.9(|T|)`; `x = T/s` (linear; no log).
- Tensor `R^{2×H×W}` (Re, Im); patches e.g. 256 (z) × 256 (x), `W` divisible by K.
- Split **by volume** (e.g. 4 train / 1 val / 1 test, or leave-one-volume-out if time allows).
- Augment: global phase `x·e^{iθ}`, θ~U[0,2π) (same θ for input and target); lateral & axial flips.
- Input: measurement on the fine grid (README §4 `D(x,T)`, sinc-interpolated). Optionally add a
  third channel = indicator of measured A-lines. Output with a global residual:
  `x̂ = input + Net(input)`.

**Architecture**
- Existing ConvNeXt U-Net, 2-in/2-out; drop time embedding for the baseline (keep for cold
  diffusion). Avoid per-channel affine in norms that treats Re and Im differently (README issue 11);
  GroupNorm/LayerNorm over all channels is fine.

**Loss** (all means over pixels; ε = 1e-3 in normalized units; δ = 1e-6)

```
L = L_c + λ_φ·L_φ + λ_Δ·L_Δ + λ_F·L_F

L_c  = mean sqrt(|x̂ − x|² + ε²)                                     # complex Charbonnier (rotation-invariant)
L_φ  = Σ w (1 − Re(x̂ x̄)/(|x̂||x| + δ)) / Σ w ,  w = |x| (or tissue mask)   # amplitude-weighted wrapped phase, no atan2
d(u) = u[:, 1:] · conj(u[:, :-1])                                    # inter-A-line phasor (Doppler-like), optionally box-filtered 3×3
L_Δ  = Σ |d(x)| (1 − Re(d(x̂) d(x)*)/(|d(x̂)||d(x)| + δ)) / Σ |d(x)|
L_F  = mean |F_lat(x̂) − F_lat(x)|                                    # complex lateral FFT (unitary), L1 on complex difference modulus
```

Starting weights (heuristics, not from literature — sweep once over {0.03, 0.1, 0.3}):
`λ_φ = 0.1, λ_Δ = 0.1, λ_F = 0.01`. Rationale: with `|x|≲1`, `L_c` is O(0.01–0.1) while phase terms
are in [0,2]; keep phase terms from dominating early (optionally warm up λ_φ, λ_Δ from 0 over the
first ~10% of steps — cf. FlowMoDL's curriculum). Ablations: (i) `L_c` only, (ii) magnitude-only L1
(to show phase destruction), (iii) ⊥-loss + L_c [Terpstra 2022], (iv) complex-valued CReLU net
[Cole 2021].

Why the phase term is needed: `|x̂−x|² = (|x̂|−|x|)² + 2(|x̂||x| − Re(x̂x̄))` — the complex loss
already supervises phase but with weight ∝ amplitude², so it neglects dim tissue; `L_φ` re-weights
to ∝ amplitude¹, and `L_Δ` supervises the quantity functional OCT actually uses.

**Data consistency (inference, every model)**
- Operator C (decimation): measured A-lines are exactly known on the kept grid, so the projection
  onto `{z : A z = y}` is: replace `x̂[:, ::K]` (the measured positions, respecting the offset used)
  by the measured samples; equivalently in the lateral spectrum, add the passband residual
  `r[κ] = Y[κ] − (A x̂)[κ]` to each of the K alias bins `κ + pW/K` (since `AA^H = I/K` up to the
  operator's scale — verify numerically with the README test harness).
- For noisy data use a soft version `x̂ ← x̂ + μ·A^+(y − A x̂)`, μ∈[0.5,1] (Schlemper 2018).

**Training**
- AdamW, lr 2e-4 with cosine decay, batch 8–16 patches, EMA 0.999, ~100–200k iterations;
  mixed precision off for the FFT/phase terms (or cast them to fp32).
- K = 2 and 4 first (K* from the MPS analysis decides whether K=2 is even aliased).

**Report (minimum table per K)**: interpolation baseline, U-Net amplitude-loss, U-Net full loss,
U-Net full loss + DC, (DC-CNN), (cold diffusion + DC) × {log-amp PSNR/SSIM, complex NRMSE, local
`|ρ|`, weighted phase error, Doppler Δφ error, speckle contrast, MPS HWHM error}.

---

## 7. Search notes / negative results
- Queries run: "sparse A-line OCT deep learning lateral undersampling complex", "complex-valued OCT
  tomogram subsampling phase", "phase-sensitive OCT deep learning complex correlation", "OCT lateral
  super-resolution complex field ISAM neural network", "deep learning Doppler OCT sparse lateral".
- No work found that reconstructs the laterally undersampled complex OCT field and evaluates phase.
  Nearest: Zhang 2021 (spectral aliasing, amplitude-only supervision), ASSAN 2025 (sparse A-scans →
  Doppler map directly), CVSR-Net 2023 (complex, axial). Re-run a Google Scholar check just before
  submission.
- "DLOCT" as a named prior work was not found.
