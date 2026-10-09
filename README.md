# ERUF — Evidence-guided multimodal Representation with Uncertainty-aware Fusion

ERUF learns a **single patient-level evidence-guided representation** from multimodal clinical data under
arbitrary modality missingness, and produces **three** quantities from one model:

1. **Risk probability** `p = σ(f(h))` — PH early-screening / risk-prediction score;
2. **Evidence sufficiency** `C = σ(f_C([h; a; |S_t|/M]))` — how much the currently
   *available* modality set supports the decision (drives calibration / error detection);
3. **Derived per-modality incremental value** `Δ_m` — the marginal contribution of adding
   modality `m`, obtained by `do(M_m = 0)`-style intervention on the same state.

A **post-hoc rectifier** `g(δ)` re-weights the risk estimate by evidence sufficiency.

The model class is `ERUFModel` in [`models/eruf.py`](models/eruf.py). Missingness is
explicit and never 0-filled-and-ignored: a missing-modality mask and a mask embedding
(`MaskEmb`, the δ-embedding) enter the fusion path, and missing slots are compensated in
latent space rather than the input space.
# eruf
