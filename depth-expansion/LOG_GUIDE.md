# Training Log Interpretation Guide

> Applies to: `new-fourier-test-new.py` (Model3B / Kronecker / TSAI 131K)
> Reference: `new-fourier-test.py` (SmolLM / Fourier / GPT2) uses the same structure.

---

## Log Structure Overview

There are **4 levels** of logging, each at different frequencies:

| Level | Frequency | Content |
|---|---|---|
| 1 | Every step | Main loss + controller metrics |
| 2 | Every 10 steps | Per-layer recurrence stats |
| 3 | Every 50 steps | Survival trace + LR check |
| 4 | Every step after step 1200 | Recurrence quality / economic signal |

All output is mirrored to `log_kronecker_adaptive.txt` via the `Tee` wrapper on stdout.
A machine-readable snapshot is also written to `stats.json` every step.

---

## Level 1 — Main Step Line (every step)

```
STEP: 1500 | TOT: 3.2145 | LM: 2.9800 | PONDER: 0.000123 (RAW: 0.000456, RWD: 0.000333) | LR: 2.8e-04 | GRAD: 0.8732 | λ_p: 1.20e-03[DB] | CTRL_ERR: +0.082 | P_NR: 3.2% | MTP: 3.1200 | AUX: 0.0021 | PL/LM: 0.04% | AVGS: exec=1.43, k=1.38, gap=0.05 | dt:  420.00ms | tok/sec:    609.52
```

| Field | What it means | Healthy range |
|---|---|---|
| `TOT` | Total loss = LM + 0.3×MTP + aux + ponder | Decreasing over time |
| `LM` | Next-token prediction loss (cross-entropy). **Primary metric** | ~5→2.5 early training |
| `PONDER` | Net compute tax after coupon deduction | Small, near 0 is fine |
| `RAW` | Ponder before coupon — raw compute cost signal | — |
| `RWD` | Coupon = η × ΔCE rewarded for useful recurrence | Higher = recurrence helping |
| `LR` | Current learning rate (warmup → cosine decay) | 0 → 3e-4 → 3e-5 |
| `GRAD` | Gradient norm after clipping at 1.0. Spikes = instability | 0.3–0.9 normal; >2 = trouble |
| `λ_p` | Compute price. Cost per extra recurrence pass | Rises until model stops overpondering |
| `[DB]` | **Deadband engaged** — λ_p stabilized, asymmetric control active | Good: means training is stable |
| `CTRL_ERR` | `avg_k − 1.30` (TARGET_K). Distance from target depth | Should trend toward ~0 |
| `P_NR` | % of gate values near lower clamp (≤0.05). Absorbing state risk | >20% = collapse risk |
| `MTP` | Multi-token prediction loss | Should track LM loss roughly |
| `AUX` | MoE router auxiliary loss (expert load balancing) | ~0.001–0.05 |
| `PL/LM` | Ponder loss as % of LM loss — compute overhead | <5% is fine |
| `AVGS: exec=, k=, gap=` | Across all layers: actual passes run, expected depth, difference | `gap` near 0 = consistent |
| `dt` | Step time in milliseconds | Hardware dependent |
| `tok/sec` | Token throughput | Higher = better |

### Special prefix flags

| Prefix | Meaning |
|---|---|
| `[AUDIT]` | Step was forced to k=2 as a counterfactual. λ_p controller does **not** update on this step. |

---

## Level 2 — Per-Layer Stats (every 10 steps)

```
L0: k_exec=2, expected_k=1.38, s_fin=0.042 | tok: avg=1.41, p95=2.00 | tail: ≥2:41.20%, ≥3:5.30%, ≥4:0.10% | ent=3.1420 | p_mu=0.82, p_std=0.120, d_rel=0.043 | Δh=0.031, Δp=-0.41 | ponder=0.031, λ_p=1.20e-03
```

| Field | What it means | What to look for |
|---|---|---|
| `k_exec` | Max recurrence passes any token actually ran in this layer | 1 = no recurrence; 4 = max (Kmax) |
| `expected_k` | Soft average depth (weighted by survival prob `s`) | Target ~1.30 |
| `s_fin` | Mean survival probability at end of all passes | Near 0 = model stopped cleanly |
| `tok: avg` | Average passes per token across the batch | Should be near `expected_k` |
| `tok: p95` | 95th percentile — how deep the "hard" tokens go | — |
| `tail: ≥2/≥3/≥4` | % of tokens that needed 2+/3+/4+ passes | High ≥4 = model not learning to stop |
| `ent` | Attention entropy — how spread out attention is | Low = focused; high = diffuse |
| `p_mu` | Mean gate probability at k=1 ("willingness to continue") | Should decrease as training progresses |
| `p_std` | Variance of gate values across tokens | Higher = more token-differentiated decisions |
| `d_rel` | Relative update norm ‖Δh‖/‖h‖ — how much state changes | Near 0 = recurrence not doing work |
| `Δh` | Mean gated state delta across passes ‖s·Δh‖/‖h‖ | Measures effective compute used |
| `Δp` | p₂ − p₁ — how much the gate changes between passes | **Negative = model learning to stop after k=1** |
| `ponder` | Combined ponder signal (0–1, before λ_p scaling) | — |
| `λ_p` | Active compute price at this step | — |

---

## Level 3 — Survival Trace (every 50 steps)

```
L0 s_trace: [1.00, 0.82, 0.23, 0.04]
LR CHECK: (1500, 2.80e-04, phase=cosine)
```

### Reading `s_trace`

`s_trace` is the survival probability entering each pass: `[s_k=1, s_k=2, s_k=3, s_k=4]`.
It always starts at `1.00` (all tokens alive at k=1).

| Pattern | Interpretation |
|---|---|
| `[1.00, 0.82, 0.23, 0.04]` | Sharp drop — model deciding to stop early. **Healthy.** |
| `[1.00, 0.95, 0.90, 0.85]` | Model not stopping — ponder collapse, λ_p too low |
| `[1.00, 0.02, 0.00, 0.00]` | Stopping too aggressively after k=1 — over-penalized |
| `[1.00, 0.50, 0.50, 0.50]` | Gate stuck at 0.5 — not learning when to stop |

### LR CHECK

Confirms the LR schedule phase:
- `warmup` → step < 500
- `cosine` → 500 ≤ step ≤ 10000
- `done` → step > 10000

---

## Level 4 — Recurrence Quality (every step after step 1200)

```
RECURRENCE QUALITY: ΔCE=0.042300 | ΔH=-0.0312 | COUPON: 0.000333 [ECONOMIC WIN] [CONFIDENCE+]
```

| Field | What it means |
|---|---|
| `ΔCE` | CE₁ − CE₂: improvement in loss from k=1 → k=2+. **Positive = recurrence helped** |
| `ΔH` | Entropy change between pass 1 and final. **Negative = model became more confident** |
| `COUPON` | η × ΔCE reward subtracted from ponder penalty |

### Status labels

| Label | Meaning |
|---|---|
| `[ECONOMIC WIN]` | Policy batch: recurrence improved predictions — worth the compute cost |
| `[TAX ONLY]` | Recurrence ran but didn't improve — paying tax for nothing |
| `[AUDIT-POS]` | Counterfactual batch: forced k=2 helped (informational only) |
| `[AUDIT-NEG]` | Counterfactual batch: forced k=2 hurt (informational only) |
| `[CONFIDENCE+]` | ΔCE > 0 AND ΔH < 0 — strongest signal that recurrence is genuinely useful |

---

## Training Phases

### Phase 1: Force Warmup (steps 0–800)
- `force_2=True` — k_exec always 2, PONDER cost = 0
- Focus: LM loss dropping, model learning basic language
- Expect `LM` to drop from ~5 → ~3

### Phase 2: Gate Learning (steps 800–1200)
- `force_2=False` — gate now decides recurrence
- Watch `s_trace` start showing drops at k=2
- `Δp` should go negative (gate learning to stop)
- PONDER still = 0 (λ_p not applied yet)

### Phase 3: Economic Control (steps 1200+)
- λ_p activates — compute has a cost
- λ_p rises if `CTRL_ERR > 0` (model overpondering)
- λ_p falls if `CTRL_ERR < 0` (model underpondering)
- `[ECONOMIC WIN]` > `[TAX ONLY]` = recurrence earning its cost
- `[DB]` appearing = λ_p stabilized — deadband engaged, training working correctly

---

## Red Flags

| Symptom | Likely Cause | Action |
|---|---|---|
| `GRAD > 2.0` sustained | Exploding gradients / LR too high | Check LR schedule; reduce if needed |
| `s_trace: [1.00, 0.98, 0.97, 0.96]` | Gate collapsed to "always continue" | λ_p too low; increase `LAMBDA_MIN` or wait |
| `P_NR > 20%` | Gate collapsed to "always stop" (absorbing state) | Model overtaxed; lower `LAMBDA_MAX` |
| `ΔCE` always negative | Recurrence is actively hurting | `REWARD_ETA` too low; or recurrence not learned yet |
| `AUX > 0.1` | MoE router unbalanced (one expert dominating) | Check `num_real_experts` / aux loss weight |
| `gap` (exec − k) > 0.5 | Hard vs soft execution mismatch | `halt_thr` too low; soft survival not tracking hard stop |
| `LM` not decreasing after step 500 | Learning rate too low or data issue | Check `LR CHECK` output and dataset loading |
| `tok/sec` very low | Memory pressure causing swaps | Reduce `seq_len` or `batch_size` |

---

## stats.json Schema

Written every step. Useful for plotting with external tools.

```json
{
  "step": 1500,
  "lm_loss": 2.98,
  "mtp_loss": 3.12,
  "aux_loss": 0.002,
  "total_ponder": 0.000123,
  "delta_ce": 0.042,
  "delta_h": -0.031,
  "deadband_active": true,
  "deadband_steps_since_trigger": 42,
  "layers": [
    {
      "k": 1.38,
      "ent": 3.14,
      "p_loss": 0.000041,
      "k_exec": 2,
      "s_final": 0.042,
      "s_trace": [1.0, 0.82, 0.23, 0.04],
      "p_mean": 0.82,
      "p_std": 0.12,
      "d_rel": 0.043,
      "ponder_signal": 0.031,
      "avg_exec_token": 1.41,
      "p95_exec_token": 2.0,
      "frac_ge2": 0.41,
      "frac_ge3": 0.053,
      "frac_ge4": 0.001,
      "h_diff": 0.031,
      "p_diff": -0.41,
      "p_near_clamp": 0.032
    }
  ]
}
```
