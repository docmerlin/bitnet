# TODO

## Locality-Concentrated MoE (RFMoE + locality regularizer)

Goal: an MoE built for **local, memory-bound** inference (single node, VRAM/RAM/disk
hierarchy) instead of data-center load-balanced serving. Concentrate expert usage into a
small hot working set that stays VRAM-resident; offload the cold tail to RAM/disk and page
in on demand. Built on **self-gating experts** so the design is also **extensible** (append
experts later, including straight to the cold tier).

Two pieces, designed together:
1. **RFMoE** — routing-free / self-gating experts (removes the centralized router).
2. **Locality regularizer** — shape the expert-usage distribution to a tier-matched staircase
   (uniform within a memory tier, stepping down across tiers) instead of uniform-everywhere.

Context for this repo: we already do ternary weights + 4-bit activations (`layers/h_bitlinear.py`)
and z-loss (router-logit regularizer, already in git history). MoE does not exist here yet — this
is a new architecture direction, not a patch to the dense model in `model.py`.

---

### Part 1 — RFMoE (self-gating experts)

Reference: "Routing-Free Mixture-of-Experts" (Liu et al., arXiv 2604.00801).
Builds on AoE (Autonomy-of-Experts, 2501.13074) and ReMoE (ReLU routing).

Each expert decides its OWN activation from its OWN internal score — no router matrix,
no softmax, no top-k. Per expert:
- FFN is GLU-style: `FFN(x) = [σ(xA_gate B_gate) ⊙ (xW_up)] W_down`
  - `A_gate`: D×r  (down-proj, **dual use**: scoring + FFN gate branch)
  - `B_gate`: r×D_act
  - `W_up`: D×D_act, `W_down`: D_act×D
- Self-score (input-dependent): `s_i(x) = ‖x A_gate,i‖₂`  (norm of the rank-r projection)
- Gate: `G_i(x) = ReLU(s_i(x) − b_i)`   (`b_i` = learnable per-expert bias)
- Fire decision: `f_i(x) = 1{ G_i(x) ≥ θ }`  (`θ` = global threshold, also an inference knob)
- **Reuse**: `z = x A_gate` computed ONCE → its norm makes the decision, the same `z` feeds
  `B_gate` for the FFN. Skip path: if it doesn't fire, stop after the rank-r proj (cost D×r),
  never run W_up/B_gate/W_down. That's where the FLOP saving lives.

Sizing: set `r ≈ D/16` (paper: D=512→r=32, 768→48, 1024→64). Gating proj ≈ (D + D_act)·r
per expert. Decision-DEDICATED params = just the scalar `b_i` per expert + global `θ`.

Combination (no divide-by-count):
- `h = x + Σ_i G_i(x)·E_i(x)` over fired experts. Output scaled by the score `G_i` (score doubles
  as mix weight). Do NOT normalize by number of active experts — rely on (a) gates being
  magnitudes so weak experts add little, (b) downstream RMSNorm re-normalizing the residual,
  (c) the locality/balance loss pinning average activation density. Only add normalization if
  training actually destabilizes.

Training (non-differentiable `f_i`):
- Use pre-threshold `G_i(x)` as the differentiable proxy.
- Adaptive aux-loss coefficient λ driving empirical activation density ρ toward target ρ∞
  (multiplicative update: λ ← λ·(1+η)^sign(ρ−ρ∞)).
- Bias warmup: init `b_i ≈ 1e-6` so ALL experts fire early (explore, specialize), then let
  λ ramp to enforce sparsity. Avoids expert collapse.
- Use GLOBAL density target, NOT per-layer (paper: per-layer 39.44 → global 28.74 PPL).
  Lets late layers use more experts (they want to), early layers fewer.

Why RFMoE for us:
- No centralized router → nothing couples N → **trivially extensible** (append experts later).
- No router means no "keep the router in FP16" problem when quantizing.
- θ = single inference-time compute/quality knob (paper: 20× fewer activations, −31% FLOPs,
  lose <2 pts) — directly useful for memory-bound local serving.

Expected gains (from paper, ≤0.8B, OpenWebText): PPL −12% to −19% (grows with scale);
downstream +0.77pp (modest, p=0.037). Validated only ≤0.8B from scratch — UNPROVEN at larger
scale. Treat as research, not a sure win.

---

### Part 2 — Locality regularizer (tier-matched usage distribution)

Insight: load balancing is a DATA-CENTER assumption (even GPU utilization under expert
parallelism). For LOCAL memory-bound inference we want the OPPOSITE — concentrated usage so a
small hot set stays VRAM-resident and the cold tail offloads to RAM/disk.

Target distribution = **tier-matched staircase**, NOT pure Zipf:
- Uniform WITHIN each memory tier (same access cost → relative order is irrelevant).
- Step DOWN across tiers (match the frequency drop to the cost cliff VRAM→RAM→disk).
- Minimum viable version = 2 regions: Zipf-ish head + UNIFORM cold tail.

Clean target that yields this for free — a mixture:
```
π = (1−α)·Zipf(s) + α·Uniform(1/N)
```
- Head: Zipf component dominates → fine ordering for hot-tier placement.
- Tail: uniform component dominates → flat floor; every cold expert equal + kept alive.
- Crossover rank emerges from α, s (don't hand-pick it).
- Knobs: `s` = head skew (→ hot-tier size), `α` = tail floor height (→ cold-expert training
  guarantee + collapse prevention).
- Generalization: one uniform plateau per memory tier (VRAM/RAM/disk) for a true multi-tier
  staircase.

Set `s` from hardware: pick s so cumulative mass of top-M experts ≥ ~0.95, where M = experts
that fit in VRAM. → ~95% of token-work hits the resident hot set, ~5% pays a fetch. Bigger
VRAM → smaller s (flatter); smaller VRAM → larger s (sharper).

Loss:
```
L = L_LM + λ_loc · KL( π ‖ sort_desc(p) )
```
- `p` = empirical per-expert usage, tracked as an EMA (rare experts noisy per-batch).
- `sort_desc(p)`: sort descending so WHICH expert is hot emerges from data; loss shapes the
  distribution shape only (permutation-invariant). Sort is non-diff → soft-sort or
  stop-gradient through the permutation (rank assignment), gradients flow through values.
- Use `KL(π ‖ p)` DIRECTION (not KL(p‖π)): it → ∞ as any p_i → 0 while π_i > 0, so it
  simultaneously pulls toward the skewed shape AND forbids any expert going fully dead.
  The uniform component guarantees π_i ≥ α/N > 0 everywhere → well-defined floor.

Equivalent simpler implementation (reuse existing machinery): **uniform load-balance loss
WITHIN the cold-tail group + concentration term ACROSS groups.** I.e. partition the existing
aux-loss-free balancing (DeepSeek bias-nudge, but Zipf/staircase target instead of uniform).

On RFMoE specifically: the per-expert bias `b_i` IS the usage control (high b_i → cold). The
locality loss shapes the distribution of `b_i`. Newly-appended experts start with high bias →
land cold by default → consistent with the tier structure. So locality reg + extensibility +
self-gating all compose cleanly.

Curriculum (resolves capacity-vs-locality tension):
- Train FLAT early (α≈1, all experts uniform → whole population trains up, tail learns to be good).
- Anneal LATE: lower α, raise s → carve out the hot head; tail stays uniform throughout.
- Result: cold experts ARE competent (trained while flat), just rarely summoned (skewed at end).
- Mirrors RFMoE's own dense-warmup-then-sparsify.

Second, orthogonal regularizer — temporal stickiness (optional, helps caching):
- Penalize changing the active set token-to-token within a document.
- Once a cold expert is paged in for a domain, reuse it for the document → no page thrashing.
- Static Zipf = what's hot; temporal stickiness = don't thrash. Both cut disk traffic.

Inference / serving side (the payoff):
- Tier experts by learned usage: hot → VRAM, warm → RAM, cold → SSD.
- Prefetch cold experts from an early-layer routing signal to hide disk latency.
- Accept input-dependent tail latency (rare input wakes a cold expert) — fine for local
  interactive, not for SLA serving.

---

### Hard parts / open problems (be honest)
- **Functional diversity of the uniform tail.** Equal *usage* ≠ equal *function*. Tail experts
  fire equally often but must each own a distinct niche or they're redundant cold weight. Need a
  diversity/orthogonality/redundancy term (cf. R²MoE) on top of the frequency target.
- **Over-skew = effectively a smaller model.** Too-large s → cold experts vestigial, diverse-input
  quality drops. Measure rare-input quality, don't over-crank s.
- **Tail under-training despite the floor.** The very inputs that justify the tail are rare → tail
  weakly trained → bad on exactly those inputs. Curriculum (flat→skew) mitigates, doesn't fully fix.
- **Prefetch needs predictable routing.** If per-token routing is unpredictable, can't hide the disk
  fetch. Self-gating scores are computed early-in-layer (helps), cross-layer prefetch needs a predictor.
- **Scale.** RFMoE only validated ≤0.8B. Locality reg is novel/unpublished as a unified thing
  (pieces exist: aux-loss-free balancing, RFMoE warmup, MoE offload caching — nobody's aimed them
  at deliberately-Zipfian local MoE). This is a research bet.

### Prior art to read first
- RFMoE: arXiv 2604.00801 (code: github.com/liuyilun2000/RoutingFreeMoE). AoE: 2501.13074. ReMoE.
- Aux-loss-free balancing: DeepSeek (2408.15664) — the bias-nudge we'd retarget from uniform to staircase.
- Offload systems (the "exploit emergent skew" half we'd replace with "train for skew"):
  Mixtral-offloading, MoE-Infinity (activation-aware), Fiddler (CPU-GPU), Pre-gated MoE (prefetch),
  EdgeMoE/AdapMoE (edge caching).
- Memory-bound serving objective: "Balance activated experts, not tokens" (arXiv 2512.09277).
- Continual/extensible MoE (the append-experts angle): Lifelong-MoE, MoE-Adapters (CVPR 2024),
  R²MoE (2507.13107, redundancy removal), CP-MoE (2605.20247).

### Rough build order
1. Minimal RFMoE FFN block (self-gating expert, skip path, residual combine) — start dense, swap one
   FFN sublayer to MoE. Validate it trains + the θ knob works on a small run.
2. Add adaptive-λ density control + global density target + bias warmup.
3. Swap the balance target uniform → staircase (mixture π). Verify usage distribution matches and the
   tail stays alive (KL(π‖p) floor).
4. Add the flat→skew curriculum (anneal α, s).
5. Add functional-diversity term for the tail.
6. (Serving) tier experts by usage, expert offload + prefetch. Optional: temporal stickiness loss.

---

## Extensible MoE (append experts after training)

Goal: grow model capacity AFTER training by **appending new experts** — without retraining the
whole model or perturbing existing behavior. New experts learn a new domain/skill; old experts
stay frozen so old knowledge is preserved. Pairs directly with the Locality-Concentrated MoE
above: appended experts default to the COLD tier (high bias), so "extend" and "offload" are the
same mechanism.

### Why self-gating (RFMoE) is the enabler — and why standard MoE can't do this
- Standard MoE router is one matrix `G ∈ D×N`, trained for exactly N experts. Adding expert N+1:
  - softmax/top-k is over fixed N → adding a logit **renormalizes ALL existing routing** (perturbs
    every token's routing), the new router vector is untrained, load balancing breaks. Requires
    retraining the router at minimum. N is baked into the router. This is the wall.
- RFMoE self-gating has **no centralized router**. Each expert is self-contained: its own `A_gate`,
  `b_i`, plus shared global `θ`. N is not stored anywhere central.
  - Append expert N+1 = just add `{new FFN, new A_gate, new b_i}`.
  - Existing experts' fire decisions are UNCHANGED (never depended on N).
  - No softmax-over-N to renormalize. No fixed top-k. Activation is independent per expert.
  - Combination is residual-add (no divide-by-count) → a new expert just adds a residual
    correction; old behavior preserved exactly when old experts are frozen.

### Procedure to add an expert
1. Append `{FFN, A_gate (D→r), b_i}`. Init `b_i` HIGH → starts cold/rarely-firing (lands in the
   cold tier, consistent with locality reg).
2. FREEZE existing experts + attention + embeddings. Train ONLY the new expert's params on the
   new data/domain.
3. Push it into an UNCLAIMED niche (see diversity below) so it doesn't duplicate an existing expert
   or free-ride. Optionally init `A_gate` to respond to currently under-served inputs.
4. Re-tune `θ` (and/or the locality-reg target) so overall activation density stays ~constant —
   otherwise inference cost creeps up as N grows (you'd slowly rebuild a dense model).

### Hard parts / open problems
- **Niche-finding is THE problem.** Bias too high → expert never fires (dead capacity). Too low →
  fires everywhere, duplicates/destabilizes. Getting it to claim *unserved* input regions reliably
  is the core research question. Use a diversity/orthogonality/redundancy term (cf. R²MoE).
- **Density/scale drift.** More experts that CAN fire → if θ doesn't hold density constant, inference
  cost grows with N. θ must keep average activation pinned.
- **Joint-optimality loss.** Frozen old experts can't co-adapt with new ones. An extended model
  likely underperforms a same-size from-scratch jointly-trained MoE. Trade-off: extensibility vs
  joint optimality. Accept it, or periodically do a light global unfreeze/finetune.
- **Tail/new-expert under-training.** Same issue as the locality cold tail — rare-niche experts get
  little gradient. Train the new expert flat-ish first (lower its bias during its own training),
  then push it cold for deployment (per-expert version of the flat→skew curriculum).
- **Task-boundary assumption.** Easiest if you KNOW when a new domain starts (to spin up the new
  expert). Truly online "append capacity whenever, no task labels" is harder/unsolved.

### Prior art to read first
- Most existing continual-MoE work patches the ROUTER (per-task routers, freeze+extend+regularize)
  rather than using pure self-gating — our RFMoE route is cleaner but less explored:
  - Lifelong-MoE (Distribution-Specialized Experts, arXiv 2305.12281) — freeze old, add new, regularize gating.
  - MoE-Adapters / Incremental MoE-Adapters (CVPR 2024, arXiv 2403.11549) — frozen CLIP, adapters as
    experts, per-task routers added incrementally.
  - R²MoE (arXiv 2507.13107) — redundancy removal for lifelong concept learning (the niche/dedup problem).
  - CP-MoE (arXiv 2605.20247) — transient "probe" expert before merging into the stable pool.
  - LLaVA-CMoE, TRGE (2508.07738), MoTE — multi-domain / class-incremental variants.
- Note: most are adapter/LoRA-scale on frozen backbones in vision-language land, NOT full FFN experts
  in a from-scratch LLM. The self-gating-makes-it-trivial angle is the under-explored opportunity.

### Build order (after the RFMoE base from the section above exists)
1. Implement append: add an expert to an existing trained RFMoE layer, freeze the rest.
2. Train the new expert on a held-out domain; verify old-domain quality is UNCHANGED (frozen → should be).
3. Add the diversity/redundancy term so the new expert claims a distinct niche (measure overlap vs
   existing experts' activation patterns).
4. Verify density stays pinned (θ re-tune) so inference cost doesn't grow with N.
5. Compare extended model vs from-scratch same-size MoE to quantify the joint-optimality gap.

---

## Diffusion + MoE interaction (locality reg is LOAD-BEARING here)

Motivation: for LOCAL inference (low batch, idle parallel compute), **diffusion LLMs** beat AR
on latency — AR is N sequential bandwidth-bound steps; diffusion is T≪N parallel denoising steps
over a block, spending the idle compute AR wastes at batch=1. (Mercury 2: ~1000 tok/s single
Blackwell; LLaDA 2.0: MoE+diffusion at 100B; block diffusion ~32-token blocks = production norm.)
So diffusion is arguably the MOST aligned generation paradigm for the local-first thesis here.

NOTE on block size (corrected — "~32" was a research-ablation number, NOT production reality):
- DiffusionGemma + Gemini Diffusion use a 256-token canvas/block, refine ≥128 tokens/step,
  encoder-decoder (AR encoder prefills prompt + KV cache, bidirectional decoder over the 256 canvas).
  DiffusionGemma is itself 26B-A4B = MoE diffusion already (Google does MoE+diffusion).
- LLaDA semi-AR: block length 128 (ablated B ∈ {1,4,16,32,128,512}).
- Block size is the AR(B=1) ↔ full-diffusion(B=seqlen) interpolation knob; quality-vs-size is
  TASK-dependent (some tasks improve with bigger B, e.g. GSM8K 8→32). Production lands at 128–256
  for throughput. Bigger block = bigger per-step expert union = locality reg matters MORE (below).

BUT diffusion + MoE has a sharp interaction that THREATENS the locality-MoE design above:

### The conflict: per-step expert union
- MoE's LOCAL benefit (offload/bandwidth) needs a TINY active working set. AR batch=1: 1 token →
  top-k experts → touch k/E of them → small set → offload works.
- Diffusion processes the WHOLE BLOCK in parallel → many tokens/step → each routes to different
  experts → the UNION touched per step is huge.
- Math: fraction of E experts touched ≈ 1 − (1 − k/E)^N for N tokens/block.
  - AR 1 token, k=8, E=64 → 12.5% (small, offload works).
  - Diffusion 32-token block, k=8, E=64 → 1 − 0.875^32 ≈ 98.6%.
  - PRODUCTION block sizes are 128–256 (Gemma/Gemini=256, LLaDA=128), so it's worse:
    256-block, k=8, E=64 → ≈100%. Even E=256: 256-block, k=8 → ≈99.97%. ⇒ ESSENTIALLY ALL
    experts touched every step at realistic block sizes. The conflict is total, not partial.
- ⇒ A diffusion step touches ~all experts → the sparse working set is GONE → naive load-balanced
  MoE + diffusion is the WORST local combo (all experts resident every step → offload thrashes,
  step becomes memory-bound on expert weights, MoE's compute saving wasted).

### What survives vs dies
- SURVIVES: per-token FLOP saving (each token still runs only k experts) → MoE cuts the FLOP-heavy
  diffusion step. And capacity-per-FLOP matters MORE (diffusion = harder any-order/bidirectional
  task). MoE HELPS diffusion. (LLaDA 2.0 = proof.)
- DIES: the memory/bandwidth/offload saving — the whole point of the locality section above.

### The rescue — locality reg becomes ESSENTIAL (not optional) for diffusion
- Concentrated routing (the staircase target above) SHRINKS the per-step union: if ~90% of mass is
  on a small hot set, even 32 parallel tokens mostly pick the SAME hot experts → union ≈ hot set +
  few cold → small working set restored → offload viable again.
- So the locality regularizer is MORE valuable for diffusion than AR: AR already has a small
  per-step set (1 token); diffusion's parallelism would blow it up — concentration prevents that.
- Diffusion CACHING ADVANTAGE to exploit: across the T denoising steps of ONE block it's the SAME
  tokens being refined → expert working set is STABLE across those T steps. So load the hot set
  ONCE per block, reuse across all T steps (amortize loading over T). AR shifts the set per token;
  diffusion holds it per block. Pairs with the temporal-stickiness loss from the locality section.

### Levers / open issues
- Block size tension: smaller block → smaller union (good for working set) but more sequential
  steps → erodes the latency win. ~32 is the compromise; for local-offload bias smaller, trade
  latency for a tighter working set. Measure the curve.
- Block-level routing: route per-BLOCK not per-token (all tokens in a block share the expert set)
  → union stays tiny → but sacrifices per-token specialization. Test as an option.
- Routing instability early in denoising: early steps (mostly masked, little context) → routing
  noisy → touched set shifts as the block fills. "Stable working set across steps" holds LATE;
  early steps may be jittery. Measure.
- Memory-bound failure mode: if union can't be shrunk and all experts must load each step, the step
  is memory-bound on expert weights and MoE's compute saving is wasted. Concentration is NOT
  optional for local diffusion+MoE.

### The full local stack (only composes if routing is concentrated)
diffusion (cut sequential steps) + concentrated-locality MoE (keep working set VRAM-resident
despite parallelism) + ternary weights / low-bit acts (cut bandwidth). Each attacks the local
bottleneck from a different side. Naive load-balanced MoE + diffusion does NOT compose — it pays
all-experts-resident every step. Flag: the locality reg (staircase target + temporal stickiness)
is a PREREQUISITE for any diffusion variant of this model, not an add-on.

### Diffusion background to read (if we go this route)
- Masked/discrete diffusion basics: SEDD (Score Entropy Discrete Diffusion), MDLM/MD4 (simplified
  masked diffusion), D3PM. Masking (absorbing state) >> uniform corruption.
- Block diffusion / BD3-LM (Arriola et al.) — AR over blocks + diffusion within block; recovers
  KV-cache + arbitrary length. The production architecture.
- LLaDA / LLaDA 2.0 (MoE + diffusion, 100B; AR-to-diffusion conversion). DiffuLLaMA/DiffuGPT
  (adapt AR checkpoints → diffusion, cheap). Mercury 2 (Inception, commercial, ~1000 tok/s).
- Efficiency: Fast-dLLM (training-free KV cache + parallel decode), dKV-Cache, d²Cache,
  "Attention Is All You Need for KV Cache in Diffusion LLMs". Step distillation: Di[M]O (one-step),
  T3D, CDLM (consistency + block-causal). Schedules: progress-aware confidence schedules.
- Noise design: the masking SCHEDULE = the inference algorithm (sets T = latency); confidence/
  progress-aware adaptive schedules trade parallelism-per-step vs coherence; block/structured
  masking = "colored" noise (recovers ordering + KV-cache).
