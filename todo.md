# TODO

Direction: MoE for **local, memory-bound** inference (single node, VRAM/RAM/disk),
not data-center load-balanced serving. Concentrate expert usage into small hot set,
stay VRAM-resident; offload cold tail. Self-gating experts → also extensible
(append experts to cold tier). Repo already got ternary weights + 4-bit acts
(`layers/h_bitlinear.py`) + logit z-loss.

## Status

Shipped (see `layers/rfmoe.py`, `train.py`, `config.py`):
- RFMoE self-gating FFN, off by default behind `use_rfmoe`. θ inference knob.
- Adaptive-λ density control → global density target. `--rfmoe-density-target/-eta`.
- Staircase locality loss KL(π‖sorted p), EMA-ranked stop-grad. `--rfmoe-locality-coef/-zipf-s/-uniform-alpha`.
- Flat→skew curriculum (anneal s:0→s, α:1→α). `--rfmoe-curriculum-ratio`.
- Functional-diversity loss (decorrelate per-token firing). `--rfmoe-diversity-coef`.
- MTP (multi-token prediction) for AR data efficiency, k extra heads reuse tied unembedding. `--mtp-depth/-loss-coef`.
## Next actions

1. **PERF (blocker for any real run):** RFMoE forward is per-expert Python loop (O(N) launches).
   Batch into grouped/padded GEMM. Current form validation-only.
2. **Ternary experts:** `RFMoEExpert` uses `nn.Linear` — swap to `HBitLinear` (Hadamard + ternary STE)
   once design frozen, stay 1.58-bit.
3. **Extensible MoE (thread 2):** `add_expert` primitive + freeze-and-train-new-expert loop + niche-finding.
   Not in tree yet (Yagni until this thread is live).
4. **Serving:** tier experts by usage (hot→VRAM, warm→RAM, cold→SSD), offload + prefetch.
   Optional temporal-stickiness loss (penalize active-set change token-to-token → less page thrash).
5. **Diffusion (thread 3):** large new direction; locality reg (built) is prerequisite (see below).

---

## RFMoE design reference

Ref: RFMoE (arXiv 2604.00801), on AoE (2501.13074) + ReMoE.

Per expert body (3 mats, depth matches dense hybrid FFN):  
`E_i(x) = W_down(silu(W_mid([σ(x A_gate B_gate) ⊙ (x W_up)])))`.  
Expand gate is sigmoid-GLU (paper), not SwiGLU; mid/down are shared depth with dense up→mid→down.  
`A_gate` D×r dual-use (score + gate).
- Score `s_i = ‖x A_gate‖₂`; gate `G_i = ReLU(s_i − b_i)`; fire `1{G_i ≥ θ}`.
- `z = x A_gate` computed ONCE: norm decides, same z feeds B_gate. Skip path = FLOP saving
  (skip W_up/W_mid/W_down + B_gate when not firing).
- Combine `h = x + Σ G_i·E_i` — NO divide-by-count (score = mix weight; RMSNorm renormalizes).
- Sizing `r ≈ D/16`. Decision-dedicated params = scalar `b_i` + global `θ`.
- Expert tensors: `{A_gate, B_gate, W_up, W_mid, W_down, b_i}`.
- Train: pre-threshold `G_i` = differentiable proxy. Bias warmup `b_i≈1e-6` (all fire early),
  λ ramps sparsity. GLOBAL density target, not per-layer.
- Gains (paper, ≤0.8B): PPL −12–19%, θ gives 20× fewer acts / −31% FLOPs. UNPROVEN >0.8B. Research bet.

## Locality reg reference

Load-balance = data-center assumption. Local wants concentration: hot set VRAM-resident, tail offloads.
Target = staircase (uniform within memory tier, step down across): `π = (1−α)·Zipf(s) + α·Uniform(1/N)`.
- Head Zipf → hot-tier ordering; tail uniform → floor π_i ≥ α/N keeps cold experts alive.
- Knobs: `s` head skew (→ hot-set size), `α` tail floor. Set s so top-M (VRAM-fitting) mass ≥ ~0.95.
- Loss `λ_loc·KL(π ‖ sort_desc(p))`. p = usage EMA. KL(π‖p) direction → ∞ as p_i→0 (forbids dead expert).
- On RFMoE: bias `b_i` IS usage control; locality shapes b_i distribution; appended experts → high bias → cold.
- Curriculum: flat early (α≈1) so whole population trains, anneal to skew → cold experts competent but rare.

## Extensible MoE reference (thread 2)

Why self-gating enables it: standard router `G∈D×N` bakes in N — adding expert N+1 renormalizes ALL
routing (softmax over N), breaks load balance, needs router retrain. RFMoE has no central router: append
`{W_up, W_mid, W_down, A_gate, B_gate, b_i}`, existing fire decisions unchanged, residual-add preserves old behavior when frozen.

Procedure: append with HIGH b_i (cold) → freeze everything else → train new expert on new domain → push
into unclaimed niche (diversity term) → re-tune θ so density stays pinned (else cost grows with N).

Hard problems: niche-finding (bias too high=dead, too low=duplicates) is THE problem; density drift with N;
joint-optimality loss vs from-scratch (frozen olds can't co-adapt); new-expert under-training (train flat
first, then cold); needs task boundaries (online append unsolved).

## Diffusion + MoE (thread 3) — locality reg is LOAD-BEARING

Why: local inference (batch=1, idle parallel compute) → diffusion beats AR on latency (T≪N parallel
denoising steps vs N sequential). Mercury 2 ~1000 tok/s; LLaDA 2.0 = MoE+diffusion @100B; DiffusionGemma
26B-A4B. Production block size 128–256 (not ~32 ablation number).

The conflict — per-step expert union: MoE offload needs tiny active set. Diffusion runs whole block
in parallel → union touched ≈ `1−(1−k/E)^N`. AR 1 tok k=8 E=64 → 12.5% (offload works). 256-block →
≈100% (even E=256 → ≈99.97%). ⇒ ~all experts every step → sparse working set GONE → naive load-balanced
MoE+diffusion is WORST local combo (memory-bound on expert weights, offload thrashes).
- Survives: per-token FLOP saving (each token runs k experts). MoE HELPS diffusion (LLaDA 2.0 proof).
- Dies: memory/bandwidth/offload win — whole point of locality section.

The rescue: concentrated routing shrinks union — if ~90% mass on small hot set, parallel tokens
mostly pick SAME experts → union ≈ hot set → offload viable. So locality reg MORE valuable for
diffusion than AR. Bonus: across T denoising steps of ONE block it's same tokens → working set stable
→ load hot set once/block, amortize over T (pairs with temporal-stickiness loss).

Levers/risks: smaller block = smaller union but more sequential steps (measure curve); block-level routing
(all block tokens share expert set) kills per-token specialization; early denoising steps = noisy routing;
concentration NOT optional for local diffusion+MoE.

Full local stack (only composes if routing concentrated): diffusion (cut steps) + locality MoE (working
set resident despite parallelism) + ternary/low-bit (cut bandwidth).

## Prior art

- RFMoE 2604.00801 (github.com/liuyilun2000/RoutingFreeMoE); AoE 2501.13074; ReMoE.
- Aux-loss-free balancing: DeepSeek 2408.15664 (bias-nudge, retarget uniform→staircase).
- Offload: Mixtral-offloading, MoE-Infinity, Fiddler, Pre-gated MoE, EdgeMoE/AdapMoE. Objective: 2512.09277.
- Continual/extensible: Lifelong-MoE 2305.12281, MoE-Adapters 2403.11549 (CVPR24), R²MoE 2507.13107
  (redundancy/niche), CP-MoE 2605.20247, LLaVA-CMoE, TRGE 2508.07738, MoTE. Mostly router-patch, not
  pure self-gating — our angle under-explored.
- Diffusion: SEDD, MDLM/MD4, D3PM (absorbing >> uniform); BD3-LM (block: AR-over-blocks + diffusion-within,
  recovers KV-cache); LLaDA/LLaDA 2.0, DiffuLLaMA/DiffuGPT, Mercury 2; Fast-dLLM, dKV-Cache, d²Cache;
  distillation Di[M]O/T3D/CDLM. Masking schedule = inference algorithm (sets T = latency).
- Data efficiency (why diffusion/any-order learn more per token): Super Data Learners 2511.03276,
  What Makes DLMs SDL 2510.04071; any-order prior XLNet 1906.08237, u-PMLM 2004.11579; data-constrained
  scaling 2305.16264 / 2606.06888. MTP: 2404.19737.