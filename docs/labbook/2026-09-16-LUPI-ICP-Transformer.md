Given the set of ideas in 2026-09-16-lupi-registration-mechanism-and-architecture-survey, one draft

# Reframing the problem first

Strip away the ML framing for a second: A and B are both samples of the same underlying curve/manifold f, each seen
through its own reparametrization (Φ_0 vs Φ_ρ for position, identity vs h for value), with no correspondence given.
That's not just "vaguely operator-learning-shaped" — it's manifold alignment without correspondence, a problem with real
classical solutions worth stealing from directly.

One fact sharpens this a lot: at d=1, T is necessarily monotone. A diffeomorphism of [0,1] onto itself has to be
monotone — that's topology, not a modeling choice. So at d=1, both T and h are monotone maps, and the whole registration
problem reduces to recovering a monotone correspondence between two point sets sampling the same function — which is
exactly what 1D optimal transport solves in closed form (the monotone rearrangement / quantile-quantile map), when the
marginals match. They don't match here (A is a scarce subregion of B, which is why CLAUDE.md already rules out
balanced/Sinkhorn transport) — but the shape of the right answer, "recover order-preserving correspondence from unpaired
samples of a shared curve," is the real target, and it's the same target functional-data-analysis calls curve
registration and computer-vision-and-robotics calls point-set registration.

That second field has a famous, simple, and exactly on-point classical algorithm: Iterative Closest Point (ICP).
Alternate two steps until convergence — (a) given the current transform estimate, find each point's nearest neighbor in
the other set; (b) given that correspondence, re-estimate the transform. That alternation is the concrete algorithmic
skeleton I think should replace "generic depth, hope it discovers something transport-shaped."

## The concrete model: an Iterative Registration Transformer (IRT)

Maintain explicit state, not just hidden representations. Per B-token j, a position estimate T̂_j ∈ R^d (init: T̂_j^
(0) = x_j^B, i.e. start from the identity — a reasonableprior since the curriculum spends real time near ρ=0). Globally
(not per-token — h is one function per draw, not n_B of them), a handful of learnable h-slots: K_h extra prior since the
curriculum spends real time near ρ=0). Globally (not per-token — h is one function per draw, not n_B of them), a handful
of learnable h-slots: K_h extra tokens, no data attached (this is exactly the "thinking rows" idea from the log, given a
specific job instead of a vague "extra scratch space" one — they carry h's

**One weight-tied iteration block (applied K times, Universal-Transformer style — same weights every time, not K
independently-parametrized layer**s):

1. Value step (uses current T̂^ (k)): soft-match each T̂_j^ (k) against A's context by
   position:                                                                            
   corr_ji = softmax_i (-‖T̂_j^ (k) − x_i^A‖² / τ_pos), giving a locally-expected A-scale value ỹ_j = Σ_i corr_ji ·
   y_i^A. Cross-attend the h-slots into the set of pseudo-pairs { (y_j^B, ỹ_j)} weighted by how peaked corr_j is —
   literally an in-context regression sub-problem, "fit a monotone Cross-attend the h-slots into the set of
   pseudo-pairs { (y_j^B, ỹ_j)} weighted by how peaked corr_j is — literally an in-context regression sub-problem, "fit
   a monotone h-slots.
2. Position step (uses updated ĥ^ (k)): now that ĥ^ (k)(y_j^B) is on A's scale, soft-match by
   value:                                                                        
   corr' _ji = softmax_i (- (ĥ^ (k)(y_j^B) − y_i^A)² / τ_val), and pull T̂_j^ (k+1) = (1−α)·T̂_j^ (k) + α·Σ_i corr'
   _ji · x_i^A.                                                 
   Because A is scarce, a handful of anchors alone will pull noisily — add a second, graph-smoothing term exploiting T's
   smoothness (unused inductive bias flagged in the log's §1): pull T̂_j also toward a kernel-weighted average of nearby
   B points' own current estimates, Σ_{j'} w_{jj'} (T̂^ (k)) · T̂_{j'}^ (k), w a soft-kNN kernel in the current
   estimated frame. This is the manifold-alignment move (same spirit as LLE/Isomap: preserve local geometry, don't just
   anchor to the few labeled points) — it's current estimated frame. This is the manifold-alignment move (same spirit as
   LLE/Isomap: preserve local geometry, don't just anchor to the few labeled points) — it's
3. Residual generic attention pass — ordinary self-attention
   over [A_ctx; B (now carrying T̂^ (k+1), ĥ^ (k)(y_j^B)); h-slots], a normal PFNBlock, so whatever the explicit
   mechanism doesn't capture still has a fallback path. This is the part that makes it a hybrid, not a pure hand-rolled
   algorithm forced through a neural net.

Every correspondence in both steps is softmax-normalized over real candidates — never a raw logit target — so the blowup
failure mode from the log's §3 is structurally   
excluded from the ground up, not patched on.

**Halting**: fixed K to start; ACT-style adaptive halting (stop when ‖T̂^ (k+1) − T̂^ (k)‖ drops below a threshold) is the
natural principled upgrade once the fixed-K version works, since "how many ICP iterations does this draw need" genuinely
varies with how hard the warp is.

**Readout**: query cross-attends into the final iteration's state, exactly like IDTokenPFN's existing query pathway —
nothing changes downstream of this.

**Training: this is where LUPI gets used maximally, not just once**

Supervise the explicit state at every iteration, not just the final output — w_k ∝ k deep supervision, but on
interpretable quantities instead of abstract hidden states:
L = NLL (final prediction, y*) + Σ_k
w_k · [ λ_T · L_T (T̂^ (k), x^{B→A}) + λ_h · L_h (ĥ^ (k)(y^B), y^{B→A}) ]                                                                 
using exactly the distributional heads from the log's §2 (TransportHead for L_T, the Gaussian-around-monotone-mean for
L_h) — and optionally supervise corr'_ji itself    
against the soft kernel-in-z-space target from §3, at every iteration too. This is a much more literal use of privileged
information than "soften the final loss" — you're directly telling the model, at every single refinement step, whether
its current guess of the actual thing you care about is right.

**Why this is the right synthesis of what you pointed me at**  
                                                                                                                                                                         
  - Operator learning (ICON): A_ctx and B are two related "demonstration sets" of the shared operator f, linked by the unknown (T, h) — the iteration is explicitly trying  
le  to identify that link, not just implicitly absorb it into a black-box final loss.                                                                                       
  - Universal Transformer: the weight-tying isn't cosmetic — it's the direct architectural expression of the Neumann-series/unrolled-fixed-point result you raised: this    
    literally is one algorithm (alternating correspondence/transform re-estimation) applied K times with the same weights, not K different layers that merely have enough   
    capacity to approximate one.                                                                                                                                            
  - Both unused structural priors from §1 get used: h's monotonicity (RQS-spline head), T's smoothness (the graph-consistency term).                                        
  - §3's normalization lesson is baked in from step 1, not bolted on after a blowup.                                                                                        
                                                                                                                                                                            
  Honest risks, because ICP is famous for exactly this failure                                                                                                              
                                                                                                                                                                            
  ICP's classical Achilles' heel is local minima from bad initialization — alternating "nearest correspondence" / "re-fit transform" can lock onto a self-consistent but    
  wrong alignment and never escape, especially with few anchors. Mitigations worth having going in, not bolted on after it happens: soft (never hard-argmax) correspondences
  throughout, τ_pos/τ_val annealed coarse-to-fine across iterations (wide early, sharp late — the same "coarse-to-fine over depth" idea as w_ℓ∝ℓ), and — critically — this  
  is trained end-to-end with real gradient supervision at every step, unlike classical ICP which just runs at inference with no learning signal at all. That's a materially 
  different, better-conditioned setting than what makes vanilla ICP get stuck, but it's not a guarantee, and it's the first thing I'd watch for empirically (does T̂ visibly
  converge to the right thing at d=1, where you can plot it, before trusting it at higher d).                                                                               
                                                                                                                                                                            
  This is also a real jump in complexity over IDTokenPFN — more moving parts (τ_pos, τ_val, α, K, K_h), more failure surface. I would not jump straight to the full thing.  
  Concrete incremental path: (1) verify the value-step and position-step each work in isolation with the other one teacher-forced to ground truth (does value-calibration   
  converge given true T? does position-refinement converge given true h?) before ever running them coupled; (2) only then couple them and add the residual attention pass;  
  (3) only then make it iterative/weight-tied. Each of those is a cheap, d=1, visualizable checkpoint before the next.                  