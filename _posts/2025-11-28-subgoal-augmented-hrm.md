---
layout: page
title: "Subgoal-Augmented Hierarchical Reasoning"
permalink: /research/
---

<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)'], ['$', '$']],
    displayMath: [['\\[', '\\]'], ['$$', '$$']],
    processEscapes: true
  },
  options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## Contents

- [Introduction: cognitive core reasoning](#introduction)
- [Subgoal-Augmented HRM: Model Design & Training](#subgoal-augmented-hrm-model-design--training)
  - [Theoretical Foundation](#1-theoretical-foundation)
  - [Mathematical Formulation](#2-mathematical-formulation)
  - [Why Directional Subgoals Help](#3-why-directional-subgoals-help)
  - [Training Algorithm](#4-training-algorithm)
  - [Real-World Intuition and Applications](#5-real-world-intuition-and-applications)
- [Does the math of HRM and Feudal Networks clash?](#does-the-math-of-hrm-and-feudal-networks-clash)
- [How feudal subgoals improve HRM (intuitively)](#how-feudal-subgoals-improve-hrm-intuitively)
- [Results from experiments](#results-from-experiments)

---

## Cognitive Core Reasoning

Modern large-scale language models conflate two logically distinct components:(1) a vast statistical memory of the world, and(2) a computational substrate that carries out inference, abstraction, and control.
The notion of a cognitive core starts from the hypothesis that these can be separated. Instead of a single model that simultaneously memorises most of the internet and performs all computation, we can imagine a compact network whose parameters are devoted almost entirely to procedural competence: algorithms for multi-step inference, search over latent states, hierarchical control, and credit assignment. Factual and episodic knowledge can then be offloaded to external memory, retrieval systems, or larger frozen language models.
In this view, a cognitive core is a relatively small model that is optimised not to encode encyclopaedic knowledge, but to implement and coordinate complex computations. Its role is to:

- maintain and update internal state over extended time horizons,
- decompose goals into subgoals and allocate computation across scales,
- decide what to query from external memory or tools and when, and
- integrate the results of those interactions into a coherent plan.
The emphasis is on orchestration of computation rather than storage of facts: the core should know how to think, search, and plan given access to tools and memory, rather than knowing everything in advance.
This work studies one concrete step in that direction. We consider a Hierarchical Reasoning Model (HRM) augmented with feudal-style subgoal mechanisms as a candidate architecture for a cognitive core. HRMs already instantiate several properties that are desirable for such a kernel:

- they are small (tens of millions of parameters rather than billions),
- they perform deep latent-space reasoning via recurrent dynamics instead of long chain-of-thought traces, and
- their design is explicitly neuroscience-inspired, with separate fast and slow modules that reflect temporal separation and hierarchical processing in cortical circuits.
However, a generic hierarchical recurrent system is not yet enough. To move closer to a cognitive core, we argue that explicit subgoal planning is essential. Human and animal behaviour is structured by intermediate objectives: long-horizon tasks are decomposed into segments, and internal representations evolve under constraints imposed by these subgoals. Analogously, a core reasoning module should not simply run an undifferentiated deep computation; it should be able to propose, revise, and enforce subgoals over its own latent trajectory.
To capture this, we introduce a Subgoal-Augmented Hierarchical Reasoning Model that blends HRM with ideas from Feudal Networks. A slow “manager” pathway periodically emits directional subgoals in the shared latent space; a fast “worker” pathway is trained, via a feudal-style intrinsic objective, to evolve its hidden state in accordance with these directions over short temporal windows. The resulting system retains HRM’s compact, brain-inspired hierarchy, but adds an explicit mechanism for hierarchical subgoal planning in latent space.
Our exploration is therefore not only about HRM in isolation, nor about feudal subgoals in isolation, but about their interaction:

- HRM provides a small, temporally-structured, recurrent backbone that behaves like a minimal reasoning engine.
- Feudal-style subgoals provide the planning structure that carves long internal computations into goal-directed segments.
Together, they shift the design compass toward a cognitive core: a model that devotes its limited parameters to organising computation—via hierarchy, temporal separation, and subgoal planning—rather than to storing large volumes of static knowledge.

---

## Subgoal-Augmented HRM: Model Design & Training

### 1. Theoretical Foundation

The Hierarchical Reasoning Model (HRM) uses two coupled recurrent modules at different time scales. A high-level manager with state \( \mathbf{z}_H^{(t)} \) plans more slowly, while a low-level worker with state \( \mathbf{z}_L^{(t)} \) updates every step. Alternating updates let the worker refine details conditioned on the manager, and the manager adjust strategy based on the worker. Coordination is the core challenge: the worker must infer the manager’s intent from shared representations, which weakens credit assignment, slows convergence, and encourages unfocused exploration. A feudal subgoal mechanism fixes this by having the manager emit explicit directional goals plus a gate that expresses confidence; the worker is trained to align its state evolution with those directions through an intrinsic feudal loss. Goals are detached in the carry state (to preserve HRM’s deep supervision stability) but left attached in the loss path so gradients still teach the subgoal head. The mechanism is lightweight (two projections), compatible with ACT, and leaves the base HRM recurrence intact.

### 2. Mathematical Formulation

At time \( t \), with input \( x_t \):

**Input encoding**
$$
\mathbf{e}_t = \text{Embed}(x_t, \text{puzzle\_id})
$$

**Base recurrence**
$$
\begin{aligned}
\mathbf{z}_L^{(t)} &= \text{L\_level}\!\left(\mathbf{z}_L^{(t-1)}, \mathbf{z}_H^{(t-1)} + \mathbf{e}_t\right) \\\\
\mathbf{z}_H^{(t)} &= \text{H\_level}\!\left(\mathbf{z}_H^{(t-1)}, \mathbf{z}_L^{(t)}\right)
\end{aligned}
$$

**Subgoal head (manager)**
$$
\begin{aligned}
\mathbf{g}_t &= \text{normalize}\!\left(\mathbf{W}_g \,\mathbf{z}_H^{(t)}\right) \\\\
\sigma_t &= \text{sigmoid}\!\left(\frac{\mathbf{W}_\sigma \,\mathbf{z}_H^{(t)}}{\tau}\right)
\end{aligned}
$$
where \( \mathbf{W}_g \in \mathbb{R}^{d \times d} \), \( \mathbf{W}_\sigma \in \mathbb{R}^{d \times 1} \), and \( \text{normalize}(\mathbf{v}) = \frac{\mathbf{v}}{\|\mathbf{v}\|_2 + \epsilon} \).

**Periodic goal update**
$$
\mathbf{g}_t =
\begin{cases}
\text{normalize}\!\left(\mathbf{W}_g \,\mathbf{z}_H^{(t)}\right), & t \bmod P = 0 \\\\
\mathbf{g}_{t-1}, & \text{otherwise}
\end{cases}
$$

**Goal injection**
$$
\begin{aligned}
\mathbf{g}_t' &= \sigma_t \, \mathbf{g}_t \\\\
\mathbf{z}_L^{(t)} &= \text{L\_level}\!\left(\mathbf{z}_L^{(t-1)}, \mathbf{z}_H^{(t-1)} + \mathbf{e}_t + \mathbf{g}_t'\right) \\\\
\mathbf{z}_H^{(t)} &= \text{H\_level}\!\left(\mathbf{z}_H^{(t-1)}, \mathbf{z}_L^{(t)} + \mathbf{g}_t'\right)
\end{aligned}
$$

**Feudal loss (intrinsic alignment)**
$$
\mathcal{L}_{\text{feudal}} = \sigma_t \,\bigl(1 - \cos(\mathbf{z}_L^{(t)}, \mathbf{g}_t)\bigr), \quad
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2\,\|\mathbf{b}\|_2 + \epsilon}
$$

**Total objective**
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \,\mathcal{L}_{\text{feudal}}
$$
with gradients flowing through manager, worker, and subgoal head; stored goals are detached between steps to retain HRM’s stability.

### 3. Why Directional Subgoals Help

Explicit goal vectors turn implicit coordination into a direct signal: the manager names a direction in latent space, the worker follows it, and the feudal loss provides immediate feedback. This improves credit assignment (each worker step is judged against the current goal), imposes temporal abstraction through the period \( P \) (goals persist for several steps before refreshing), and guides exploration toward manager-relevant regions rather than diffuse search. Gating \( \sigma_t \) lets the manager modulate confidence, strengthening or weakening the pull of a goal as needed.

### 4. Training Algorithm

```python
# Pseudocode for one training epoch
z_H, z_L, g = init_state()
for batch in dataloader:
    for t, x_t in enumerate(batch.sequence):
        e_t = Embed(x_t, puzzle_id)
        if t % P == 0:
            g = normalize(W_g @ z_H)
            sigma = sigmoid((W_sigma @ z_H) / tau)
        g_gate = sigma * g
        z_L = L_level(z_L, z_H + e_t + g_gate)
        z_H = H_level(z_H, z_L + g_gate)
    logits = lm_head(z_H)
    L_task = CrossEntropy(logits, labels)
    L_feudal = sigma * (1 - cos(z_L, g))
    L_total = L_task + lambda_feudal * L_feudal
    L_total.backward()
    optimizer.step(); optimizer.zero_grad()
    z_H, z_L, g = detach(z_H), detach(z_L), detach(g)
```

Key hyperparameters: \( \lambda = 0.05 \) (feudal weight), \( P \in \{3,4\} \) (manager period), \( \tau = 1.0 \) (gate temperature), and hidden size \( d \) for goals.

### 5. Real-World Intuition and Applications

Directional subgoals are most useful when fast, local moves must stay aligned with slower, strategic intent. In symbolic reasoning (ARC-style grids, Sudoku), the manager proposes abstract moves (“grow this pattern,” “focus on row 3”) while the worker executes cell-level updates. In planning tasks (maze navigation, tool/API sequencing), manager goals act as waypoints so the worker follows coherent paths. For long-context language and document reasoning, manager goals maintain topic and intent while the worker handles token-level generation. In embodied settings (navigation, manipulation), goals provide directional guidance while the worker handles low-level control. Across these cases, subgoals give a compact, explicit channel for guidance without altering the base HRM backbone.

---

## Does the math of HRM and Feudal Networks clash?

Mathematically, the two ideas **fit together cleanly**:

- HRM already defines a two-time-scale recurrent computation with a manager and a worker.
- Feudal Networks contribute a **particular way to parameterize and train** the interaction: a subgoal vector in latent space, plus an alignment loss on the resulting state change.

In my setup, the manager’s forward pass is unchanged except for an extra head that emits $g_\tau$. The worker’s forward pass is unchanged except for being conditioned on this extra vector. The gradient flows from:

1. the main task loss into both manager and worker (as in vanilla HRM), and  
2. the feudal loss into both the subgoal head and the worker states.

There is no hard constraint that forces the model into an inconsistent geometry; we are simply **adding an extra, differentiable regularizer** that nudges the system toward a decomposition where:

- the manager chooses directions $g_\tau$, and  
- the worker is encouraged to realize those directions in its dynamics.

The only “tension” is practical, not mathematical: if $\lambda_{\text{feudal}}$ is too large, the model may over-optimize for making its state changes align with $g_\tau$ even when that’s not optimal for the task. In practice I keep this weight small and treat feudal supervision as a **soft prior on coordination**, not as a hard constraint.

---

## How feudal subgoals improve HRM (intuitively)

The motivation for adding feudal subgoals is to give the manager a **more structured role** than just "being another context vector":

1. **Better credit assignment for the manager**  
   In vanilla HRM, the manager’s influence on performance is often several steps away: it sets a state, the worker runs for a while, and only then do we see whether the puzzle was solved. With feudal subgoals, each manager update gets an immediate, dense signal: *did the worker’s latent trajectory move roughly in the direction I asked for?* This gives the manager a local objective that aligns with its global role.

2. **Commitment windows for the worker**  
   Over each $K$-step window, the worker is asked to “move in direction $g_\tau$”. This acts like a **soft plan**: instead of re-deciding from scratch at every step, the worker is biased toward a consistent direction of computation for a short horizon. That helps avoid oscillatory or myopic behavior, especially on puzzles requiring multi-step transformations.

3. **Implicit decomposition of the space**  
   The alignment loss encourages the model to organize its latent space so that meaningful “moves” in reasoning correspond to reasonably well-behaved directions. Over time, the manager can learn to reuse similar subgoals across different instances (“grow this pattern,” “swap these two regions,” “focus on the top-left sub-grid”), which is closer to how we think about reusable mental operations.

4. **Bridging RL-style planning and supervised reasoning**  
   Feudal Nets were originally defined in RL, where subgoals are about moving an agent in physical or abstract state space. Here, the “state” is a high-dimensional representation of a puzzle and partial computation. By importing the same idea into a supervised HRM, we get a small, principled step toward **latent planning** inside a reasoning model.

---

## Results from experiments

I've been testing this Subgoal-Augmented HRM on small, ARC-mini–style grid puzzles with a supervised learning setup (predicting target grids from input grids). Some early, but consistent, observations:

- **Setup**
  - **Baseline:** vanilla HRM with a two-level recurrent architecture (manager + worker), no feudal loss.
  - **Subgoal model:** same architecture, plus subgoal head and feudal loss as described above.
  - **Manager period $K$:** swept over small values (e.g., 3–6).
  - **Feudal loss weight $\lambda_{\text{feudal}}$:** swept over small values (e.g., around 0.05).

- **Training dynamics**
  - For reasonable hyperparameters (e.g., $K \in \{3,4\}$, $\lambda_{\text{feudal}} \approx 0.05$), the subgoal model:
    - converges more smoothly (less noisy validation loss),
    - avoids some degenerate modes where the manager’s state stops changing meaningfully.

- **Quantitative improvements**
  - At comparable training steps, the subgoal-augmented HRM shows roughly:
    - **5–6% lower task / language-model loss** than the baseline HRM, and  
    - **≈8 percentage-point higher accuracy** on held-out ARC-mini puzzles.
  - These numbers vary across seeds and exact configs, but the direction of improvement is stable.

- **Qualitative behavior**
  - Inspecting hidden trajectories suggests that subgoals are not random: similar puzzle types tend to induce similar subgoal directions.
  - Changing the manager period changes the “granularity” of these moves: shorter periods produce finer-grained, more reactive subgoals; longer periods seem to encourage coarser, more global shifts.

Overall, the experiments support the initial intuition: **adding feudal-style subgoals on top of HRM does not break the math; instead, it gives the model a more structured way to coordinate its two levels.** The gains so far are modest but consistent, and the more interesting part is conceptual—this looks like a viable path toward small, tool-ready “cognitive cores” that can be slotted into larger systems and asked to handle the hard parts of multi-step reasoning.

