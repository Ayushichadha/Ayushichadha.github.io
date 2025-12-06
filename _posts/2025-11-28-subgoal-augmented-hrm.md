---
layout: page
title: "Subgoal-Augmented Hierarchical Reasoning"
permalink: /research/
---

## Contents

- [Introduction: cognitive core reasoning](#introduction)
- [Algorithm and Architecture](#algorithm-and-architecture)
  - [Feudal subgoal head](#feudal-subgoal-head)
  - [Feudal loss](#feudal-loss)
- [Does the math of HRM and Feudal Networks clash?](#does-the-math-of-hrm-and-feudal-networks-clash)
- [How feudal subgoals improve HRM (intuitively)](#how-feudal-subgoals-improve-hrm-intuitively)
- [Results from experiments](#results-from-experiments)

---

## Cognitive Core Reasoning

Modern large-scale language models conflate two logically distinct components: (1) a vast statistical memory of the world, and (2) a computational substrate that carries out inference, abstraction, and control.
The notion of a cognitive core starts from the hypothesis that these can be separated. Instead of a single model that simultaneously memorises most of the internet and performs all computation, we can imagine a compact network whose parameters are devoted almost entirely to procedural competence: algorithms for multi-step inference, search over latent states, hierarchical control, and credit assignment. Factual and episodic knowledge can then be offloaded to external memory, retrieval systems, or larger frozen language models.
In this view, a cognitive core is a relatively small model that is optimised not to encode encyclopaedic knowledge, but to implement and coordinate complex computations. Its role is to:
* maintain and update internal state over extended time horizons,
* decompose goals into subgoals and allocate computation across scales,
* decide what to query from external memory or tools and when, and
* integrate the results of those interactions into a coherent plan.
The emphasis is on orchestration of computation rather than storage of facts: the core should know how to think, search, and plan given access to tools and memory, rather than knowing everything in advance.
This work studies one concrete step in that direction. We consider a Hierarchical Reasoning Model (HRM) augmented with feudal-style subgoal mechanisms as a candidate architecture for a cognitive core. HRMs already instantiate several properties that are desirable for such a kernel:
* they are small (tens of millions of parameters rather than billions),
* they perform deep latent-space reasoning via recurrent dynamics instead of long chain-of-thought traces, and
* their design is explicitly neuroscience-inspired, with separate fast and slow modules that reflect temporal separation and hierarchical processing in cortical circuits.
However, a generic hierarchical recurrent system is not yet enough. To move closer to a cognitive core, we argue that explicit subgoal planning is essential. Human and animal behaviour is structured by intermediate objectives: long-horizon tasks are decomposed into segments, and internal representations evolve under constraints imposed by these subgoals. Analogously, a core reasoning module should not simply run an undifferentiated deep computation; it should be able to propose, revise, and enforce subgoals over its own latent trajectory.
To capture this, we introduce a Subgoal-Augmented Hierarchical Reasoning Model that blends HRM with ideas from Feudal Networks. A slow “manager” pathway periodically emits directional subgoals in the shared latent space; a fast “worker” pathway is trained, via a feudal-style intrinsic objective, to evolve its hidden state in accordance with these directions over short temporal windows. The resulting system retains HRM’s compact, brain-inspired hierarchy, but adds an explicit mechanism for hierarchical subgoal planning in latent space.
Our exploration is therefore not only about HRM in isolation, nor about feudal subgoals in isolation, but about their interaction:
* HRM provides a small, temporally-structured, recurrent backbone that behaves like a minimal reasoning engine.
* Feudal-style subgoals provide the planning structure that carves long internal computations into goal-directed segments.
Together, they shift the design compass toward a cognitive core: a model that devotes its limited parameters to organising computation—via hierarchy, temporal separation, and subgoal planning—rather than to storing large volumes of static knowledge.

---

## Algorithm and Architecture

At a high level, the baseline HRM can be described in two interacting pieces:

- A **manager (slow module)** with hidden state $m_t$ that updates every $K$ steps.
- A **worker (fast module)** with hidden state $h_t$ that updates at every token / time-step.

On most steps, only the worker runs:

- The worker takes the current token (or grid observation), the previous worker state $h_t$, and a *cached* manager state $m_{\tau}$ from the last manager update, and produces:
  - the next worker state $h_{t+1}$
  - the usual language-model or puzzle head outputs (logits, actions, etc.).

Every $K$ steps (i.e., when $t$ hits a manager boundary), we:

1. Aggregate information from the last $K$ worker steps (e.g., via the final worker state or a pooled summary).
2. Update the manager state $m_{\tau+1}$.
3. Broadcast this updated manager context to the worker for the next $K$-step window.

### Feudal subgoal head

My extension adds a **subgoal head** on top of the manager:

- At each manager update index $\tau$, the manager produces a **subgoal vector**
  $$
  g_\tau \in \mathbb{R}^d
  $$
  in the same latent space as the worker’s hidden state, or a projection of it.
- For the next $K$ worker steps $t \in [\tau K, (\tau+1)K)$, the worker is *conditioned* on this subgoal:
  - either by concatenating $g_\tau$ to its input,
  - or by using it as an affine modulation / bias inside the worker’s blocks.

So structurally:

1. **Manager forward (every $K$ steps)**  
   - Input: summary of recent worker states.  
   - Output: new $m_{\tau+1}$, plus subgoal vector $g_{\tau+1}$.

2. **Worker forward (every step)**  
   - Input: token / observation, previous $h_t$, current manager state $m_{\tau}$, and subgoal $g_\tau$.  
   - Output: new $h_{t+1}$, predictions.

### Feudal loss

To actually *train* the subgoals, I add an auxiliary loss that encourages the worker’s change in state over a window to align with the manager’s chosen direction. Concretely, for each window:

- Let $h_{\text{start}}$ and $h_{\text{end}}$ be the worker states at the beginning and end of the window.
- Define a **direction of progress**:
  $$
  \Delta h_\tau = h_{\text{end}} - h_{\text{start}}.
  $$
- Encourage $\Delta h_\tau$ to align with $g_\tau$ using a cosine-similarity based term:
  $$
  \mathcal{L}_{\text{feudal},\tau}
    = - \cos(\Delta h_\tau, g_\tau)
    = - \frac{\Delta h_\tau \cdot g_\tau}{\|\Delta h_\tau\|\;\|g_\tau\| + \epsilon}
  $$

The total loss is then:
$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_{\text{feudal}} \cdot \mathcal{L}_{\text{feudal}},
$$
where $\mathcal{L}_{\text{task}}$ is the usual cross-entropy / puzzle loss, and $\lambda_{\text{feudal}}$ is a small weight that keeps the subgoal supervision gentle rather than dominating training.

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

The motivation for adding feudal subgoals is to give the manager a **more structured role** than just “being another context vector”:

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

I’ve been testing this Subgoal-Augmented HRM on small, ARC-mini–style grid puzzles with a supervised learning setup (predicting target grids from input grids). Some early, but consistent, observations:

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

