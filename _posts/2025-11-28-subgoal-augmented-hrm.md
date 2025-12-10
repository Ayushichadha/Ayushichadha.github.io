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
- [Architecture and Algorithm](#architecture-and-algorithm)
  - [Theoretical Foundation](#1-theoretical-foundation)
  - [Mathematical Formulation](#2-mathematical-formulation)
  - [Why Directional Subgoals Help HRM](#3-why-directional-subgoals-help-hrm)
  - [Architectural Benefits to HRM](#4-architectural-benefits-to-hrm)
  - [Real-World Use Cases and Applications](#5-real-world-use-cases-and-applications)
  - [Algorithm Summary](#6-algorithm-summary)
- [Does the math of HRM and Feudal Networks clash?](#does-the-math-of-hrm-and-feudal-networks-clash)
- [How feudal subgoals improve HRM (intuitively)](#how-feudal-subgoals-improve-hrm-intuitively)
- [Results from experiments](#results-from-experiments)

---

## Cognitive Core Reasoning

Modern large-scale language models conflate two logically distinct components: (1) a vast statistical memory of the world, and (2) a computational substrate that carries out inference, abstraction, and control.
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

## Architecture and Algorithm

### 1. Theoretical Foundation

#### 1.1 Hierarchical Reasoning Model (HRM) Overview

The Hierarchical Reasoning Model (HRM) is a recurrent architecture that performs sequential reasoning through two interdependent modules operating at different timescales:

- **High-level module (Manager, H)**: Maintains abstract state \( z_H \in \mathbb{R}^d \) and performs strategic planning at a slower timescale
- **Low-level module (Worker, L)**: Maintains detailed state \( z_L \in \mathbb{R}^d \) and executes rapid computations at a faster timescale

The modules interact through alternating updates, where each level conditions on the other's output, creating a hierarchical information flow that enables both abstract planning and detailed execution.

#### 1.2 The Coordination Problem

While HRM's hierarchical structure enables multi-scale reasoning, the coordination between H and L levels is implicit. The worker must infer the manager's intent from shared representations, which can lead to:

- **Credit assignment ambiguity**: Difficulty determining which worker actions contribute to high-level objectives
- **Suboptimal exploration**: Worker may explore directions not aligned with manager's goals
- **Slow convergence**: Lack of explicit guidance increases the search space

#### 1.3 Feudal Subgoal Mechanism: A Solution

We address these limitations by integrating a feudal subgoal mechanism inspired by Feudal Networks (FuN). The key insight is to provide **explicit directional guidance** from manager to worker through latent subgoals, combined with an **intrinsic reward signal** that encourages alignment.

### 2. Mathematical Formulation

#### 2.1 HRM Base Architecture

At each time step \( t \), HRM processes input \( x_t \) through alternating H and L updates:

**Input Encoding:**
\[
\mathbf{e}_t = \text{Embed}(x_t, \text{puzzle\_id})
\]

**Hierarchical Updates:**
\[
\begin{aligned}
\mathbf{z}_L^{(t)} &= \text{L\_level}\left(\mathbf{z}_L^{(t-1)}, \mathbf{z}_H^{(t-1)} + \mathbf{e}_t\right) \\
\mathbf{z}_H^{(t)} &= \text{H\_level}\left(\mathbf{z}_H^{(t-1)}, \mathbf{z}_L^{(t)}\right)
\end{aligned}
\]

where L_level and H_level are transformer-based reasoning modules with \( L\_cycles \) and \( H\_cycles \) internal iterations respectively.

#### 2.2 Feudal Subgoal Head

The subgoal head transforms the manager's hidden state into a directional goal vector and optional gating signal:

**Goal Generation:**
\[
\begin{aligned}
\mathbf{g}_t &= \text{normalize}\left(\mathbf{W}_g \cdot \mathbf{z}_H^{(t)}\right) \\
\sigma_t &= \text{sigmoid}\left(\frac{\mathbf{W}_\sigma \cdot \mathbf{z}_H^{(t)}}{\tau}\right)
\end{aligned}
\]

where:

- \( \mathbf{W}_g \in \mathbb{R}^{d \times d} \): Goal projection matrix
- \( \mathbf{W}_\sigma \in \mathbb{R}^{d \times 1} \): Gating projection matrix  
- \( \tau > 0 \): Temperature parameter (typically \( \tau = 1.0 \))
- \( \text{normalize}(\mathbf{v}) = \frac{\mathbf{v}}{\|\mathbf{v}\|_2 + \epsilon} \): L2 normalization

**Periodic Update Schedule:**
\[
\mathbf{g}_t = \begin{cases}
\text{normalize}\left(\mathbf{W}_g \cdot \mathbf{z}_H^{(t)}\right) & \text{if } t \bmod P = 0 \\
\mathbf{g}_{t-1} & \text{otherwise}
\end{cases}
\]

where \( P \) is the manager period (typically \( P = 3 \) or \( 4 \)), providing temporal stability while allowing adaptation.

#### 2.3 Goal Injection into Hierarchical Updates

The goal vector is injected as an additive bias into both H and L computations:

**Modified Hierarchical Updates:**
\[
\begin{aligned}
\mathbf{g}_t' &= \sigma_t \cdot \mathbf{g}_t \quad \text{(gated goal)} \\
\mathbf{z}_L^{(t)} &= \text{L\_level}\left(\mathbf{z}_L^{(t-1)}, \mathbf{z}_H^{(t-1)} + \mathbf{e}_t + \mathbf{g}_t'\right) \\
\mathbf{z}_H^{(t)} &= \text{H\_level}\left(\mathbf{z}_H^{(t-1)}, \mathbf{z}_L^{(t)} + \mathbf{g}_t'\right)
\end{aligned}
\]

The gating signal \( \sigma_t \) modulates goal strength, allowing the manager to express confidence in the proposed direction.

#### 2.4 Feudal Loss: Intrinsic Reward Signal

The feudal loss provides an intrinsic reward that encourages the worker to align with the manager's goal:

**Feudal Loss Definition:**
\[
\mathcal{L}_{\text{feudal}} = \sigma_t \cdot \left(1 - \cos\left(\mathbf{z}_L^{(t)}, \mathbf{g}_t\right)\right)
\]

where \( \cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2} \) is the cosine similarity.

**Properties:**

- **Range**: \( \mathcal{L}_{\text{feudal}} \in [0, 2\sigma_t] \)
  - Minimum (0): Perfect alignment (\( \cos = 1 \))
  - Maximum (\( 2\sigma_t \)): Opposite directions (\( \cos = -1 \))
- **Scale-invariant**: Normalization ensures directional semantics, independent of hidden state magnitude
- **Differentiable**: Enables gradient-based optimization

**Batch Aggregation:**
\[
\mathcal{L}_{\text{feudal}} = \sum_{i=1}^{B} \sigma_t^{(i)} \cdot \left(1 - \cos\left(\mathbf{z}_L^{(t,i)}, \mathbf{g}_t^{(i)}\right)\right)
\]

#### 2.5 Combined Training Objective

The total loss combines task-specific loss with feudal loss:
\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{feudal}}
\]

where:

- \( \mathcal{L}_{\text{task}} \): Task loss (e.g., cross-entropy for sequence prediction)
- \( \lambda \): Feudal loss weight (typically \( \lambda = 0.05 \))
- \( \mathcal{L}_{\text{feudal}} \): Intrinsic reward loss

**Gradient Flow:**
\[
\frac{\partial \mathcal{L}_{\text{total}}}{\partial \theta} = \frac{\partial \mathcal{L}_{\text{task}}}{\partial \theta} + \lambda \cdot \frac{\partial \mathcal{L}_{\text{feudal}}}{\partial \theta}
\]

The feudal loss provides additional gradient signals that:

- Guide worker parameters \( \theta_L \) toward goal-aligned behaviors
- Train manager parameters \( \theta_H \) to produce useful subgoals
- Update subgoal head parameters \( \theta_g, \theta_\sigma \) to generate effective goals

#### 2.6 Deep Supervision Compatibility

To maintain HRM's deep supervision training stability, goals stored in state are detached from the computation graph:
\[
\mathbf{g}_t^{\text{stored}} = \text{detach}(\mathbf{g}_t)
\]

However, for feudal loss computation, we use non-detached goals to allow gradients to flow back to the subgoal head:
\[
\mathcal{L}_{\text{feudal}} = \sigma_t \cdot \left(1 - \cos\left(\mathbf{z}_L^{(t)}, \mathbf{g}_t\right)\right) \quad \text{(non-detached)}
\]

This design preserves HRM's training stability while enabling end-to-end learning of the subgoal mechanism.

### 3. Why Directional Subgoals Help HRM

#### 3.1 Explicit Hierarchical Communication

**Problem**: In baseline HRM, coordination between H and L is implicit—the worker must infer manager intent from shared hidden states.

**Solution**: Directional subgoals provide explicit communication:

- **Normalized vectors encode direction**: \( \mathbf{g}_t = \text{normalize}(\mathbf{W}_g \mathbf{z}_H) \) extracts directional intent
- **Scale-invariant semantics**: Magnitude doesn't matter, only direction—robust to hidden state scaling
- **Interpretable guidance**: The goal vector directly tells the worker "move in this direction"

**Theoretical Justification**: This transforms an implicit coordination game into an explicit signaling problem, reducing the information-theoretic complexity of hierarchical learning.

#### 3.2 Improved Credit Assignment

**Problem**: When the worker performs actions, it's unclear which actions contribute to high-level objectives.

**Solution**: Feudal loss provides immediate feedback:
\[
\mathcal{L}_{\text{feudal}} = \sigma_t \cdot (1 - \cos(\mathbf{z}_L, \mathbf{g}_t))
\]

- **Direct alignment signal**: Worker receives gradient signal proportional to alignment with goal
- **Temporal credit assignment**: Each worker step is evaluated against current goal
- **Gated importance**: Gate \( \sigma_t \) modulates how much credit matters

**Theoretical Justification**: This implements a form of **intrinsic motivation** where the worker is rewarded for making progress toward manager-specified directions, similar to reward shaping in hierarchical RL.

#### 3.3 Temporal Abstraction and Commitment

**Problem**: Without temporal structure, goals may change too frequently, preventing worker from making progress.

**Solution**: Periodic goal updates with persistence:
\[
\mathbf{g}_t = \begin{cases}
\text{new goal} & \text{if } t \bmod P = 0 \\
\mathbf{g}_{t-1} & \text{otherwise}
\end{cases}
\]

- **Commitment period**: Worker has \( P \) steps to pursue a goal before it changes
- **Stability**: Prevents goal thrashing that would confuse the worker
- **Adaptability**: Regular updates allow refinement as manager learns

**Theoretical Justification**: This implements a **temporal abstraction** where manager operates at a slower timescale (every \( P \) steps) than worker (every step), matching the hierarchical structure.

#### 3.4 Exploration Guidance

**Problem**: Worker may explore directions not aligned with manager's objectives.

**Solution**: Directional goals guide exploration:

- **Constrained exploration**: Worker explores in goal-aligned directions
- **Intrinsic reward**: Feudal loss encourages discovery of goal-relevant patterns
- **Reduced search space**: Focuses worker attention on promising regions

**Theoretical Justification**: This implements **goal-conditioned exploration** where the worker's exploration is biased toward manager-specified directions, improving sample efficiency.

### 4. Architectural Benefits to HRM

#### 4.1 Minimal Overhead, Maximum Impact

The feudal mechanism adds minimal computational cost:

- **Small subgoal head**: Two linear projections (\( \mathbf{W}_g, \mathbf{W}_\sigma \)) with \( O(d^2) \) parameters
- **Efficient loss**: Cosine similarity computation is \( O(d) \) per sample
- **No architectural changes**: Integrates seamlessly with existing HRM structure

**Result**: ~2.7% improvement in LM loss with <1% parameter overhead.

#### 4.2 Compatibility with ACT Mechanism

HRM uses Adaptive Computation Time (ACT) to dynamically determine computation steps. The feudal mechanism is fully compatible:

- **Goal persistence**: Goals persist across ACT steps, providing consistent guidance
- **Q-learning integration**: Feudal loss doesn't interfere with halt/continue Q-learning
- **Deep supervision**: Detached goals maintain HRM's training stability

**Result**: Feudal mechanism enhances ACT by providing directional guidance for each computation step.

#### 4.3 Preserves Deep Supervision

HRM's deep supervision training (detaching states between steps) is critical for stability. The feudal mechanism respects this:

- **Stored goals detached**: Goals in carry state are detached to prevent gradient explosion
- **Loss goals non-detached**: Feudal loss uses non-detached goals to enable learning
- **Selective gradient flow**: Only feudal loss path has gradients, preserving stability

**Result**: Maintains HRM's training stability while enabling subgoal learning.

#### 4.4 Two-Way Learning

The feudal mechanism enables bidirectional learning:

- **Worker learns to follow**: Worker parameters learn to align with manager goals
- **Manager learns to guide**: Manager parameters learn to produce useful goals
- **Joint optimization**: Both levels improve through the shared feudal loss signal

**Result**: Emergent specialization where manager becomes better at goal-setting and worker becomes better at goal-following.

### 5. Real-World Use Cases and Applications

#### 5.1 Symbolic Reasoning Tasks

**ARC (Abstraction and Reasoning Corpus)**: Requires discovering patterns and applying them to new instances.

- **Manager role**: Identifies abstract patterns (e.g., "find symmetry", "extract object")
- **Worker role**: Executes pattern application (e.g., "rotate grid", "color matching cells")
- **Feudal benefit**: Manager guides worker toward pattern-relevant operations, improving generalization

**Sudoku Solving**: Requires constraint satisfaction and logical deduction.

- **Manager role**: Plans high-level strategies (e.g., "focus on row 3", "eliminate candidates")
- **Worker role**: Performs cell-level operations (e.g., "place digit", "update constraints")
- **Feudal benefit**: Manager directs worker attention to promising regions, reducing search

#### 5.2 Multi-Step Planning Tasks

**Maze Navigation**: Requires path planning and execution.

- **Manager role**: Plans high-level path segments (e.g., "move toward northeast corner")
- **Worker role**: Executes step-by-step movements
- **Feudal benefit**: Manager provides waypoint directions, worker follows efficiently

**Tool Use and API Sequencing**: Requires orchestrating multiple tools to achieve goals.

- **Manager role**: Plans tool sequences (e.g., "gather information → process → output")
- **Worker role**: Executes individual tool calls
- **Feudal benefit**: Manager guides worker through tool sequences, reducing errors

#### 5.3 Long-Context Reasoning

**Multi-Turn Conversations**: Requires maintaining context and planning responses.

- **Manager role**: Maintains conversation goals and high-level intent
- **Worker role**: Generates individual tokens and phrases
- **Feudal benefit**: Manager keeps worker aligned with conversation objectives across long contexts

**Document Analysis**: Requires understanding structure and extracting information.

- **Manager role**: Identifies document sections and information types
- **Worker role**: Performs token-level reading and extraction
- **Feudal benefit**: Manager guides worker attention to relevant sections

#### 5.4 Embodied AI and Robotics

**Task Planning**: Requires breaking down tasks into executable actions.

- **Manager role**: Plans task decomposition (e.g., "approach object → grasp → move")
- **Worker role**: Executes low-level motor commands
- **Feudal benefit**: Manager provides directional waypoints, worker executes smoothly

**Navigation**: Requires path planning and obstacle avoidance.

- **Manager role**: Plans high-level routes
- **Worker role**: Executes local movements and obstacle avoidance
- **Feudal benefit**: Manager provides directional guidance, worker adapts locally

### 6. Algorithm Summary

**Training Procedure:**

```
1. Initialize: z_H, z_L, subgoal_state = initial_state()
2. For each batch:
   a. Encode inputs: e = Embed(x, puzzle_id)
   b. Update subgoal (if period reached):
      g = normalize(W_g · z_H)
      σ = sigmoid(W_σ · z_H / τ)
   c. Hierarchical reasoning:
      z_L = L_level(z_L, z_H + e + σ·g)
      z_H = H_level(z_H, z_L + σ·g)
   d. Compute losses:
      L_task = CrossEntropy(lm_head(z_H), labels)
      L_feudal = σ · (1 - cos(z_L, g))
      L_total = L_task + λ · L_feudal
   e. Backpropagate and update parameters
   f. Detach states: z_H, z_L, g → stored for next step
```

**Key Hyperparameters:**

- \( \lambda = 0.05 \): Feudal loss weight (balances task vs. intrinsic reward)
- \( P = 3 \): Manager period (optimal balance of stability and adaptability)
- \( \tau = 1.0 \): Gating temperature (controls gate sharpness)
- Goal dimension: \( d \) (matches hidden size)

---

This section provides the theoretical foundation, mathematical formulation, and practical insights needed for your paper. Should I adjust any part or add more detail to specific subsections?

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

