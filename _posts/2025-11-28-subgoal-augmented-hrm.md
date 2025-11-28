---
layout: post
title: "Subgoal-Augmented Hierarchical Reasoning (HRM + Feudal Subgoals)"
---

My main independent project builds on the **Hierarchical Reasoning Model (HRM)** — a compact, brain-inspired recurrent architecture that separates reasoning into two coupled modules:

- a **slow high-level module** that updates infrequently and sets the direction of reasoning, and
- a **fast low-level module** that performs detailed step-by-step computation.

HRM offers an alternative to chain-of-thought style approaches for difficult reasoning tasks (ARC-like puzzles, extreme Sudoku, maze navigation) by performing deep latent computation with adaptive computation time rather than generating long textual explanations.

In this post I describe a simple extension: a **feudal-style subgoal head** that helps the high-level controller coordinate the low-level processing.

- The high-level module periodically emits a **latent subgoal vector**.
- The low-level module is conditioned on that subgoal while it runs for several steps, biasing its trajectory through latent state-space.
- An auxiliary **feudal loss** encourages the change in the low-level state to align with the subgoal direction, similar in spirit to Feudal Networks from hierarchical reinforcement learning.

Early experiments on ARC-like grid puzzles show that providing a small amount of subgoal supervision improves both training loss and downstream accuracy compared to a vanilla HRM baseline. Intuitively, the subgoal signal helps the modules specialize: the high-level unit learns to propose useful intermediate objectives while the low-level unit focuses on executing them.

Why I find this promising:

- It enforces an explicit separation between planning and execution inside a compact recurrent core.
- It can be trained with modest amounts of supervision (or auxiliary objectives) to improve coordination.
- It points toward small, reusable “cognitive cores” that implement algorithms of reasoning and can be plugged into larger systems.

Code and detailed experimental notes will appear here when ready. For now, this post records the architecture idea and preliminary results.
