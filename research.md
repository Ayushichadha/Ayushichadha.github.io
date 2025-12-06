---
layout: page
title: "Research"
permalink: /research/
---

## Subgoal-Augmented Hierarchical Reasoning Model (HRM)

My main independent project builds on the **Hierarchical Reasoning Model (HRM)** — a 27M-parameter, brain-inspired recurrent model with two coupled modules:

- a **slow high-level module** that updates infrequently and sets the direction of reasoning, and  
- a **fast low-level module** that performs detailed step-by-step computation.

HRM was originally proposed as an alternative to chain-of-thought prompting for hard reasoning tasks like ARC-AGI puzzles, extreme Sudoku, and maze navigation. Instead of generating long text explanations, it performs deep **latent** computation with adaptive computation time.

In my work, I extend HRM with a **feudal-style subgoal head**:

- The high-level module periodically outputs a **latent subgoal vector**.
- The low-level module is conditioned on this subgoal while it runs for several steps.
- An auxiliary **feudal loss** encourages the change in the low-level state to align with the subgoal direction, similar in spirit to Feudal Networks in hierarchical reinforcement learning.

Early experiments on ARC-like grid puzzles show that a **small amount of subgoal supervision** improves both loss and accuracy compared to the vanilla HRM baseline, suggesting better coordination between high- and low-level reasoning.

**Code & Resources:**
- Repository: https://github.com/Ayushichadha/scout

I see this as a step toward compact “cognitive cores” — small models focused on the **algorithms of reasoning**, which can later be plugged into larger systems or tools.
