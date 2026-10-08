---
layout: null
title: Ayushi Chadha
description: Researcher and engineer working on reasoning and agentic systems, latent reasoning, hierarchical control, and adaptive computation.
---
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ayushi Chadha</title>
  <meta name="description" content="Researcher and engineer working on reasoning and agentic systems, latent reasoning, hierarchical control, and adaptive computation.">
  <link rel="canonical" href="{{ site.url }}/">
  <meta property="og:type" content="website">
  <meta property="og:title" content="Ayushi Chadha">
  <meta property="og:description" content="Research on latent reasoning, hierarchical control, and adaptive computation.">
  <meta property="og:url" content="{{ site.url }}/">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="Ayushi Chadha">
  <meta name="twitter:description" content="Research on latent reasoning, hierarchical control, and adaptive computation.">
  <link rel="stylesheet" href="{{ '/assets/portfolio.css' | relative_url }}?v=20261008">
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "Person",
    "name": "Ayushi Chadha",
    "url": "{{ site.url }}/",
    "sameAs": [
      "https://github.com/Ayushichadha",
      "https://www.linkedin.com/in/ayushi-chadha-ai",
      "https://x.com/AyushiChadha24",
      "https://substack.com/@ayushi25"
    ],
    "knowsAbout": ["latent reasoning", "recurrent reasoning", "hierarchical control", "adaptive computation", "agentic systems"]
  }
  </script>
  {% include analytics.html %}
</head>
<body>
  <header class="site-header home-header">
    <div class="wrap header-inner">
      <nav aria-label="Primary navigation">
        <a href="#research">Research</a>
        <a href="#experience">Experience</a>
        <a href="#programs">Programs</a>
        <a href="#education">Education</a>
        <a href="#news">News</a>
        <a href="{{ '/reading/' | relative_url }}">Selected reading</a>
        <a href="#writing">Writing</a>
      </nav>
    </div>
  </header>

  <main id="top" class="wrap">
    <section class="intro" aria-labelledby="intro-title">
      <h1 id="intro-title">Ayushi Chadha</h1>
      <p class="intro-lead">I work on <strong>reasoning and agentic systems</strong>, with a particular interest in latent reasoning in hierarchical recurrent models: how internal computation develops across <strong>recurrent depth</strong>, how medium-horizon intent can be represented in latent space, and when that intent should persist or be revised.</p>
      <p>My recent work approaches these questions through hierarchical control and adaptive computation. My first paper studies persistent directional subgoals inside a recurrent latent reasoner. My second studies learned supervisory control and asks whether an apparently adaptive internal signal actually produces useful decisions.</p>
      <p>Before this work, I spent several years in <strong>machine learning and applied AI at Propero</strong>, where I led the development of ShopiBot, a domain-specific AI system, and worked across retrieval, agentic workflows, evaluation, and ML system design.</p>
      <p>I am interested in research and engineering work around <strong>reasoning, adaptive computation, agents, and model-facing systems</strong>.</p>
      <p class="direct-links"><a href="mailto:ayushichadha48@gmail.com">Email</a><a href="{{ '/assets/Ayushi_Chadha_CV.pdf' | relative_url }}">CV</a><a href="https://github.com/Ayushichadha">GitHub</a><a href="https://www.linkedin.com/in/ayushi-chadha-ai">LinkedIn</a><a href="https://x.com/AyushiChadha24">X</a><a href="https://substack.com/@ayushi25">Substack</a><a href="{{ '/reading/' | relative_url }}">Selected reading</a></p>
    </section>

    <section id="research" aria-labelledby="research-title">
      <h2 id="research-title">Selected Research</h2>
      <article class="entry research-entry">
        <h3>When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning</h3>
        <p class="meta"><strong>Accepted · 2nd Workshop on Compositional Learning: Safety, Interpretability, and Agents @ ICML 2026 · Seoul, South Korea · Sole author</strong></p>
        <p>This work studies <strong>latent-space reasoning at recurrent depth</strong> through the problem of temporal abstraction.</p>
        <p><em>How long should a latent reasoner commit to an intent before revising it?</em></p>
        <p>I extend the Hierarchical Reasoning Model with a manager-worker interface in which a slow high-level module emits a <strong>directional subgoal in latent space</strong>. The subgoal represents <strong>medium-horizon intent</strong> and persists across multiple low-level recurrent steps, steering the worker's hidden-state trajectory without specifying an absolute target.</p>
        <p>The central variable is <strong>subgoal persistence</strong>: how long the directional intent remains active before the model re-plans. The results show that intent must persist across enough computational steps for longer-horizon structure to form, while remaining flexible enough to be revised. More frequent re-planning is not necessarily more adaptive.</p>
        <p class="entry-links"><a href="https://arxiv.org/abs/2606.03741">Paper</a><a href="https://github.com/Ayushichadha/scout">Code</a></p>
      </article>

      <article class="entry research-entry">
        <h3>A Score Is Not a Policy: Measuring Whether a Supervisory Module's Decisions Are Worth Making</h3>
        <p class="meta"><strong>Preprint · Under review · Sole author · 2026</strong></p>
        <p>This work studies <strong>learned supervisory control</strong> in hierarchical latent reasoning. When a controller decides whether an internal commitment should persist or be revised, a state-dependent score can look like evidence that the system has learned when to intervene.</p>
        <p><em>State dependence, behavioral adaptation, and decision value are different properties.</em></p>
        <p>A supervisory score can vary with the model's internal state without meaningfully changing its decisions, and a policy can make different decisions across states without improving the final outcome. The paper separates these three levels rather than treating them as equivalent forms of adaptation.</p>
        <p>A second question is <em>whether adaptation is worth learning at all.</em> If a strong non-adaptive policy already captures most of the available value, the more useful control problem may be to learn <strong>when to deviate from a strong prior</strong>, rather than making every decision fully adaptive.</p>
        <p>This reframes supervisory control around <strong>decision value</strong>: how much improvement is actually available from changing the decision, and whether a learned controller can capture it.</p>
        <p class="entry-links"><a href="https://arxiv.org/abs/2609.00874">Paper</a><span>Code to be released</span></p>
      </article>
      <hr class="section-end">
      <p class="closing-line"><em>Together, these projects study a broader problem in reasoning systems: <strong>how computation should be structured over time, when internal intent should change, and how to tell whether learned control over those decisions is genuinely useful.</strong></em></p>
    </section>

    <section id="saint" aria-labelledby="saint-title">
      <h2 id="saint-title">Saint</h2>
      <p class="meta"><strong>Ongoing</strong></p>
      <p>Exploring how the questions in my research translate to agentic systems, particularly <strong>whether another attempt, repair, investigation, or reasoning step is actually worth taking.</strong></p>
    </section>

    <section id="experience" aria-labelledby="experience-title">
      <h2 id="experience-title">Experience</h2>
      <article class="entry experience-entry">
        <h3>Propero</h3>
        <p class="role"><strong>Software Developer · Machine Learning &amp; Applied AI</strong><br>Aug 2022 to Dec 2024</p>
        <p>I worked across machine learning and applied AI, eventually leading the design and technical development of <strong>ShopiBot</strong>, a domain-specific AI system.</p>
        <p>My work covered dense retrieval, RAPTOR, corrective and adaptive retrieval, query analysis and routing, vector-database infrastructure, agentic workflows, automated evaluation and testing, and hallucination mitigation. I also worked on tool-calling systems using LangChain and later Sema4AI and Robocorp as the system evolved.</p>
        <p>Beyond implementation, I worked directly with the CEO on technical direction and product priorities, ran internal technical sessions, and built demos for prospective partnerships. Additional engineers later joined the project for deployment and management while I continued to own the core ML and applied-AI architecture.</p>
      </article>
      <article class="entry experience-entry">
        <h3>Unify</h3>
        <p class="role"><strong>Contributor Program</strong><br>Jul 2024</p>
        <p>Selected for the contributor program at Unify, a London-based Y Combinator-backed company working on model-routing infrastructure. The work connected with experiments I was already doing around query analysis and routing requests across model and retrieval pipelines.</p>
      </article>
    </section>

    <section id="programs" aria-labelledby="programs-title">
      <h2 id="programs-title">Selected Programs</h2>
      <article class="entry"><h3>MATS 2026</h3><p class="role"><strong>Empirical Research Track</strong></p><p>Advanced through the <strong>Research Taste Test</strong> and <strong>Applied AI Assessment</strong>, then progressed to a subsequent written application stage for a Redwood Research stream.</p></article>
      <article class="entry"><h3>Anthropic Fellows Program 2025</h3><p>Advanced to the <strong>technical assessment stage</strong> of the Anthropic Fellows Program. The program stated that its cohort would include 32 fellows.</p></article>
    </section>

    <section id="service" aria-labelledby="service-title">
      <h2 id="service-title">Research Service</h2>
      <p><strong>Invited Reviewer</strong><br>NeurIPS 2026 Workshop on Responsible Use of Meta-Agents</p>
      <p><strong>Reviewer</strong><br>2nd Workshop on Compositional Learning: Safety, Interpretability, and Agents @ ICML 2026</p>
    </section>

    <section id="education" aria-labelledby="education-title">
      <h2 id="education-title">Education</h2>
      <article class="entry"><h3>Northeastern University</h3><p class="role"><strong>MS in Artificial Intelligence · Machine Learning concentration</strong><br>Fall 2026 admission · Spring 2027 start option</p><p>Admitted with the <strong>International Impact Award</strong>, covering 30% of tuition.</p><p>Chose not to enroll at this time and continued my <strong>solo research</strong>.</p></article>
      <article class="entry"><h3>G. B. Pant University of Agriculture and Technology</h3><p class="role"><strong>B.Tech in Electrical Engineering · 2017 to 2021</strong></p><p>First Division.</p><p><strong>Relevant coursework:</strong> Engineering Mathematics I, II, III; Physics I, II; Probability, Statistics &amp; Queuing Models; Introduction to Computers &amp; Programming; Computer Methods in Electrical Engineering; Digital Logic &amp; Circuits; Microprocessors; Circuit Theory; Network Analysis &amp; Synthesis; Control Systems; Advanced Control Systems.</p></article>
    </section>

    <section id="news" aria-labelledby="news-title">
      <h2 id="news-title">News</h2>
      <div class="timeline">
        <div class="timeline-row"><time>Sep 2026</time><p>Invited to give an oral presentation at the <strong>2nd Global Summit on Innovating the Future of Artificial Intelligence (Inno-AI) 2027</strong> in Kuala Lumpur, Malaysia.</p></div>
        <div class="timeline-row"><time>Sep 2026</time><p>Released <em>A Score Is Not a Policy: Measuring Whether a Supervisory Module's Decisions Are Worth Making</em> as a preprint.</p></div>
        <div class="timeline-row"><time>Sep 2026</time><p>Invited to review for the NeurIPS 2026 Workshop on Responsible Use of Meta-Agents.</p></div>
        <div class="timeline-row"><time>Jun 2026</time><p>Advanced through the Research Taste Test and Applied AI Assessment for the MATS 2026 Empirical Research Track.</p></div>
        <div class="timeline-row"><time>Jun 2026</time><p><em>When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning</em> accepted at the 2nd Workshop on Compositional Learning: Safety, Interpretability, and Agents @ ICML 2026 in Seoul.</p></div>
        <div class="timeline-row"><time>May 2026</time><p>Served as a reviewer for the 2nd Workshop on Compositional Learning: Safety, Interpretability, and Agents @ ICML 2026.</p></div>
        <div class="timeline-row"><time>Sep 2025</time><p>Advanced to the technical assessment stage of the Anthropic Fellows Program.</p></div>
        <div class="timeline-row"><time>Jan 2025</time><p>Began focused work on recurrent and latent reasoning, adaptive computation, and hierarchical control.</p></div>
      </div>
    </section>

    <section id="writing" aria-labelledby="writing-title">
      <h2 id="writing-title">Writing</h2>
      <p>I write longer notes and essays on research, reasoning systems, and the ideas shaping my work on Substack.</p>
      <p class="direct-links"><a href="https://substack.com/@ayushi25">Substack</a><a href="{{ '/reading/' | relative_url }}">Selected reading</a></p>
    </section>
  </main>

  <footer class="wrap site-footer"><p>For research, engineering, or collaboration conversations: <a href="mailto:ayushichadha48@gmail.com">Email</a> · <a href="https://www.linkedin.com/in/ayushi-chadha-ai">LinkedIn</a> · <a href="https://x.com/AyushiChadha24">X</a> · <a href="https://github.com/Ayushichadha">GitHub</a></p><p>© 2026 Ayushi Chadha</p></footer>
</body>
</html>
