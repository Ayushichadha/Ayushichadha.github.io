---
layout: null
title: Ayushi Chadha
description: Independent researcher and software engineer working on recurrent and latent reasoning, adaptive computation, and meta-agent systems.
---
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ayushi Chadha</title>
  <meta name="description" content="Independent researcher and software engineer working on recurrent and latent reasoning, adaptive computation, and meta-agent systems.">
  <link rel="canonical" href="{{ site.url }}/">
  <meta property="og:type" content="website">
  <meta property="og:title" content="Ayushi Chadha">
  <meta property="og:description" content="Research on recurrent and latent reasoning, adaptive computation, and meta-agent systems.">
  <meta property="og:url" content="{{ site.url }}/">
  <link rel="stylesheet" href="{{ '/assets/portfolio.css' | relative_url }}">
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "Person",
    "name": "Ayushi Chadha",
    "url": "{{ site.url }}/",
    "sameAs": [
      "https://github.com/Ayushichadha",
      "https://www.linkedin.com/in/ayushi-chadha-ai",
      "https://substack.com/@ayushi25"
    ],
    "knowsAbout": ["latent reasoning", "recurrent reasoning", "adaptive computation", "meta-agent systems"]
  }
  </script>
  {% include analytics.html %}
</head>
<body>
  <header class="site-header">
    <div class="wrap header-inner">
      <a class="site-name" href="#top">Ayushi Chadha</a>
      <nav aria-label="Primary navigation">
        <a href="#research">Research</a>
        <a href="#news">News</a>
        <a href="#experience">Experience</a>
        <a href="#education">Education</a>
        <a href="#writing">Writing</a>
      </nav>
    </div>
  </header>

  <main id="top" class="wrap">
    <section class="intro" aria-labelledby="intro-title">
      <h1 id="intro-title">Ayushi Chadha</h1>
      <p class="intro-lead">I am an independent researcher and software engineer interested in how reasoning systems use computation internally: how latent states evolve, when an internal goal should persist, when it should be revised, and how those decisions can be learned.</p>

      <p>I began working seriously on latent reasoning in January 2025. The path grew from Andrej Karpathy’s idea of a cognitive core and from a broader question about human reasoning: people often reach useful abstractions with limited examples, imperfect memory, and computation that is not fully verbalized. Reading cognitive science alongside work on hierarchical reinforcement learning, recurrent computation, chain-of-thought, and abstraction shaped the models I wanted to study and the experiments I wanted to run.</p>

      <p>My first paper studies persistence and adaptive re-planning in a hierarchical recurrent reasoner. My current work asks what happens when learned control is placed inside a reasoning loop, and how we should evaluate a controller when its internal decision changes without improving the final outcome. I am also exploring meta-agent systems that revise prompts, programs, and agent harnesses, with care about separating changes in an internal score from changes in decisions and task performance.</p>

      <p>Before this research, I spent several years building software and applied AI systems at Propero. I moved from browser automation and machine learning experiments to leading the design and engineering of ShopiBot, a retrieval-based assistant for Shopify developers. The role included research, system design, evaluation, developer workflows, internal seminars, and partnership demos. It taught me to connect an uncertain technical idea to a product that other people could test and use.</p>

      <p>I am looking for research and engineering work on reasoning, adaptive computation, agents, and systems that learn how to improve their own problem-solving process.</p>

      <p class="direct-links">
        <a href="mailto:ayushichadha48@gmail.com">Email</a>
        <a href="{{ '/assets/Ayushi_Chadha_CV.pdf' | relative_url }}">CV</a>
        <a href="https://github.com/Ayushichadha">GitHub</a>
        <a href="https://www.linkedin.com/in/ayushi-chadha-ai">LinkedIn</a>
        <a href="https://substack.com/@ayushi25">Substack</a>
        <a href="{{ '/about/' | relative_url }}">Research journey</a>
      </p>
    </section>

    <section id="research" aria-labelledby="research-title">
      <h2 id="research-title">Selected research</h2>

      <article class="entry research-entry">
        <h3>When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning</h3>
        <p class="meta"><strong>Accepted, Compositional Learning Workshop at ICML 2026</strong> · Seoul, South Korea · Sole author · 2026</p>
        <p>This work augments a hierarchical recurrent reasoner with a learned controller that decides whether a high-level latent plan should persist or be revised. The central question is whether re-planning can respond to the state of an internal computation instead of following a fixed schedule.</p>
        <p class="entry-links">
          <a href="https://arxiv.org/abs/2606.03741">Paper</a>
          <a href="https://github.com/Ayushichadha/scout">Code</a>
        </p>
      </article>

      <article class="entry research-entry">
        <h3>Beyond the Clock: Measuring the Value of Adaptive Revision</h3>
        <p class="meta"><strong>Under review</strong> · Sole author · 2026</p>
        <p>This paper examines a cautionary result from learned re-planning: a controller can produce a varying score while still making nearly constant decisions, and those decisions may not improve the final task outcome. The work separates score variation, behavioral adaptation, and performance improvement rather than treating them as the same result.</p>
        <p>The distinction may also matter for compound agentic systems and meta-agents that modify prompts, programs, or agent harnesses. A system can appear adaptive because an internal evaluator changes while its interventions remain narrow or ineffective. This is a broader research direction; the experiments in this paper are limited to a hierarchical latent reasoner.</p>
        <p class="entry-links"><a href="https://arxiv.org/abs/2609.00874">Paper</a><span>Code to be released</span></p>
      </article>
    </section>

    <section id="news" aria-labelledby="news-title">
      <h2 id="news-title">News</h2>
      <div class="timeline">
        <div class="timeline-row"><time>Sep 2026</time><p>Accepted an invitation to give an oral presentation at the INNO AI Summit in Kuala Lumpur, Malaysia.</p></div>
        <div class="timeline-row"><time>Sep 2026</time><p>Nominated to serve as a reviewer for the NeurIPS 2026 Workshop on Meta-Agents.</p></div>
        <div class="timeline-row"><time>Sep 2026</time><p>Released <em>Beyond the Clock</em>, a study of evaluation and learned control in hierarchical latent reasoning.</p></div>
        <div class="timeline-row"><time>Jul 2026</time><p>Admitted to the MS in Artificial Intelligence at Northeastern University with the International Impact Award, covering 30% of tuition. I chose not to enroll and continued my independent research.</p></div>
        <div class="timeline-row"><time>Jun 2026</time><p>Advanced through the MATS empirical research assessments and was invited to complete the next written application stage for a Redwood Research stream. This was an application-stage selection, not a fellowship appointment.</p></div>
        <div class="timeline-row"><time>May 2026</time><p><em>When to Re-Plan</em> was accepted at the Compositional Learning Workshop at ICML 2026 in Seoul, South Korea.</p></div>
        <div class="timeline-row"><time>Sep 2025</time><p>Advanced to the technical assessment stage of the Anthropic Fellows Program. The application stated that the cohort would include 32 participants; this was an assessment-stage selection, not a fellowship appointment.</p></div>
        <div class="timeline-row"><time>Jan 2025</time><p>Began independent research on recurrent and latent reasoning, adaptive computation, and hierarchical control.</p></div>
        <div class="timeline-row"><time>Jul 2024</time><p>Selected for the contributor program at Unify, a London-based, Y Combinator-backed startup working on model routing.</p></div>
      </div>
    </section>

    <section id="experience" aria-labelledby="experience-title">
      <h2 id="experience-title">Experience</h2>

      <article class="entry experience-entry">
        <div class="entry-heading">
          <div><h3>Propero</h3><p class="role">Software Engineering Intern, then Software Developer</p></div>
          <p class="date">Aug 2021 to Dec 2024</p>
        </div>
        <p>I joined as a software engineering intern in August 2021 and moved into a software developer role in January 2022. My early work covered browser automation, DOM-tree representations, and machine learning experiments for more robust automation.</p>
        <p>I later led the design and engineering of ShopiBot, an assistant grounded in the Shopify developer domain. I worked across problem definition, retrieval pipelines, query analysis and routing, evaluation, testing, and the developer experience. I also studied and tested contemporary retrieval techniques, gave internal seminars during the company’s move toward AI-based automation, and built technical demos for prospective startup partnerships in the United States.</p>
        <p>The work required balancing research questions with product and business constraints. I worked directly with the CEO on priorities, partnerships, and the pace between experimentation and delivery.</p>
      </article>

      <article class="entry experience-entry">
        <div class="entry-heading"><div><h3>Unify</h3><p class="role">Contributor Program</p></div><p class="date">Jul 2024</p></div>
        <p>Selected for a contributor program at a London-based, Y Combinator-backed startup building model-routing infrastructure. The work connected with my existing experiments on query analysis and routing requests to different retrieval pipelines.</p>
      </article>

      <article class="entry experience-entry">
        <div class="entry-heading"><div><h3>Techniche</h3><p class="role">Writer, College Technology Magazine</p></div><p class="date">Aug 2020</p></div>
        <p>Wrote for the college technology magazine, translating technical ideas for a broader student audience.</p>
      </article>

      <article class="entry experience-entry">
        <div class="entry-heading"><div><h3>Oil and Natural Gas Corporation</h3><p class="role">Engineering Intern</p></div><p class="date">Jun 2020</p></div>
        <p>Completed an engineering internship at ONGC.</p>
      </article>

      <article class="entry experience-entry">
        <div class="entry-heading"><div><h3>Engineering and Technical Society</h3><p class="role">Student Executive Body</p></div><p class="date">Aug 2018</p></div>
        <p>Selected for the student executive body.</p>
      </article>

      <article class="entry experience-entry">
        <div class="entry-heading"><div><h3>Science and Literary Bureau</h3><p class="role">Student Organizer</p></div><p class="date">Aug 2018</p></div>
        <p>Selected to help organize national-level events in science, technology, engineering, and literature.</p>
      </article>
    </section>

    <section id="education" aria-labelledby="education-title">
      <h2 id="education-title">Education</h2>

      <article class="entry education-entry">
        <div class="entry-heading"><div><h3>Northeastern University</h3><p class="role">MS in Artificial Intelligence, Machine Learning concentration</p></div><p class="date">Fall 2026 admission</p></div>
        <p>Admitted with the International Impact Award, a scholarship covering 30% of tuition. I chose not to enroll and continued my independent research.</p>
      </article>

      <article class="entry education-entry">
        <div class="entry-heading"><div><h3>G. B. Pant University of Agriculture and Technology</h3><p class="role">BTech in Electrical Engineering</p></div><p class="date">2017 to 2021</p></div>
        <p>First Division, 73.2%.</p>
      </article>
    </section>

    <section id="writing" aria-labelledby="writing-title">
      <h2 id="writing-title">Writing and notes</h2>
      <p>I am organizing notes from the books, papers, experiments, and research questions that shaped this work. I will publish longer essays and technical reflections on Substack.</p>
      <ul class="link-list">
        <li><a href="https://substack.com/@ayushi25">Substack</a></li>
        <li><a href="{{ '/about/' | relative_url }}">Research journey</a></li>
        <li><a href="{{ '/reading.html' | relative_url }}">Selected reading</a></li>
      </ul>
    </section>
  </main>

  <footer class="wrap site-footer">
    <p>© 2026 Ayushi Chadha</p>
  </footer>
</body>
</html>
