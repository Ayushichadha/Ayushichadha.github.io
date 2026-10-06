# Ayushi Chadha — personal website

Jekyll website hosted at https://ayushichadha.github.io.

- `index.md`: self-contained homepage, including its styling and section navigation. `layout: null` prevents a second theme wrapper; `permalink: /` preserves the homepage URL.
- `assets/Ayushi_Chadha_CV.pdf`: current downloadable CV.
- `research.md`: existing detailed research notes at `/research/`.
- Other existing pages and assets remain available at their original URLs.

The homepage needs no JavaScript build or extra dependency. GitHub Pages continues using the existing Jekyll setup. Update publication statuses, dates, and writing links in `index.md` as needed.


## Search and analytics

Each portfolio page has its own title, description, canonical URL, and social metadata. The homepage includes Person structured data. `sitemap.xml` includes titled Jekyll pages and `robots.txt` advertises it.

Analytics is disabled. To enable GA4, create a property and web stream for https://ayushichadha.github.io, then add its public `G-...` measurement ID to `google_analytics` in `_config.yml`. Review consent and privacy configuration for the selected setup before enabling. The production-only include is on the three custom portfolio pages; local previews do not collect visits. Advertising signals are disabled. Visits are not visitor names, and past visits cannot be recovered retroactively.

Verify the site in Google Search Console and submit https://ayushichadha.github.io/sitemap.xml after deployment. Ownership verification requires the account holder’s verification token or method. Neither service has been connected by this change.
