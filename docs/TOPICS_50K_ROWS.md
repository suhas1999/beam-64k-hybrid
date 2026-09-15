# Topic plan for a 50,000-probe compact dataset

This stage prepares topics only. It does not call Gemini or generate any
conversations.

At 20 probing questions per conversation, 50,000 probing rows require exactly
2,500 conversations. The prepared manifest uses 250 base chatbot use cases
across 29 categories. Each base use case has ten scenario variants designed to
support multi-session memory tests such as changing requirements, preferences,
contradictions, updates, ordering, and temporal reasoning.

## Files

- `topics/50k_rows/2500_topics.json`: complete topic manifest
- `topics/50k_rows/coverage_report.json`: counts and validation summary
- `src/beam/prepare_50k_row_topics.py`: deterministic topic builder

Rebuild the files without making an API call:

```bash
python3 -m src.beam.prepare_50k_row_topics
```

## Coverage design

The manifest covers technical work, physical and biological sciences, health
and education, writing and research, professional and business workflows,
personal organization, household life, travel, food, relationships, hobbies,
recommendations, sports, community work, and administrative navigation.

Each category contains 80 or 90 topics: eight or nine concrete use cases crossed
with ten scenario variants. The round-robin selection keeps all 29 domains in
the final 250-base-topic catalog. Every topic has a unique title and exactly
eight subtopics.
Health, finance, legal, privacy, and security scenarios explicitly constrain
the eventual conversations to educational, organizational, or defensive help
and appropriate professional review.

Generation should begin only after the topic manifest is reviewed and approved.
