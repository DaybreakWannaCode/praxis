# Static selection manifest preparation

`selection_manifest.py` builds Alignment and Random batch memberships from a complete fixed score pool before independent outcomes are inspected. It requires a shared parent, scoring protocol, calibration split and likelihood objective; distinct candidate/prompt IDs; finite scores; and per-candidate receipt hashes. Extra fields, including outcomes, are rejected. Higher alignment wins even when every score is negative. Fixed hash ordering breaks ties and controls membership/order reproducibly.

The default proposal is 64 candidate batches, 32 selected batches and 32 prompts per batch. Seeds must be supplied explicitly; this does not freeze their choice. Alignment membership is static across training seeds. Random membership depends on seed and candidate identity, never alignment values. Every arm receives the same prompt count and a seed-defined batch order. Training must still generate fresh rollouts; no tentative child is adopted as a trained arm.

The saved manifest includes the canonical score-pool hash and cannot overwrite an existing path. Three local tests passed: negative-score ordering and input-order invariance; random membership unchanged by score reversal; and rejection of outcomes, mixed parents, repeated prompts and output overwrite. These are fixture tests; no real pool or selected dataset was generated.

Upstream must validate the actual score receipts, input/code identities and scene separation, then archive the exact score pool before using the manifest. Hash strings alone do not prove those conditions. Pool membership, seed choices, scoring objective, independent endpoint, compute allocation and storage remain to be frozen/authorized. This module does not launch runs or establish a selection benefit.
