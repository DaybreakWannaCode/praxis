# Evaluation inventory and cost limitations

Read-only inspection of the existing pod's source files on September 14 found:

- `VIVA_annotation.json`: 1,240 annotation rows, 1,240 unique indices, 1,240 nonempty image URLs, 1,235 distinct URL strings.
- `sources/images`: 48 files. The download log contains 48 entries.
- The active visual manifest has 50 items: two text training items, 16 image-score items and 32 image-development items. Thus the currently downloaded images do not constitute an untouched test panel.

Annotation counts are not counts of downloaded, usable independent scenes. Repeated URLs already reduce the inventory; near-duplicates, invalid links, source-scene overlap and image validity may reduce it further. Do not fill missing images by reusing score/development scenes in test.

Before a 256-scene ordinary before/after development evaluation or final test can run, acquire additional images and audit their identities and split assignments without inspecting model outcomes. Record this as engineering/data-preparation time and possible transfer time, separate from model inference. The source text eligibility audit currently excludes exact overlaps against the old visual manifest only; expanding the image panels also requires an overlap check against the selected text rows. This may reduce the eligible evaluation pool.

The earlier 1,000-scene timing projection is a scaling scenario, not a promise of 1,000 independent test scenes. Reserving 16 distinct score images and 256 distinct development images from 1,235 distinct URLs would leave at most 963 other source URLs, even before other exclusions.

For a concrete cost-planning scenario (not a frozen split), 256 final scenes for one common parent plus nine trained models means 2,560 final responses. Two ordinary-baseline endpoints on 256 development scenes plus 32 shuffled-image controls each add 576 responses. If each endpoint's correct-image and shuffled-image evaluation shares one loaded model, this is 3,136 responses and 12 model loads in total. Count endpoint timing/replay checks, data acquisition, historical work and reruns separately. The available warm-parent rate is 8.176 seconds per response and 226.315 seconds per load; endpoint length changes remain unmeasured.

No new image panel has been declared frozen or evaluated by this inventory check. No final-test model outcomes have been inspected.
