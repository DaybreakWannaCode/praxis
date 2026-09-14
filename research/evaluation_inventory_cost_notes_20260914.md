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

## Outcome-blind annotation audit, September 15

A fresh audit of all 1,240 annotations against the old 48 image IDs and all 10,000 released text rows leaves 1,069 eligible annotation rows. Exclusion reason counts (overlapping): 102 duplicate URLs or normalized descriptions, 61 existing-panel IDs/URLs/descriptions, 23 invalid annotations, and 4 non-HTTPS/invalid links. Some descriptions are numeric missing values; these are explicitly excluded, not converted to text. No exact normalized scene/QA overlap with the full text source was detected. The source files are hashed in the audit.

The script produces a fixed hash-ranked eligible list without inspecting any model outcomes. This is not a split freeze or a claim of 1,069 independent images. Image downloads, byte/content duplicate checks, near-duplicate and scene review, and explicit development/test assignments remain required. No new evaluation images were downloaded by this audit. Proposed next data step: acquire an outcome-blind bounded development candidate pool, while reserving disjoint annotation candidates for final test; audit image identities before choosing the final 256 development scenes. Do not inspect endpoint outcomes to decide eligibility or panel size.

Evidence: `/workspace/praxis/data/independent-visual-inventory-20260915/annotation-audit.json`, backed up locally under `runs/independent-visual-inventory-20260915`.

## Development image acquisition and content audit, September 15

Before any new outcome evaluation, the first 512 annotations in the frozen hash order were assigned to the development acquisition pool. The remaining 557 were reserved and not downloaded. This is a candidate partition, not a claim of final scene independence. The intended endpoint panel remains 256 scenes, conditional on review. Acquisition used four CPU/network workers, a 2,000,000-byte per-image cap, a 1,024,000,000-byte total image bound and a 45-minute job cap, after verifying a 16 GiB persistent-volume reserve.

Acquisition completed in 67.27 seconds: 451 valid images, 61,072,409 bytes, 61 failed attempts. Failure categories: 52 HTTP errors, 5 URL errors, 1 invalid image, 2 validation/size errors, and 1 timeout. Missing images are recorded rather than replaced based on model outcomes. Availability and size restrictions can affect the eventual evaluation distribution and must be reported.

The independent content audit rechecked all downloaded bytes and decoded pixel identities. Pixel equality and dHash distance at most six were screened against the old 48-image panel and within the new pool. One cross-panel near-duplicate was flagged: viva-121 versus viva-831. Direct visual review confirmed the same photograph of an overturned car behind a driving-safety sign, with resizing/watermark differences. Exclude viva-831 from new development and final test. This leaves 450 acquired candidates not flagged by this screen. Hash screening can miss crops, related frames and shared underlying scenes; remaining image/annotation review and the final panel freeze are outstanding.

All 451 image files and 512 receipts were backed up in a local archive, and their individual checksums were verified against the persistent audit. Raw images remain outside Git. The acquisition plan, content audit, local backup receipt and duplicate decision are saved under `runs/independent-visual-inventory-20260915` locally and `/workspace/praxis/data/independent-visual-inventory-20260915` on the pod. No new model endpoint was evaluated and no final-test image was fetched.
