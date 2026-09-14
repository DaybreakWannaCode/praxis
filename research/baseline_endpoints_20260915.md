# Independent development baseline endpoints

The reviewed development panel is frozen at 256 correct-image examples and 32 fixed shuffled-image controls. Panel SHA-256: `ede6cd3af5199f89c7bb806e4adb5a6aadb6b08e01bf5b42d51b4f41e7b89439`. Membership was fixed after baseline training but before these endpoint outcomes. This is development evidence, not an untouched final test or pretraining preregistration.

The next measurement compares the pinned pretrained Qwen2.5-VL-3B-Instruct revision `66285546d2b821cf421d4f5eb2576359d3770cd3` with the ordinary baseline's audited step-32 actor. The old preparation configuration points to a trained step-2 actor and must not be reused as the initial model. Explicit pretrained mode loads original weights directly, avoiding another full initial checkpoint on disk.

The runner uses greedy decoding, a 512-token cap, `explicit_final_v3`, the same visual prompt and image preprocessing at both endpoints, and no situation description in the model input. Each endpoint saves 288 responses individually and checks one exact greedy replay. Interrupted runs verify frozen artifacts and the first saved response before continuing. These checks support recoverability; they do not replace real GPU validation of the new initial-loading path.

Before launch, freeze the exact source snapshot, all pretrained model files, final checkpoint, canonical coordinates, panel, prompt, parser and budget in one hashed plan. Keep the live scorer's source snapshot unchanged. A two-hour per-endpoint execution cap bounds this measurement; 578 generations including terminal replay are planned across both endpoints, plus any documented resume checks. Timing remains an estimate until this panel is measured.

Three orchestration tests passed: interrupted/resumed response preservation, changed model artifact rejection, and changed saved item identity rejection. They use a fake backend and do not establish model correctness. No independent endpoint has been launched by this change. The full selection matrix remains unlaunched.

Analyze paired correct-image accuracy, parse failures and truncation, and the 32-example image-shuffling comparison. Treat image groups as scene proxies and disclose assistant-only visual review. A single adapted baseline trajectory cannot establish selection benefit.

## Locked analysis and storage guard

Primary reporting is the paired accuracy difference on all 256 correct-image examples, with the four before/after correctness transition counts. Report a paired image-bootstrap 95% interval using 10,000 draws and fixed seed 20260915, explicitly conditional on this panel and single training trajectory. Report parser and truncation counts separately; do not remove failed parses or truncated responses after observing outcomes. For the preselected 32 controls, report correct-minus-shuffled accuracy at each model and its change. These small control estimates are diagnostic. Scene-proxy grouping is not verified independence of underlying scenes.

The shared mount reports cluster-wide capacity, so the launcher counts unique file inodes using the larger of logical and allocated bytes against the recorded 250,000,000,000-byte purchased quota. It preserves 16 GiB of headroom. No historical deletion is performed.
