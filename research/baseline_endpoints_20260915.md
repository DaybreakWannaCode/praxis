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

## Launch record

Plan SHA-256 `8f0db76d4e2c6ddc3f4d865bf70aae7c625791ceaa802b49d86048750f4d99bc` freezes 88 artifacts. Its local backup and the three deployed evaluator module hashes were independently checked. Initial evaluation was submitted to tmux `praxis-baseline-initial`; launcher PID 74651 and quota-preflight PID 74653 were verified live. This is a launch record, not completed evaluation. Final endpoint is not launched. Monitor the existing process and persistent logs before any restart.

## Initial runtime evidence

The initial endpoint passed the quota guard (214,516,974,592 accounted bytes) and was verified live as PID 74687. A local checksum-verified partial archive contains the first 31 responses plus the plan and endpoint manifest (245,760 bytes total). All backed-up responses include sequence tokens and prompt lengths. Their mean generation time is 9.187 seconds, median 9.600 and maximum 12.573; startup was 34.243 seconds. Extrapolating 289 generations gives about 44.8 minutes for this endpoint, conditional on this partial timing sample. Final-checkpoint response timing remains unknown. No endpoint accuracy analysis or selection decision was made from these partial outputs. The two-hour endpoint cap remains unchanged.

## One-time audited handoff

Controller revision `efe8058` passed live preflight and is running in tmux `praxis-baseline-handoff` as PID 75130. It binds the initial PID 74687 and creation time 1789418346.07, plan hash, launcher hash and independent auditor hash. It waits for actual process termination and the successful exit receipt, then audits all 288 archived responses and summary counts against frozen labels before launching only the final endpoint. The final launcher retains its GPU/quota checks and two-hour cap; the controller additionally bounds final-launch wall time and stops owned descendants on timeout. No automatic retries, full selection sweep or pod stop are included. The controller receipt is not proof of liveness; inspect the process and exit state before recovery. Initial progress was 124/288 at controller verification.

## Pretrained endpoint complete; trained endpoint live

Initial evaluation exited 0 after 2,634.357 seconds (43.91 minutes), saved all 288 responses and passed exact terminal replay. Correct-image development accuracy is 201/256 (78.515625%); 254/256 responses parsed and none truncated. The shuffled-control endpoint produced 21/32 correct, 31 parsed and no truncations. Do not subtract these unequal-panel rates to estimate image dependence: the planned analysis compares the same 32 targets. This is the pretrained baseline, not a measured training gain.

The one-time controller independently audited the frozen labels, identities and all response summaries, then launched final evaluation. Its endpoint PID 75629 and timeout PID 75628 were verified live. The final quota check accounted for 214,520,468,480 bytes against the 250 GB limit. No new model or gradient files were generated by the initial evaluation.

The complete initial archive is 2,129,920 bytes, SHA-256 `1266f2b10144f13c021cef30d49f9e6ce64c00e89aa53a2b35bea0fcc0a8fe94`. Local verification checked 295 archived files and independently repeated the 288-response audit; response hashes matched the remote audit. The final endpoint must still finish and pass its own audit before the paired change is reported.
