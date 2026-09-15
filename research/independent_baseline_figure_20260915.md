# Independent baseline figure

Caption: Adapted text-only Praxis baseline on 256 development images. Left: greedy visual correctness before training (201/256) and after 32 text-only updates on 1,024 prompts (203/256). Right: paired accuracy change of +0.78 percentage points, with the locked 10,000-draw paired image-bootstrap 95% interval [−1.95, +3.91]. Eight images improved and six worsened; 242 retained their correctness status. The gain is unresolved. The interval is conditional on one development panel and one training trajectory and does not measure between-seed uncertainty. This is an adapted baseline, not a full reproduction of the reported Praxis setting or evidence of beneficial alignment selection.

Artifacts: `runs/baseline-figure-20260915/baseline.png`, `.pdf`, `.svg`, and `provenance.json`. The figure reads the audited `paired-analysis.json` directly, verifies consistency of transition counts and accuracies, and does not refit, select or recompute the interval. Provenance records the analysis hash, plan hash, renderer hash, plotting version and output hashes. The PNG was visually inspected for legibility, clipping and consistency with the audited numbers. PDF and SVG are exports of the same Matplotlib figure.

Regenerate in an environment containing Matplotlib:

```bash
python research/scripts/plot_independent_baseline.py runs/independent-baseline-endpoints-20260915/paired-analysis.json runs/baseline-figure-20260915-new
```

The output directory must be new; existing artifacts are not overwritten. Raw response files and checkpoint data remain outside Git. This figure represents existing evidence only and does not change any outcome or experimental endpoint.
