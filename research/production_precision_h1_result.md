# Completed H=1 precision check

The frozen run completed with exit code 0 and 1,800 response records. The parent replay matched exactly. This is a fixed-parent development precision result, not evidence of population transfer.

Elapsed measurement time: 6.876 hours. Peak torch allocation: 35.306 GiB.

## Candidate outcome contrasts

Pointwise 95% fixed-panel intervals from the frozen paired sign-category Chernoff-KL procedure. Units are percentage points.

| Candidates (zero-based) | Difference | 95% interval |
|---|---:|---:|
| 0 minus 1 | 3.91 | [-6.25, 13.90] |
| 0 minus 2 | 1.95 | [-8.75, 12.57] |
| 0 minus 3 | 5.47 | [-4.41, 15.11] |
| 1 minus 2 | -1.95 | [-12.76, 8.95] |
| 1 minus 3 | 1.56 | [-8.83, 11.88] |
| 2 minus 3 | 3.52 | [-5.58, 12.46] |

## Frozen decision

Jointly resolved pairs: 0/6; required: 4/6. Decision: test the predeclared H=4 alternative once. All six outcome intervals contain zero and extend outside the practical-tie band of +/-2 percentage points.

## Completed fallback

The predeclared H4 fallback subsequently completed with all 1,800 responses,
exact replay, exit zero and a verified archive. It also resolved 0/6 pairs.
See [the H4 report](production_precision_h4_result.md) for the combined decision:
this bounded budget is insufficient; no main sweep or adaptive retry follows.
