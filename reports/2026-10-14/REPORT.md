# Magma explicit-object grounding results

Completed: 500 cases. Micro accuracy: 74.2%; macro accuracy: 77.4%.
Candidate-ranking accuracy: 57.4%. Invalid responses: 0. Valid choices: LEFT 137, RIGHT 363.

| Category | N | Accuracy | 95% pair-bootstrap CI | Candidate accuracy | Invalid | Swap consistency |
|---|---:|---:|---:|---:|---:|---:|
| attribute | 40 | 60.0% | 47.5%–72.5% | 57.5% | 0 | 40.0% |
| celebrity | 100 | 68.0% | 62.0%–75.0% | 52.0% | 0 | 36.0% |
| counting | 20 | 100.0% | 100.0%–100.0% | 75.0% | 0 | 100.0% |
| emotion | 20 | 65.0% | 50.0%–80.0% | 55.0% | 0 | 30.0% |
| living_world | 100 | 91.0% | 84.0%–96.0% | 66.0% | 0 | 86.0% |
| public_info | 20 | 75.0% | 60.0%–90.0% | 55.0% | 0 | 50.0% |
| state | 60 | 58.3% | 46.7%–70.0% | 51.7% | 0 | 43.3% |
| symmetry | 20 | 100.0% | 100.0%–100.0% | 70.0% | 0 | 100.0% |
| time_clock | 20 | 90.0% | 75.0%–100.0% | 50.0% | 0 | 80.0% |
| traffic | 100 | 67.0% | 60.0%–74.0% | 54.0% | 0 | 38.0% |

## Interpretation

This is a static RGB + text LEFT/RIGHT task, not robot action success. Invalid free responses count as incorrect; candidate accuracy ranks the two label-token scores separately.
Swap consistency measures changing side under a mirrored pair; a consistently wrong target can also satisfy it. Attention is from the last prompt position, averaged over heads; every layer is exported in layers.csv.
Masks cover the whole tile. Attention and gray-tile occlusion do not establish recognition of the object inside the tile. Positive occlusion margin drop means the intervention reduced the target-versus-distractor log-probability margin. The scattered background control differs in topology.
Counting and Symmetry explicit-object prompts use color cues. Category sample sizes differ. Confidence intervals resample original pairs (2,000 draws).
