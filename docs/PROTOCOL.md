# Explicit-object grounding protocol (2026-10-14)

## Population

Use all 500 recorded Explicit object cases (250 original pairs, both layouts)
from the supplied ten-category seed-0 bundle. No semantic-reasoning or spatial
instruction rows are mixed into this run. Category counts are 40 attribute,
100 celebrity, 20 counting, 20 emotion, 100 living_world, 20 public_info,
60 state, 20 symmetry, 20 time_clock and 100 traffic.

The bundle's own SHA256 fields and swapped labels are checked before rendering.
The source CSV is UTF-8 with BOM. The original condition/instruction is preserved.
No answers, filenames, category names or segmentation masks are fed to the model.

## Scene and question

Capture the initial policy RGB after ten existing grasp steps from the available
YeahPick Act2AnswerV4 simulator. Use the bundle's GLBs through its dataset path,
not placeholders. Record simulator commit and the RGB hash. This recreates scenes
with the available simulator; it is not evidence that all historical simulator
versions and random states are identical to the earlier team runs.

Query the target tile of the recorded instruction, requiring LEFT or RIGHT in
viewer coordinates. This is a text-answer evaluation and is not an action rollout.
Environment-generated instruction text is recorded separately because the
available older simulator may format it differently from the bundle CSV.

## Attention definition

Capture self-attention from the final prompt position predicting the first answer
token to visual keys at every language-model layer and head. Use eager attention;
discard the unused quadratic weight matrices immediately after copying that row.
The capture hook does not alter the attention output vectors. Check candidate
scores and greedy first token with capture disabled on pilot inputs.

Reconstruct the crop grid from actual projector shape and image_sizes. Remove
one row-separator token per spatial row. Assert expanded sequence length and
attention-row sums. Save spatial head maps rather than all attention matrices.
Only the current crop-based Magma layout is supported; fail on unfamiliar layouts.
Report total spatial-image attention alongside region-conditioned fractions.

Ground-truth masks are renderer actor-ID masks of whole candidate tiles. They
are NOT boxes/masks of objects depicted inside photographs. Region metrics use
fractional cell occupancy; density divides mass by occupied cell area. The main
summary averages all layers and heads. Layers 1,8,16,24,32 are fixed visualization
panels. Peak-hit uses head-average maps and >=0.5 tile occupancy at the peak cell.

## Predictions and controls

Report strict free-response accuracy (up to eight generated tokens) separately
from candidate ranking using single-token LEFT/RIGHT log probabilities. Retain
invalid responses. No silently sanitized NaN/Inf logits are allowed.

Occlude left tile, right tile, and an equal-target-area scattered background
control with gray pixels. Report original-to-occluded correct-minus-incorrect
log-probability margin changes and masked pixel counts. Tile areas can differ;
the scattered control also differs in topology. These interventions are diagnostic
and do not establish causality or rule out out-of-distribution artifacts.

## Aggregation and reproducibility

Preserve each raw response, prompt, input IDs, configuration, model revision,
mask/RGB paths, attention maps and timings. The method signature guards against
mixing protocols when resuming. Publish macro and micro accuracy, per-category
accuracy, invalid rate, candidate-ranking accuracy, swap consistency and attention
metrics. Resample original pairs (both layouts together) for category confidence
intervals: 2,000 bootstrap draws, fixed seed 20261014. Same-image generation seeds
are not treated as independent experimental replicates.

A pipeline status of `waiting_for_inputs` or `waiting_for_scenes` is preparation,
not completed inference. `pilot_running` and `main_running` are distinct. Report
completion only if all 500 result files exist and summarization succeeds.
