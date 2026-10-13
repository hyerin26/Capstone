# Follow-up protocol: answer scoring and visual/response controls

Frozen before follow-up results are inspected. Reuse all 500 saved scenes (250
original pairs, both swaps), unchanged Magma revision and greedy text generation.
This is a static tile-selection experiment, not a robot action evaluation.

## Study 1: scoring repair and actual decisions under occlusion

Repeat the exact original unmarked LEFT/RIGHT question for the original RGB and
three original interventions: left tile gray, right tile gray, and seed-0 scattered
background gray with the target tile's pixel count. Preserve intervention pixels
so the scoring change is not confounded by a different occlusion protocol.
Record actual generated text after every intervention. Record first-answer-token
log probabilities for LEFT/RIGHT, Left/Right and left/right separately. Also sum
probabilities across the three case variants per side (logsumexp). All labels must
be one token and distinct. This marginal is restricted to those six tokens, not
the probability of every possible natural-language response. Report restricted
choice mass and invalid free responses rather than interpreting it as total
semantic confidence. Validate generation-returned scores against a plain model
forward pass on the pilot. Retain the old uppercase metric for comparison.

Primary: free-response accuracy and decision flips. Secondary: each spelling's
ranking accuracy/agreement with actual generation, and corrected target-minus-
distractor margin changes. Gray occlusion and scattered background topology
remain limitations, not causal proofs.

## Study 2: visual layout and response controls

Visual conditions are original camera RGB versus two perspective-rectified tiles
on an equal-size neutral canvas. Use observed RGB pixels and segmentation masks,
not higher-quality source textures. No internal mirroring of clock/sign images.
Both canvases remain 640x480, with identical native Magma preprocessing/token budget.
Equalized tiles are 256x256 in equal slots; equalization jointly changes perspective,
scale and background, so this experiment identifies the combined presentation
effect, not a unique cause among those factors.

Conditions for every scene:
- original_lr, equal_lr: unmarked image, original LEFT/RIGHT question (1 each).
- original_ab, equal_ab: marked A/B image, ask only A or B (2 label mappings each).
- original_marked_lr, equal_marked_lr: exactly the same marked pixels as the A/B
  conditions, but ask the original LEFT/RIGHT question (2 label mappings each).

Mappings AB and BA mean A-left/B-right and B-left/A-right. Every scene receives
both mappings, avoiding confounding a label preference with a spatial preference.
Marks have identical style and do not encode the answer. Report whether predictions
follow label identity or physical side when mappings are reversed.

Thus 5,000 baseline predictions + 1,500 additional occluded original_lr predictions
= 6,500 generations. These are repeated measurements of 250 original pairs, not
6,500 independent examples. Primary comparison: free-response accuracy, accuracy
by target side, right-side selection rate, and both-swaps correctness. Separate
marked-LR controls let us compare response language on identical marked images.

## Image QA

Derive a four-corner convex hull approximation from each whole-tile actor mask.
Record mask/quad overlap, quadrilateral coordinates and image hashes. Stop if a
credible quadrilateral is unavailable; do not silently replace it with a box.
Warp from RGB using the same corner order on both tiles. Inspect category examples
before inference. Residual occlusion, lighting and resampling remain visible.

## Execution and analysis

Use the original verified checkpoint loader including LayerScale restoration.
First run a category-spanning pilot, validate score equivalence and free baseline
agreement, then run the unchanged protocol for all scenes. Each GPU gets one
independent model replica if memory permits. Atomic case files and method hashes
support resumption without mixing code versions. Never modify original results.
Save every prompt/input ID, response, label mapping, token score and intervention.
Save native pixel-value shape/image_sizes to verify preprocessing equivalence.

Use per-category and macro/micro summaries. For paired condition differences,
bootstrap original pairs with all swaps and label mappings kept together (2,000
draws, seed 20261014). Treat category analyses and comparisons as exploratory;
avoid selecting best prompts/layers from this test set. Counting/Symmetry's
explicit-object cases are color-cue selection, not reasoning tests.
