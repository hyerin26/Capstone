# Capstone

Magma explicit-object grounding and attention analysis on the YeahPick bundle.

## Status

Data auditing is implemented. GPU inference and attention extraction are not yet
implemented or run. GitHub publication and current VESSL connectivity are pending.

## Data validation

The source bundle remains outside Git. Run:

```sh
python3 scripts/audit_bundle.py --bundle /path/to/YeahPick_10categories_prompts_glb
```

The audit checks GLB SHA256, unique cases, answer labels, and swapped asset pairs.
It writes the 500 explicit-object cases and a summary under `outputs/audit`.
The bundle contains 250 pairs across 10 categories, with two layouts per pair.
Counting and Symmetry explicit-object prompts use color cues, not numerical or
symmetry reasoning. Category sample sizes differ; report per-category and macro
averages, and bootstrap by original pair rather than individual layout.

## Planned protocol

Use the recorded instruction and an explicit LEFT/RIGHT response request with
actual rendered RGB inputs. Save the complete model prompt and model revision.
Do not mistake the original action prompt for an identical text-answer task.
Acquire simulator code and camera configuration before claiming replication of
robot-camera images. The provided runner snapshots depend on external code.

Extract visual-key attention from the last prompt position predicting the first
answer token. Retain per-layer summaries, image-token mappings, padding/crop
metadata, and masks from the renderer. Never infer a square spatial token grid
without checking the actual model preprocessing. Validate attention extraction
against normal inference, and mark backend disagreements.

Evaluate selection accuracy, invalid outputs, swap consistency, visual attention
mass, region-area-corrected attention density, and target/distractor/background
occlusion score changes. Attention alone is not evidence of causal recognition.
Pilot before full evaluation; freeze analysis choices before inspecting full-run
outcomes. Store full attention matrices only when necessary for memory reasons.
