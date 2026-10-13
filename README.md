# Capstone

Magma explicit-object grounding and attention analysis on the YeahPick bundle.

## Status

The 500 explicit-object inputs passed asset SHA256 and swap-label validation.
All 500 scenes were rendered. The two-case Magma pilot passed checkpoint loading,
finite-logit checks, visual token-grid checks, and attention-hook prediction
agreement. The full 500-case attention + occlusion experiment is running on VESSL;
completion is recorded in `outputs/pipeline_status.json` only after all results
and summaries exist.

Repository: https://github.com/hyerin26/Capstone (private).

## Execution

The renderer requires the existing YeahPick checkout (tested source commit
`26cc321d3e4cbbc68225f7304f144f46bbbf5519`) and its working ManiSkill environment.
Magma runs in a separate venv with inherited PyTorch 2.2.0+cu121 and Transformers
4.47.0. Added dependencies: open_clip_torch 2.26.1 and ftfy 6.3.1.
The new VESSL container needed `libgl1` and `libvulkan1` reinstalled.
The loader skips an unused CLIP weight download, restores checkpoint LayerScale
parameters renamed by Transformers, and rejects unresolved inference weights.
Three unused CLIP pooling-head parameters are guarded against accidental execution.
Magma source/weight target revision: `4159b1dedade8f88b5bbaaae7810299dee5228f5`.

```sh
source scripts/server_env.sh
python scripts/render_scenes.py --bundle /path/to/bundle --yeahpick /root/YeahPick
python scripts/run_attention.py --checkpoint /path/to/magma --limit 2 --validate-hook --occlusion --output outputs/pilot
# Only after inspecting and validating the pilot:
python scripts/run_attention.py --checkpoint /path/to/magma --occlusion
python scripts/summarize.py
```

`outputs/scenes/<id>/` contains RGB, actor-ID masks and metadata.
`outputs/attention/<id>/` contains results, per-head spatial attention and a
five-layer heatmap panel. The summary exports cases.csv, categories.csv, layers.csv
and REPORT.md. CSV results preserve invalid free responses separately
from forced candidate ranking. All-layer means are fixed in advance; layers
1/8/16/24/32 are visualization panels, not selected based on test success.
Masks cover the candidate tiles; they do not identify the object inside a photo.
Occlusion fills a whole tile with gray. The background control uses equal-area
scattered pixels and therefore differs in topology; interpret it as a diagnostic,
not a definitive causal test. Pair-cluster bootstrap intervals use 2,000 draws.

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
