# Gripper attention intervention in Act2Answer

Hypotheses: gripper attention can interfere with target choice, contribute to control,
or both. A high heatmap value alone does not distinguish these hypotheses.

## Frozen design

- Source: user-supplied `YeahPick_10categories_prompts_glb`, original Explicit object instructions.
- Pair-level split with deterministic shuffle: 20 validation pairs, 30 test pairs,
  two separate state-category pilot pairs. Both layouts per pair. Test has 3 pairs/category.
- Same source GLB hashes, environment seed 0, canonical model chat/action prompt,
  RGB resized to 256 square as in native Magma wrapper, bridge_orig action decoding.
- Greedy generation and eager attention used in all conditions. This is a controlled
  intervention study, not a numerical reproduction of historical sampled runs.
- Native wrapper pre-grasps cube for 10 control steps and forces closure for first
  15 policy steps. Episodes continue for 80 policy steps as in the benchmark.
- Initial simulator and controller states are saved/restored across conditions.
- Full 256 action-token range is validated before native 255-center clipping decoder.
  Invalid sequences are failed rollouts, never silently clipped text into actions.

Intervention: subtract lambda * fractional mask occupancy from attention logits
before softmax/value multiplication. All 32 layers/heads; visual keys only; final
prompt and subsequently generated query positions only. Earlier prompt and image
query representations are unchanged. Stored attention is the modified real forward
pass, not a changed plotting array. Row separators and text keys are excluded.

Conditions: baseline; gripper lambda .5/1/2; translated background mask lambda
.5/1/2; cube lambda1; gripper lambda1 in transport only or placement only.
Gripper links: gripper_link, ee_gripper_link, left_finger_link, right_finger_link.
Masks follow simulator segmentation each frame and the image resize/token grid.
Background is a translated visible gripper shape excluding robot, cube and tiles;
clipping area ratio is logged. No baseline-attention-mass matching is claimed.
Transport: currently grasped and TCP horizontal distance >= .12 m from both tile
centers. Placement: TCP distance < .12 m from either tile, regardless of correctness.
Other frames: ungrasped. These stages approximate a control phase, not a latent
semantic decision time; transport includes early pre-grasped motion.

## Metrics and limits

Primary: released cube resting on correct tile (world-oriented board bounds,
center XY tolerance .015 m; Z within .02 m of support height), not grasped, speed
< .05 m/s for 5 consecutive control steps (1 second). Also record whether this
condition still holds at final step. Benchmark success is reported separately:
its original vertical criterion has no upper bound and can count a held cube.

First-choice proxy: first unambiguous TCP entry over a tile (bounds + .015 m,
TCP >= board top + .02 m). This is observable motion intent, not an explicit
model decision. Never entering either tile is missing, not a correct choice.
Log first grasp-to-release event and correctness, executed actions, cube/TCP
trajectories, invalid outputs, region attention fractions and dynamic mask area.
No free-standing initial-pick skill is evaluated because cube starts pre-grasped.

Attention records average the seven action-token prediction query maps, retaining
layer/head axes. First generated-token maps are retained separately. Generation
may include preceding text; original generated IDs and exact prompts are saved.
Interventions also affect preceding generated queries, not only the final 7.

Validation uses matched fixed observations, without stepping between conditions.
Test uses closed-loop rollouts from the same initial state. Subsequent observations
can diverge, so rollout attention differences include changed trajectories.
Compare paired outcome differences with 2,000 pair-cluster bootstrap draws.
Incomplete interim pairs are omitted from bootstrap and completion counts shown.
Invalid policy output counts as task failure, not excluded as an inconvenient case.
A small test set cannot establish category-specific generalization or robot transfer.

## Execution on VESSL

From `/root/Capstone`:

```sh
source scripts/server_env.sh
USE_TF=0 .venv/bin/python gripper_experiment/test_intervention.py
# Start policy.py once in a separate process, then:
USE_TF=0 .venv/bin/python -u gripper_experiment/launch.py
```

`plan.json`: fixed sample and asset hashes. `run_config.json`: source hashes and
simulator commit. `status.json`: running/failed/complete and current case.
`technical_gate.json`: technical pilot gate. `logs/`: per-case logs.
`outputs_v1/<mode>/<case-hash>/<condition>/`: episode.json, steps, attention,
first RGB/masks and rollout.mp4. `reports/`: CSVs, paired effects and Korean report.
Earlier `outputs/` are development diagnostics and are not included in final analysis.
Large raw outputs and queues are excluded from Git; source, plan and reports are tracked.

Any environment failure stops the launcher with a recorded traceback. An invalid
model action is recorded as a failed episode. No outcomes are used to choose heads,
layers, sample membership or phase-ablation strength.
