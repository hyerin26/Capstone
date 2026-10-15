# Desk image command attention experiment

Six independent prompts on the same supplied RGB image:
`Pick up the headphones.`, `Pick up the speaker.`, `Pick up the clock.`,
`Pick up the eyeglasses.`, `Pick up the camera.`, `Pick up the mobile phone.`

No LEFT/RIGHT instruction, extra localization request, robot rollout or action decoder.
The model generates text with greedy decoding (maximum 64 new tokens); a response
reaching this limit can be truncated. Responses are not scored as action success.

Run from /root/Capstone:

```
USE_TF=0 TOKENIZERS_PARALLELISM=false .venv/bin/python desk_experiment/run.py
.venv/bin/python desk_experiment/plot.py
```

Reuses scripts/run_attention.py checkpoint compatibility and spatial token extraction.
Attention is from the final prompt position predicting the first response token,
32 layers x 32 heads x 32 x 32 visual cells. All six capture passes reproduced
the uncaptured first-token prediction and LEFT/RIGHT diagnostic scores within 0.1;
those unrelated diagnostic scores are removed from saved results.

results/<object>/attention.npz: raw head_maps (float16).
result.json: exact prompt, input IDs, response, image SHA256, revision and grid.
heatmap.png: mean over all heads/layers, image-conditional normalization per head
before averaging. The six means use one shared linear color scale.
layers.png: fixed layers 1, 8, 16, 24, 32, head mean, shared scale within each panel.
overview.png: all six mean overlays. Bright yellow means higher attention.
The grid is bilinearly interpolated for display, so pixel localization is approximate.
Common edge/background peaks remain visible; no target masking or post-hoc head selection.
These visualizations do not establish causal recognition or physical grasp ability.

Observed responses: headphones mentioned; speaker confused with Marshall headphones;
clock, eyeglasses and camera receive location descriptions; phone command receives a
scene description cut off at the token limit. Inspect responses.json for exact text.
