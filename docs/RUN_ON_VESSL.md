# Running on the configured VESSL workspace

Open `/root/Capstone` through VS Code Remote SSH. The experiment has separate
rendering and Magma environments so package changes do not disturb YeahPick.

## Existing paths

- Code: `/root/Capstone`
- Input bundle: `/root/Capstone/bundle`
- Simulator: `/root/YeahPick`
- Renderer Python: `/root/envs/openvla_rl4vla/bin/python`
- Magma Python: `/root/Capstone/.venv/bin/python`
- Weights: `/root/Capstone/weights/magma`
- Results: `/root/Capstone/outputs/attention`

## Inspect the current run

```sh
cd /root/Capstone
cat outputs/pipeline_status.json
tail -n 5 logs/pipeline-v4.log
find outputs/attention -name result.json | wc -l
nvidia-smi
```

`main_running` means inference is still in progress. `complete` requires all 500
results and successful CSV summarization. A process ID alone is not completion.
Logs from earlier failed pilot attempts remain under `logs/` for diagnosis.

## Reproduce in a fresh output directory

The data and model are intentionally outside Git. Obtain the original bundle and
keep `assets/` and `prompts/` together. Do not overwrite another run's results.

```sh
cd /root/Capstone
source scripts/server_env.sh
.venv/bin/python scripts/audit_bundle.py --bundle bundle
/root/envs/openvla_rl4vla/bin/python scripts/render_scenes.py \
  --bundle /root/Capstone/bundle --yeahpick /root/YeahPick
USE_TF=0 .venv/bin/python scripts/run_attention.py \
  --checkpoint weights/magma --occlusion --validate-hook --limit 2 \
  --output outputs/new-pilot
# Inspect the pilot before the full run.
USE_TF=0 .venv/bin/python scripts/run_attention.py \
  --checkpoint weights/magma --occlusion --output outputs/new-run
.venv/bin/python scripts/recover_model_prompts.py --results outputs/new-run
.venv/bin/python scripts/verify_results.py --results outputs/new-run
.venv/bin/python scripts/summarize.py --results outputs/new-run
.venv/bin/python scripts/plot_results.py --results outputs/new-run
```

Resume with exactly the same code, parameters, checkpoint and output directory.
A method-signature mismatch stops the run instead of mixing protocols. These are
greedy static-image predictions; re-running generation seeds does not create
independent data. The supplied 500 cases include both layouts of 250 pairs.

## What to read

- `REPORT.md`: accuracy table and interpretation limits.
- `cases.csv`: one row per scene, responses, attention and occlusion metrics.
- `categories.csv`: category aggregates and pair-bootstrap confidence intervals.
- `layers.csv`: every layer's regional attention summaries by category.
- `figures/`: accuracy/occlusion charts, layer matrix and fixed-selection examples.
- `<scene-id>/result.json`: source prompt, exact input IDs, decoded model input, answer and method signature.
- `<scene-id>/attention.npz`: layer × head × height × width spatial attention.

This evaluates selection in a text response, not robot action success. Masks mark
whole tiles. Neither attention nor gray-tile occlusion alone proves object
recognition or causal reasoning.
