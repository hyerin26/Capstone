# Reproducing the follow-up studies

Run from `/root/Capstone` using `.venv/bin/python` in the existing VESSL environment documented in [RUN_ON_VESSL.md](../RUN_ON_VESSL.md). Requires the pinned Magma checkpoint, original `outputs/scenes`, `outputs/attention`, and `outputs/audit/explicit_object_cases.jsonl`. These large/private inputs are not bundled in Git. See [PROTOCOL.md](PROTOCOL.md).

```sh
.venv/bin/python scripts/followup/prepare_inputs.py
```

Inspect generated `outputs/followup_inputs/preview_*.png`. Build the category-spanning pilot subset with symlinks (the manifest is intentionally separate):

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
src=Path('outputs/followup_inputs').resolve()
dst=Path('outputs/followup_pilot_inputs');dst.mkdir(exist_ok=True)
selected={}
for p in sorted(src.glob('*/metadata.json')):
    r=json.loads(p.read_text())
    selected.setdefault(r['category'],p.parent)
assert len(selected)==10
for p in selected.values():
    link=dst/p.name
    if not link.exists():link.symlink_to(p,target_is_directory=True)
m=json.loads((src/'manifest.json').read_text());m['cases']=10
(dst/'manifest.json').write_text(json.dumps(m,indent=2))
PY
.venv/bin/python scripts/followup/run.py --inputs outputs/followup_pilot_inputs --output outputs/followup_pilot --pilot
.venv/bin/python scripts/followup/launch.py
```

The launcher validates the pilot, runs two workers and aggregates results. A result file is written atomically per scene. Existing scene records with a matching method signature can be resumed. Do not change inference/preparation code during a run. The pilot has 130 generations; the main study has 6,500 and excludes the pilot.

To re-aggregate existing main-study results without loading the model:

```sh
.venv/bin/python scripts/followup/summarize.py
.venv/bin/python scripts/followup/plot.py
```

Outputs go to `outputs/followup/report/`. The committed snapshot is [reports/followup-2026-10-14](../../reports/followup-2026-10-14/). The saved compressed raw records can also reconstruct the per-scene JSON results for auditing; the summarizer additionally requires the original audit manifest.

The inference source committed before full execution is `52aabb9`. Summarization and plotting were added after inference, following the frozen protocol. `provenance/SHA256SUMS.json` records script hashes at report export. Runtime loading reports contain initial missing-key diagnostics: LayerScale parameters are subsequently restored from the checkpoint, and unused vision pooling heads are guarded, as implemented in the loader. Initial missing keys alone must not be read as uncorrected missing parameters.
