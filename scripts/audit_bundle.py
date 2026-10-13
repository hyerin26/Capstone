"""Validate explicit-object inputs without changing the source bundle."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def audit(root, output):
    root = Path(root).resolve()
    with (root / 'prompts/vla_prompts_seed0.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['condition'] == 'Explicit object']
    if not rows:
        raise ValueError('No Explicit object rows found')
    seen, hashes, groups = set(), {}, collections.defaultdict(list)
    for r in rows:
        if r['case_id'] in seen:
            raise ValueError(f"Duplicate case: {r['case_id']}")
        seen.add(r['case_id'])
        if r['answer_side'] not in ('left', 'right') or r['swap'] not in ('0', '1'):
            raise ValueError(f"Invalid label: {r['case_id']}")
        if r['answer_AB'] != {'left': 'A', 'right': 'B'}[r['answer_side']]:
            raise ValueError(f"Inconsistent answer_AB: {r['case_id']}")
        for side in ('left', 'right'):
            path = (root / r[f'{side}_glb']).resolve()
            if root not in path.parents:
                raise ValueError(f'Path outside bundle: {path}')
            if path not in hashes:
                hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashes[path] != r[f'{side}_glb_sha256']:
                raise ValueError(f'Hash mismatch: {path}')
        groups[(r['category'], r['frozen_asset'], r['question_id'])].append(r)
    for key, pair in groups.items():
        if len(pair) != 2 or {r['swap'] for r in pair} != {'0', '1'}:
            raise ValueError(f'Missing/duplicate swap: {key}')
        a, b = sorted(pair, key=lambda r: r['swap'])
        if not (a['left_glb'] == b['right_glb'] and a['right_glb'] == b['left_glb']
                and a['answer_side'] != b['answer_side']):
            raise ValueError(f'Inconsistent swap: {key}')
    summary = dict(cases=len(rows), unique_pairs=len(groups), checked_glbs=len(hashes),
                   categories=dict(sorted(collections.Counter(r['category'] for r in rows).items())))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'explicit_object_cases.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    (output / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True)
    p.add_argument('--output', default='outputs/audit')
    a = p.parse_args()
    audit(a.bundle, a.output)
