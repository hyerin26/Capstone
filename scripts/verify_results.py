"""Check complete result coverage and saved spatial-attention integrity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',default='outputs/attention')
    p.add_argument('--scenes',default='outputs/scenes')
    p.add_argument('--cases',default='outputs/audit/explicit_object_cases.jsonl')
    a=p.parse_args();root=Path(a.results)
    expected={r['case_id']:r for r in map(json.loads,Path(a.cases).read_text().splitlines())}
    seen=set();signatures=set();grids=set()
    for f in sorted(root.glob('*/result.json')):
        r=json.loads(f.read_text());case=r['case_id']
        assert case in expected and case not in seen, case
        seen.add(case);signatures.add(r['method_signature']);grids.add(tuple(r['grid']))
        for k,v in expected[case].items():
            assert r[k]==v,(case,k)
        assert r['instruction'] in r['decoded_model_input'],case
        assert r['valid']==(r['predicted_side'] in ('left','right')),case
        assert r['correct']==(r['predicted_side']==r['answer_side']),case
        assert r['candidate_prediction']==max(r['scores'],key=r['scores'].get).lower(),case
        assert set(r['occlusion'])=={'left','right','background'},case
        rgb=Path(a.scenes)/r['scene_id']/'rgb.png'
        assert hashlib.sha256(rgb.read_bytes()).hexdigest()==r['rgb_sha256'],case
        with np.load(f.parent/'attention.npz') as data:
            maps=data['head_maps'].astype('float32')
        assert maps.shape==(32,32,*r['grid']),(case,maps.shape)
        assert np.isfinite(maps).all() and (maps>=0).all(),case
        assert np.allclose(maps.sum((-1,-2)).mean(1),r['visual_mass_per_layer'],atol=1e-4,rtol=1e-3),case
    assert seen==set(expected),f'Missing {len(set(expected)-seen)} cases'
    assert len(signatures)==1,'Mixed method signatures'
    result=dict(verified=True,cases=len(seen),method_signature=next(iter(signatures)),spatial_grids=sorted(grids),
                checks=['complete unique coverage','original metadata preserved','answer consistency','all occlusion variants',
                        'input RGB SHA256','finite nonnegative 32-layer 32-head attention','saved attention mass agreement'])
    (root/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
