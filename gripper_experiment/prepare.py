"""Freeze outcome-independent pair splits, source hashes and conditions."""
import argparse,csv,hashlib,json,random,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args()
root=Path(a.bundle);dest=Path(__file__).resolve().parent
with (root/'prompts/vla_prompts_seed0.csv').open(encoding='utf-8-sig') as f:
    rows=[r for r in csv.DictReader(f) if r['condition']=='Explicit object']
groups=collections.defaultdict(list)
for r in rows:groups[(r['category'],r['frozen_asset'],r['question_id'])].append(r)
bycat=collections.defaultdict(list)
for key,rs in sorted(groups.items()):
    assert len(rs)==2 and {r['swap'] for r in rs}=={'0','1'}
    bycat[key[0]].append(rs)
selected=[]
for cat,pairs in sorted(bycat.items()):
    random.Random('gripper-v1-'+cat).shuffle(pairs)
    for i,pair in enumerate(pairs[:5]):
        for r in pair:selected.append(dict(r,split='validation' if i<2 else 'test'))
    if cat=='state':
        for pair in pairs[5:7]:
            for r in pair:selected.append(dict(r,split='pilot'))
conditions=[dict(name='baseline',region='none',strength=0.,phase='all')]
for region in ['gripper','background']:
    for strength in [.5,1.,2.]:conditions.append(dict(name=f'{region}_{strength:g}',region=region,strength=strength,phase='all'))
conditions.append(dict(name='cube_1',region='cube',strength=1.,phase='all'))
for phase in ['transport','placement']:
    conditions.append(dict(name='gripper_1_'+phase,region='gripper',strength=1.,phase=phase))
hashes={}
for r in selected:
    for side in ['left','right']:
        rel=r[side+'_glb'];digest=hashlib.sha256((root/rel).read_bytes()).hexdigest()
        assert digest==r[side+'_glb_sha256'];hashes[rel]=digest
plan=dict(version='gripper-v1',selection_seed='gripper-v1-<category>',condition='explicit_object',cases=selected,conditions=conditions,
          source_prompt_sha256=hashlib.sha256((root/'prompts/vla_prompts_seed0.csv').read_bytes()).hexdigest(),asset_hashes=hashes,
          policy_steps=80,init_grasp_steps=10,forced_hold_policy_steps=15,max_new_tokens=128,decoding='greedy',
          intervention='all 32 layers/heads, only last-prompt and generated queries to visual keys; pre-softmax -lambda*occupancy',
          gripper_links=['gripper_link','ee_gripper_link','left_finger_link','right_finger_link'],
          primary_metric='released_stable_correct_5_steps',bootstrap_draws=2000)
(dest/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False))
print(collections.Counter(r['split'] for r in selected))
