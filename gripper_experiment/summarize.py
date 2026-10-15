"""Export paired results and pair-cluster bootstrap without pooling pilot/test."""
import json,csv,collections,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs_v1'
DEST=ROOT/'reports';DEST.mkdir(exist_ok=True)
rows=[];raw=[]
for p in sorted(OUT.glob('*/*/*/episode.json')):
 r=json.loads(p.read_text());raw.append(r);case=r['case'];tr=r['traces'];first=tr[0]['policy']
 fractions={key:float(np.mean([t['policy']['region_fractions'][key] for t in tr])) for key in first['region_fractions']}
 rows.append(dict(case_id=case['case_id'],category=case['category'],question_id=case['question_id'],swap=case['swap'],split=case['split'],mode=r['mode'],condition=r['condition']['name'],completed=r['completed'],steps=r['steps'],
                  initial_state_hash=r['initial_state_hash'],strict_success=r['released_stable_correct_5_steps'],final_strict_success=r['final_stable_correct_5_steps'],benchmark_success=r['benchmark_success'],
                  first_choice=None if r['first_entry'] is None else r['first_entry']['correct'],release_correct=None if r['first_release'] is None else r['first_release']['correct'],
                  first_action=first['action'],zero_hook_sequence_validated=first['zero_hook_sequence_validated'],
                  mean_gripper_fraction=fractions['gripper'],mean_cube_fraction=fractions['cube'],mean_background_fraction=fractions['background'],
                  min_background_area_ratio=min(t['mask_meta']['background_area_ratio'] for t in tr),
                  episode_file=str(p.relative_to(ROOT))))
if rows:
 with (DEST/'episodes.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lookup={(r['case_id'],r['mode'],r['condition']):r for r in rows}
paired=[]
for r in rows:
 b=lookup.get((r['case_id'],r['mode'],'baseline'))
 if b is None or r['condition']=='baseline':continue
 # Full physical reset state equality verified per case; hash includes grasp contact flag.
 same=r['initial_state_hash']==b['initial_state_hash']
 pair=dict(case_id=r['case_id'],category=r['category'],question_id=r['question_id'],swap=r['swap'],split=r['split'],mode=r['mode'],condition=r['condition'],initial_hash_equal=same,
           gripper_fraction_change=r['mean_gripper_fraction']-b['mean_gripper_fraction'],cube_fraction_change=r['mean_cube_fraction']-b['mean_cube_fraction'],
           strict_success_change=int(r['strict_success'])-int(b['strict_success']),benchmark_success_change=int(r['benchmark_success'])-int(b['benchmark_success']),
           both_valid=r['completed'] and b['completed'])
 if r['first_action'] is not None and b['first_action'] is not None:
  delta=np.asarray(r['first_action'])-np.asarray(b['first_action'])
  pair.update(translation_delta_norm=float(np.linalg.norm(delta[:3])),rotation_delta_norm=float(np.linalg.norm(delta[3:6])),gripper_action_changed=bool(delta[6]))
 paired.append(pair)
if paired:
 fields=list(dict.fromkeys(k for r in paired for k in r))
 with (DEST/'paired.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(paired)
summary=[]
for condition in sorted({r['condition'] for r in paired if r['split']=='test' and r['mode']=='rollout'}):
 rs=[r for r in paired if r['condition']==condition and r['split']=='test' and r['mode']=='rollout']
 units=collections.defaultdict(list)
 for r in rs:units[(r['category'],r['question_id'])].append(r['strict_success_change'])
 # Only report CIs for complete mirrored pairs; incomplete live results marked partial.
 complete=[v for v in units.values() if len(v)==2]
 if not complete:continue
 rng=np.random.default_rng(20261015);values=np.array([np.mean(v) for v in complete]);boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
 summary.append(dict(condition=condition,complete_pairs=len(values),strict_success_difference=float(values.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
(DEST/'summary.json').write_text(json.dumps(dict(episode_records=len(rows),paired_differences=summary),indent=2))
lines=['# 그리퍼 attention 개입 실험','',f'현재 기록: {len(rows)}개. 완료 여부는 status.json을 확인하세요. 중간 결과를 최종 성능으로 해석하지 마세요.','',
       '원본 Act2Answer의 success는 큐브가 타일 위쪽에 있는지만 검사하며, 놓기·높이 상한·안정성을 보장하지 않습니다. 그래서 released_stable_correct_5_steps를 주 지표로 추가했습니다.',
       '그리퍼/큐브는 이미 잡은 초기 상태에서 시작합니다. 처음 집기 능력을 평가하지 않습니다. 행동 토큰 전체가 유효하지 않으면 실패로 기록하며 임의의 텍스트 토큰을 행동으로 변환하지 않습니다.',
       '개입은 모든 레이어/head에서 마지막 프롬프트 및 생성 query의 이미지 key에만 적용합니다. 이미지 토큰 표현 자체와 이전 프롬프트 query는 변경하지 않습니다.',
       '검증/평가 분리는 원본 pair 단위입니다. 2,000회 bootstrap은 좌우 장면을 같은 pair로 묶습니다. 카테고리당 평가 pair 3개이므로 카테고리별 확정적 주장은 피해야 합니다.','',
       '| 조건 | 완료된 평가 pair | 성공률 차이 | 95% CI |','|---|---:|---:|---|']
for r in summary:lines.append(f"| {r['condition']} | {r['complete_pairs']} | {r['strict_success_difference']:+.1%} | [{r['ci_low']:+.1%}, {r['ci_high']:+.1%}] |")
lines += ['', '## 동일 관측에서의 행동 및 attention 변화', '', '| Split | 조건 | 사례 수 | 그리퍼 attention 비중 변화 | 이동 출력 차이 (m) |', '|---|---|---:|---:|---:|']
fixed_summary=[]
for split in ['pilot','validation']:
 for cond in sorted({r['condition'] for r in paired if r['mode']=='fixed' and r['split']==split}):
  rs=[r for r in paired if r['mode']=='fixed' and r['split']==split and r['condition']==cond]
  valid=[r for r in rs if r['both_valid'] and 'translation_delta_norm' in r]
  if not valid:continue
  entry=dict(split=split,condition=cond,n=len(valid),mean_gripper_fraction_change=float(np.mean([r['gripper_fraction_change'] for r in valid])),mean_translation_delta_norm=float(np.mean([r['translation_delta_norm'] for r in valid])))
  fixed_summary.append(entry)
  lines.append(f"| {split} | {cond} | {len(valid)} | {entry['mean_gripper_fraction_change']:+.4f} | {entry['mean_translation_delta_norm']:.5f} |")
(DEST/'fixed_summary.json').write_text(json.dumps(fixed_summary,indent=2))
(DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
# Fixed pilot example selected by case ID, irrespective of success.
pilots=sorted([r for r in raw if r['case']['split']=='pilot' and r['mode']=='fixed'],key=lambda r:r['case']['case_id'])
if pilots:
 cid=pilots[0]['case']['case_id'];examples=[r for r in pilots if r['case']['case_id']==cid]
 fig,axes=plt.subplots(1,len(examples),figsize=(5*len(examples),5),squeeze=False,layout='constrained')
 maps=[]
 for r in examples:
  dest=OUT/'fixed'/hashlib.sha256(cid.encode()).hexdigest()[:16]/r['condition']['name']
  arr=np.load(dest/'step_000.npz')['action_query_mean'].astype(float)
  maps.append(arr.mean((0,1)))
 vmax=max(m.max() for m in maps)
 for ax,r,heat in zip(axes.flat,examples,maps):
  dest=OUT/'fixed'/hashlib.sha256(cid.encode()).hexdigest()[:16]/r['condition']['name']
  from PIL import Image
  img=Image.open(dest/'initial.png');ax.imshow(img)
  im=ax.imshow(heat,cmap='inferno',alpha=.65,vmin=0,vmax=vmax,extent=(-.5,img.width-.5,img.height-.5,-.5),interpolation='bilinear')
  ax.set_title(r['condition']['name']);ax.axis('off')
 fig.colorbar(im,ax=list(axes.flat),shrink=.8,label='Raw attention, action-query mean; shared scale')
 fig.suptitle(cid);fig.savefig(DEST/'pilot_attention.png',dpi=150);plt.close(fig)
print(json.dumps(dict(episodes=len(rows),paired=len(paired))))
