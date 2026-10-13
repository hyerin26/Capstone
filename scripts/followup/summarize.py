"""Verify all generations and aggregate paired, repeated-measures follow-ups."""
import argparse,collections,csv,gzip,json,math,statistics
from pathlib import Path
import numpy as np

CONDITIONS=['original_lr','equal_lr','original_marked_lr','equal_marked_lr','original_ab','equal_ab']
METHODS=['uppercase','titlecase','lowercase','case_marginal']

def mean(xs):return sum(xs)/len(xs) if xs else None

def pair(r):return (r['frozen_asset'],r['question_id'])

def grouped(rows,key):
    d=collections.defaultdict(list)
    for r in rows:d[key(r)].append(r)
    return d

def interval(values):
    values=np.asarray(values,float);rng=np.random.default_rng(20261014)
    boot=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(1)
    return [float(x) for x in np.quantile(boot,[.025,.975])]

def dump_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def condition_stats(rows):
    units=grouped(rows,pair);swaps=grouped(rows,lambda r:(*pair(r),r['mapping']))
    assert all(len(v)==2 and {r['swap'] for r in v}=={'0','1'} for v in swaps.values())
    ci=interval([mean([r['correct'] for r in v]) for v in units.values()])
    return dict(n_predictions=len(rows),n_scenes=len({r['scene_id'] for r in rows}),n_pairs=len(units),
        correct=sum(r['correct'] for r in rows),accuracy=mean([r['correct'] for r in rows]),ci_low=ci[0],ci_high=ci[1],
        macro_accuracy=mean([mean([r['correct'] for r in v]) for v in grouped(rows,lambda r:r['category']).values()]),
        left_target_accuracy=mean([r['correct'] for r in rows if r['answer_side']=='left']),
        right_target_accuracy=mean([r['correct'] for r in rows if r['answer_side']=='right']),
        right_selection_rate=mean([r['predicted_side']=='right' for r in rows]),invalid=sum(not r['valid'] for r in rows),
        both_swaps_correct_rate=mean([all(r['correct'] for r in v) for v in swaps.values()]),
        swap_consistency_rate=mean([all(r['valid'] for r in v) and v[0]['predicted_side']!=v[1]['predicted_side'] for v in swaps.values()]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='outputs/followup');p.add_argument('--audit',default='outputs/audit/explicit_object_cases.jsonl');a=p.parse_args()
    root=Path(a.results);report=root/'report';report.mkdir(exist_ok=True)
    expected={r['case_id']:r for r in map(json.loads,Path(a.audit).read_text().splitlines())}
    cases=[json.loads(f.read_text()) for f in sorted(root.glob('*/result.json'))]
    assert len(cases)==500 and {c['source']['case_id'] for c in cases}==set(expected)
    assert len({c['method_signature'] for c in cases})==1
    rows=[];shapes=set();sizes=set();uids=set()
    for c in cases:
        src=c['source'];assert len(c['records'])==13
        for k,v in expected[src['case_id']].items():assert src[k]==v,(src['case_id'],k)
        for r in c['records']:
            uid=(src['case_id'],r['condition'],r['mapping'],r['intervention']);assert uid not in uids;uids.add(uid)
            assert r['correct']==(r['predicted_side']==src['answer_side'])
            assert r['valid']==(r['predicted_side'] in ('left','right'))
            assert 0<=r['restricted_label_mass']<=1.00001
            assert all(math.isfinite(v) for values in r['scores'].values() for v in values.values())
            assert src['instruction'] in r['model_prompt']
            shapes.add(tuple(r['pixel_values_shape']));sizes.add(json.dumps(r['image_sizes']))
            row={k:src[k] for k in ('case_id','category','question_id','frozen_asset','swap','answer_side','instruction','scene_id')}
            row.update(r);rows.append(row)
        for layout in ('original','equal'):
            for mapping in ('AB','BA'):
                controls=[r for r in c['records'] if r['condition'] in (layout+'_ab',layout+'_marked_lr') and r['mapping']==mapping]
                assert len(controls)==2 and controls[0]['image_rgb_sha256']==controls[1]['image_rgb_sha256']
    assert len(rows)==6500 and len(shapes)==1 and len(sizes)==1
    clean=[r for r in rows if r['intervention']=='none'];assert len(clean)==5000
    bycond=grouped(clean,lambda r:r['condition']);assert set(bycond)==set(CONDITIONS)
    conditions=[dict(condition=c,**condition_stats(bycond[c])) for c in CONDITIONS]
    categories=[]
    for cond in CONDITIONS:
        for cat,rs in sorted(grouped(bycond[cond],lambda r:r['category']).items()):categories.append(dict(condition=cond,category=cat,**condition_stats(rs)))
    contrasts=[]
    comparisons=[('equal_lr','original_lr','presentation_unmarked'),('original_marked_lr','original_lr','marks_original'),('equal_marked_lr','equal_lr','marks_equal'),('original_ab','original_marked_lr','response_original_identical_pixels'),('equal_ab','equal_marked_lr','response_equal_identical_pixels'),('equal_ab','original_ab','presentation_ab'),('equal_ab','original_lr','combined')]
    for treatment,control,name in comparisons:
        t={k:mean([r['correct'] for r in rs]) for k,rs in grouped(bycond[treatment],pair).items()}
        b={k:mean([r['correct'] for r in rs]) for k,rs in grouped(bycond[control],pair).items()};assert set(t)==set(b)
        diff=[t[k]-b[k] for k in sorted(t)];lo,hi=interval(diff)
        contrasts.append(dict(comparison=name,treatment=treatment,control=control,accuracy_difference=mean(diff),ci_low=lo,ci_high=hi,n_pairs=len(diff)))
    mappings=[]
    for cond in CONDITIONS:
        if cond.endswith('_lr') and 'marked' not in cond:continue
        units=grouped(bycond[cond],lambda r:r['scene_id']);assert all(len(v)==2 for v in units.values())
        maps=dict(condition=cond,n_scenes=len(units),physical_side_consistency=mean([all(r['valid'] for r in v) and v[0]['predicted_side']==v[1]['predicted_side'] for v in units.values()]),both_mappings_correct=mean([all(r['correct'] for r in v) for v in units.values()]))
        if cond.endswith('_ab'):
            for label in ('A','B'):
                maps['always_'+label+'_rate']=mean([all(r['valid'] and r['mapping'][('left','right').index(r['predicted_side'])]==label for r in v) for v in units.values()])
        mappings.append(maps)
    base={r['scene_id']:r for r in bycond['original_lr']};study1=[]
    for intervention in ('none','target','distractor','background'):
        chosen=[]
        for r in rows:
            if r['condition']!='original_lr':continue
            actual=r['intervention'];expected_side=r['answer_side'] if intervention=='target' else ('right' if r['answer_side']=='left' else 'left')
            if (intervention in ('none','background') and actual==intervention) or (intervention in ('target','distractor') and actual==expected_side):chosen.append(r)
        assert len(chosen)==500
        initially_correct=[r for r in chosen if base[r['scene_id']]['correct']]
        initially_wrong=[r for r in chosen if not base[r['scene_id']]['correct']]
        for method in METHODS:
            drops=[]
            for r in chosen:
                target=r['answer_side'];wrong='right' if target=='left' else 'left';b=base[r['scene_id']]['scores'][method];s=r['scores'][method]
                drops.append((b[target]-b[wrong])-(s[target]-s[wrong]))
            study1.append(dict(intervention=intervention,scoring=method,n=500,free_accuracy=mean([r['correct'] for r in chosen]),
                score_accuracy=mean([r['score_predictions'][method]==r['answer_side'] for r in chosen]),
                agreement_with_generation=mean([r['score_predictions'][method]==r['predicted_side'] for r in chosen]),
                mean_target_margin_drop=mean(drops),right_selection_rate=mean([r['predicted_side']=='right' for r in chosen]),invalid=sum(not r['valid'] for r in chosen),
                changed_physical_choice_rate=mean([r['valid'] and base[r['scene_id']]['valid'] and r['predicted_side']!=base[r['scene_id']]['predicted_side'] for r in chosen]),
                correct_to_wrong_rate=mean([not r['correct'] for r in initially_correct]),wrong_to_correct_rate=mean([r['correct'] for r in initially_wrong]),
                mean_restricted_label_mass=mean([r['restricted_label_mass'] for r in chosen])))
    flat=[]
    for r in rows:
        row={k:r[k] for k in ('case_id','scene_id','category','question_id','swap','answer_side','instruction','condition','mapping','intervention','raw_response','predicted_side','valid','correct','restricted_label_mass','image_rgb_sha256')}
        for method,values in r['scores'].items():
            row[method+'_predicted_side']=r['score_predictions'][method]
            for side,value in values.items():row[method+'_'+side+'_logprob']=value
        flat.append(row)
    for name,values in [('generations',flat),('conditions',conditions),('categories',categories),('paired_contrasts',contrasts),('label_mappings',mappings),('scoring_occlusion',study1)]:dump_csv(report/(name+'.csv'),values)
    qa=dict(cases=500,generations=6500,method_signature=cases[0]['method_signature'],baseline_raw_agreement_count=sum(c['baseline_raw_agreement'] for c in cases),baseline_uppercase_max_difference=max(c['baseline_uppercase_max_difference'] for c in cases),native_pixel_shapes=[list(x) for x in shapes],native_image_sizes=[json.loads(x) for x in sizes],min_quad_mask_iou=min(c['source']['geometry'][s]['mask_quad_iou'] for c in cases for s in ('left','right')),mean_original_right_to_left_tile_area=mean([c['source']['geometry']['right']['original_mask_pixels']/c['source']['geometry']['left']['original_mask_pixels'] for c in cases]),identical_marked_pixels_verified=True)
    (report/'verification.json').write_text(json.dumps(qa,indent=2))
    summary=dict(verification=qa,conditions=conditions,categories=categories,paired_contrasts=contrasts,label_mappings=mappings,scoring_occlusion=study1)
    (report/'summary.json').write_text(json.dumps(summary,indent=2))
    with gzip.open(report/'raw_records.jsonl.gz','wt',encoding='utf-8') as f:
        for c in cases:f.write(json.dumps(c)+'\n')
    print(json.dumps(dict(verification=qa,conditions=conditions,paired_contrasts=contrasts,scoring_occlusion=study1),indent=2))

if __name__=='__main__':main()
