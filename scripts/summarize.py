"""Export completed cases and category summaries; never invent missing results."""
import argparse
import collections
import csv
import json
import random
import statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',default='outputs/attention')
    a=p.parse_args(); root=Path(a.results)
    rows=[json.loads(f.read_text()) for f in sorted(root.glob('*/result.json'))]
    if not rows:raise SystemExit('No completed inference results')
    for r in rows:
        target=r['answer_side'];wrong='right' if target=='left' else 'left'
        r['target_attention_fraction']=statistics.mean(r[target+'_fraction_by_layer'])
        r['target_attention_density']=statistics.mean(r[target+'_density_by_layer'])
        r['distractor_attention_density']=statistics.mean(r[wrong+'_density_by_layer'])
        r['target_peak_hit_fraction_layers']=statistics.mean(r[target+'_peak_hit_by_layer'])
        for side in (target,wrong,'background'):
            if side in r.get('occlusion',{}):
                scores=r['occlusion'][side]['scores']; base=r['scores']
                r['occlusion_'+side+'_margin_drop']=(base[target.upper()]-base[wrong.upper()])-(scores[target.upper()]-scores[wrong.upper()])
        if r.get('occlusion'):
            r['occlusion_target_margin_drop']=r['occlusion_'+target+'_margin_drop']
            r['occlusion_distractor_margin_drop']=r['occlusion_'+wrong+'_margin_drop']
    fields=['case_id','category','question_id','swap','instruction','left_tile','right_tile',
            'answer_side','raw_response','predicted_side','valid','correct','candidate_prediction','candidate_correct','scene_id','target_attention_fraction','target_attention_density','distractor_attention_density','target_peak_hit_fraction_layers','occlusion_left_margin_drop','occlusion_right_margin_drop','occlusion_background_margin_drop','occlusion_target_margin_drop','occlusion_distractor_margin_drop']
    with (root/'cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    grouped=collections.defaultdict(list)
    for r in rows:grouped[r['category']].append(r)
    stats=[]
    for category,rs in sorted(grouped.items()):
        n=len(rs);pairs=collections.defaultdict(list)
        for r in rs:pairs[(r['frozen_asset'],r['question_id'])].append(r)
        complete=[v for v in pairs.values() if len(v)==2 and {r['swap'] for r in v}=={'0','1'}]
        consistent=sum(all(r['valid'] for r in v) and v[0]['predicted_side']!=v[1]['predicted_side'] for v in complete)
        rng=random.Random(20261014)
        units=[v for v in pairs.values()]
        boot=[]
        for _ in range(2000):
            sample=[r for _ in units for r in rng.choice(units)]
            boot.append(sum(r['correct'] for r in sample)/len(sample))
        boot.sort()
        occlusion={key:statistics.mean(r[key] for r in rs if key in r) for key in ('occlusion_target_margin_drop','occlusion_distractor_margin_drop','occlusion_background_margin_drop') if any(key in r for r in rs)}
        stats.append(dict(**occlusion,category=category,accuracy_ci_low=boot[49],accuracy_ci_high=boot[1949],
            target_attention_fraction=statistics.mean(r['target_attention_fraction'] for r in rs),
            target_minus_distractor_density=statistics.mean(r['target_attention_density']-r['distractor_attention_density'] for r in rs),n=n,correct=sum(r['correct'] for r in rs),
            accuracy=sum(r['correct'] for r in rs)/n,invalid=sum(not r['valid'] for r in rs),
            candidate_accuracy=sum(r['candidate_correct'] for r in rs)/n,
            predicted_left=sum(r['predicted_side']=='left' for r in rs),
            predicted_right=sum(r['predicted_side']=='right' for r in rs),
            both_layouts_correct=sum(all(r['correct'] for r in v) for v in complete),
            both_layouts_correct_rate=sum(all(r['correct'] for r in v) for v in complete)/len(complete) if complete else None,
            completed_swap_pairs=len(complete),swap_consistency=consistent/len(complete) if complete else None))
    with (root/'categories.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
    layer_rows=[]
    for category,rs in sorted(grouped.items()):
        for layer in range(len(rs[0]['visual_mass_per_layer'])):
            layer_rows.append(dict(category=category,layer=layer+1,n=len(rs),
                visual_mass=statistics.mean(r['visual_mass_per_layer'][layer] for r in rs),
                target_fraction=statistics.mean(r[r['answer_side']+'_fraction_by_layer'][layer] for r in rs),
                target_density=statistics.mean(r[r['answer_side']+'_density_by_layer'][layer] for r in rs),
                distractor_density=statistics.mean(r[('right' if r['answer_side']=='left' else 'left')+'_density_by_layer'][layer] for r in rs),
                target_peak_hit=statistics.mean(r[r['answer_side']+'_peak_hit_by_layer'][layer] for r in rs)))
    with (root/'layers.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(layer_rows[0]));w.writeheader();w.writerows(layer_rows)
    summary=dict(completed=len(rows),invalid=sum(not r['valid'] for r in rows),
                 predicted_left=sum(r['predicted_side']=='left' for r in rows),
                 predicted_right=sum(r['predicted_side']=='right' for r in rows),
                 candidate_accuracy=sum(r['candidate_correct'] for r in rows)/len(rows),
                 micro_accuracy=sum(r['correct'] for r in rows)/len(rows),
                 macro_accuracy=sum(r['accuracy'] for r in stats)/len(stats),categories=stats)
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    report=['# Magma explicit-object grounding results', '',
        f"Completed: {len(rows)} cases. Micro accuracy: {summary['micro_accuracy']:.1%}; macro accuracy: {summary['macro_accuracy']:.1%}.",
        f"Candidate-ranking accuracy: {summary['candidate_accuracy']:.1%}. Invalid responses: {summary['invalid']}. Valid choices: LEFT {summary['predicted_left']}, RIGHT {summary['predicted_right']}.", '',
        '| Category | N | Accuracy | 95% pair-bootstrap CI | Candidate accuracy | Invalid | Swap consistency |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in stats:
        swap=f"{r['swap_consistency']:.1%}" if r['swap_consistency'] is not None else 'NA'
        report.append(f"| {r['category']} | {r['n']} | {r['accuracy']:.1%} | {r['accuracy_ci_low']:.1%}–{r['accuracy_ci_high']:.1%} | {r['candidate_accuracy']:.1%} | {r['invalid']} | {swap} |")
    report += ['', '## Interpretation', '',
        'This is a static RGB + text LEFT/RIGHT task, not robot action success. Invalid free responses count as incorrect; candidate accuracy ranks the two label-token scores separately.',
        'Swap consistency measures changing side under a mirrored pair; a consistently wrong target can also satisfy it. Attention is from the last prompt position, averaged over heads; every layer is exported in layers.csv.',
        'Masks cover the whole tile. Attention and gray-tile occlusion do not establish recognition of the object inside the tile. Positive occlusion margin drop means the intervention reduced the target-versus-distractor log-probability margin. The scattered background control differs in topology.',
        'Counting and Symmetry explicit-object prompts use color cues. Category sample sizes differ. Confidence intervals resample original pairs (2,000 draws).', '']
    (root/'REPORT.md').write_text('\n'.join(report))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
