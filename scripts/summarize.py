"""Export completed cases and category summaries; never invent missing results."""
import argparse
import collections
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',default='outputs/attention')
    a=p.parse_args(); root=Path(a.results)
    rows=[json.loads(f.read_text()) for f in sorted(root.glob('*/result.json'))]
    if not rows:raise SystemExit('No completed inference results')
    fields=['case_id','category','question_id','swap','instruction','left_tile','right_tile',
            'answer_side','raw_response','predicted_side','valid','correct','candidate_prediction','candidate_correct','scene_id']
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
        stats.append(dict(category=category,n=n,correct=sum(r['correct'] for r in rs),
            accuracy=sum(r['correct'] for r in rs)/n,invalid=sum(not r['valid'] for r in rs),
            candidate_accuracy=sum(r['candidate_correct'] for r in rs)/n,
            completed_swap_pairs=len(complete),swap_consistency=consistent/len(complete) if complete else None))
    with (root/'categories.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
    summary=dict(completed=len(rows),micro_accuracy=sum(r['correct'] for r in rows)/len(rows),
                 macro_accuracy=sum(r['accuracy'] for r in stats)/len(stats),categories=stats)
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
