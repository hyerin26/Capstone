"""Decode saved model input IDs without rerunning or modifying predictions.

The input bundle itself has a `prompt` column. Preserve that source column and
store the actual decoded model input under a distinct, unambiguous name.
"""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='weights/magma')
    p.add_argument('--results',default='outputs/attention')
    a=p.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(a.checkpoint)
    count=0
    for f in sorted(Path(a.results).glob('*/result.json')):
        r=json.loads(f.read_text())
        decoded=tokenizer.decode(r['input_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        assert tokenizer.encode(decoded,add_special_tokens=False)==r['input_ids'],r['case_id']
        assert r['instruction'] in decoded,r['case_id']
        assert "Answer with only LEFT or RIGHT" in decoded,r['case_id']
        r['decoded_model_input']=decoded
        r['decoded_model_input_source']='lossless decoding of recorded pre-expansion input_ids; source CSV prompt remains separate'
        tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2));tmp.replace(f);count+=1
    print(f'Decoded and verified model inputs for {count} results; predictions unchanged.')

if __name__=='__main__':main()
