"""Run the validated pilot and full analysis after all render/weight files exist.

This launcher does not treat process launch as experimental completion. It writes
status.json at each stage and stops on any validation/inference failure.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='weights/magma')
    p.add_argument('--expected-scenes',type=int,default=500)
    p.add_argument('--timeout-hours',type=float,default=12)
    a=p.parse_args();start=time.time()
    def status(stage,**kw):
        value=dict(stage=stage,updated_at_unix=time.time(),**kw)
        Path('outputs').mkdir(exist_ok=True)
        tmp=Path('outputs/pipeline_status.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace('outputs/pipeline_status.json')
        print(json.dumps(value),flush=True)
    try:
        while True:
            scenes=len(list(Path('outputs/scenes').glob('*/metadata.json')))
            checkpoint=Path(a.checkpoint)
            index=checkpoint/'model.safetensors.index.json'
            shards=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else set()
            ready=bool(shards) and all((checkpoint/s).is_file() for s in shards)
            status('waiting_for_inputs',rendered=scenes,expected=a.expected_scenes,weights_ready=ready)
            if ready and scenes>=2:break
            if scenes>a.expected_scenes:raise RuntimeError('Unexpected extra scenes')
            if time.time()-start>a.timeout_hours*3600:raise TimeoutError('Input preparation timed out')
            time.sleep(30)
        command=[sys.executable,'scripts/run_attention.py','--checkpoint',a.checkpoint,'--occlusion']
        status('pilot_running')
        subprocess.run(command+['--limit','2','--validate-hook','--output','outputs/pilot'],check=True)
        while len(list(Path('outputs/scenes').glob('*/metadata.json')))<a.expected_scenes:
            status('waiting_for_scenes',rendered=len(list(Path('outputs/scenes').glob('*/metadata.json'))))
            if time.time()-start>a.timeout_hours*3600:raise TimeoutError('Rendering timed out')
            time.sleep(30)
        status('main_running')
        subprocess.run(command,check=True)
        results=len(list(Path('outputs/attention').glob('*/result.json')))
        if results!=a.expected_scenes:raise RuntimeError(f'Only {results} results')
        subprocess.run([sys.executable,'scripts/summarize.py'],check=True)
        status('complete',completed=results)
    except Exception as e:
        status('failed',error=str(e))
        raise

if __name__=='__main__':main()
