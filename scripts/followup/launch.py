"""Launch two resumable workers after completed input and pilot validation."""
import json,subprocess,sys,time
from pathlib import Path

def status(stage,**kw):
    p=Path('outputs/followup');p.mkdir(exist_ok=True)
    value=dict(stage=stage,updated_at=time.time(),**kw)
    temp=p/'pipeline_status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(p/'pipeline_status.json');print(json.dumps(value),flush=True)

def main():
    logs=[];processes=[]
    try:
        started=time.time()
        while not Path('outputs/followup_inputs/manifest.json').exists():
            status('waiting_for_images',rendered=len(list(Path('outputs/followup_inputs').glob('*/metadata.json'))))
            if time.time()-started>1200:raise TimeoutError('Image preparation did not complete')
            time.sleep(15)
        manifest=json.loads(Path('outputs/followup_inputs/manifest.json').read_text());assert manifest['cases']==500
        assert len(list(Path('outputs/followup_inputs').glob('*/metadata.json')))==500
        assert Path('outputs/followup_pilot/complete_0.json').exists()
        pilot=list(Path('outputs/followup_pilot').glob('*/result.json'));assert len(pilot)==10
        for f in pilot:
            r=json.loads(f.read_text());assert len(r['records'])==13 and r['baseline_raw_agreement']
            assert r['records'][0]['score_validation_max_abs_difference']<=1e-4
        for worker in (0,1):
            log=Path(f'logs/followup-worker-{worker}.log').open('a');logs.append(log)
            cmd=[sys.executable,'-u','scripts/followup/run.py','--worker',str(worker),'--workers','2','--device',f'cuda:{worker}']
            processes.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            for p in processes:
                if p.poll() not in (None,0):raise RuntimeError(f'Worker failed with {p.returncode}')
            count=len(list(Path('outputs/followup').glob('*/result.json')))
            status('running',completed_scenes=count,expected_scenes=500,completed_generations=count*13,expected_generations=6500)
            time.sleep(30)
        assert all(p.returncode==0 for p in processes)
        status('summarizing',completed_scenes=500)
        subprocess.run([sys.executable,'scripts/followup/summarize.py'],check=True)
        subprocess.run([sys.executable,'scripts/followup/plot.py'],check=True)
        status('complete',completed_scenes=500,completed_generations=6500)
    except Exception as e:
        for p in processes:
            if p.poll() is None:p.terminate()
        status('failed',error=str(e));raise
    finally:
        for f in logs:f.close()

if __name__=='__main__':main()
