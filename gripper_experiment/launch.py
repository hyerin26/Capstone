"""Sequential, resumable VESSL pipeline with pilot gates and status reporting."""
import json,subprocess,sys,time,hashlib,os,traceback,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON=str(ROOT.parent/'.venv/bin/python')
PLAN=json.loads((ROOT/'plan.json').read_text())
ENV=dict(os.environ,USE_TF='0',TOKENIZERS_PARALLELISM='false',PYTHONNOUSERSITE='1',
         VK_ICD_FILENAMES='/root/.config/vulkan/nvidia_egl_icd.json',XDG_RUNTIME_DIR='/root/.cache/xdg-runtime')
def status(**kw):
 p=ROOT/'status.json';old=json.loads(p.read_text()) if p.exists() else {}
 old.update(updated_at_unix=time.time(),**kw);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(old,indent=2));tmp.replace(p)
def invoke(index,mode,conditions=''):
 r=PLAN['cases'][index];name=f"{mode}-{index:03d}";log=ROOT/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
 status(state='running',current=dict(index=index,case_id=r['case_id'],split=r['split'],mode=mode),log=str(log))
 cmd=[PYTHON,'-u',str(ROOT/'worker.py'),'--case-index',str(index),'--mode',mode]
 if conditions:cmd+=['--conditions',conditions]
 with log.open('a') as f:subprocess.run(cmd,env=ENV,cwd=ROOT.parent,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=7200)
 subprocess.run([PYTHON,str(ROOT/'summarize.py')],env=ENV,check=True)

def main():
 (ROOT/'logs').mkdir(exist_ok=True)
 # Prevent overlapping launchers. This lock is released by the OS on exit.
 import fcntl
 lock=(ROOT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
 config=dict(plan_sha256=hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest(),source_hashes=source_hashes,
             simulator_commit=subprocess.check_output(['git','-C','/root/YeahPick','rev-parse','HEAD'],text=True).strip(),
             revision='4159b1dedade8f88b5bbaaae7810299dee5228f5',expected=dict(pilot_fixed=32,pilot_rollout=16,validation_fixed=320,test_rollout=600))
 cfg=ROOT/'run_config.json'
 if cfg.exists():assert json.loads(cfg.read_text())==config,'Source/protocol changed: use a new output version'
 else:cfg.write_text(json.dumps(config,indent=2))
 for rel,digest in PLAN['asset_hashes'].items():assert hashlib.sha256((ROOT.parent/'bundle'/rel).read_bytes()).hexdigest()==digest
 # The initial interactive pilot must finish before this launcher takes over its case.
 if '--wait-existing-pilot' in sys.argv:
  cid=hashlib.sha256(PLAN['cases'][70]['case_id'].encode()).hexdigest()[:16]
  final=ROOT/'outputs_v1/rollout'/cid/'cube_1/episode.json';start=time.monotonic()
  while not final.exists():
   status(state='waiting_for_existing_pilot')
   if time.monotonic()-start>1800:raise TimeoutError('Existing pilot did not finish')
   time.sleep(5)
 for i,r in enumerate(PLAN['cases']):
  if r['split']=='pilot':invoke(i,'fixed')
 # Technical gate: genuine finite action outputs, unchanged zero-intervention sequence,
 # matched initial positions, visible gripper and essentially equal-area background masks.
 episodes=[json.loads(p.read_text()) for p in (ROOT/'outputs_v1/fixed').glob('*/*/episode.json')]
 pilot=[e for e in episodes if e['case']['split']=='pilot']
 assert len(pilot)==32
 for r in pilot:
  t=r['traces'][0];assert r['completed'] and t['policy']['zero_hook_sequence_validated']
  assert t['mask_meta']['mask_pixels']['gripper']>0 and t['mask_meta']['background_area_ratio']>=.95
 gate=dict(passed=True,pilot_fixed_records=32,checked='valid action tokens, zero-hook sequence identity, gripper visibility, background area >=95%')
 (ROOT/'technical_gate.json').write_text(json.dumps(gate,indent=2))
 for i,r in enumerate(PLAN['cases']):
  if r['split']=='pilot':invoke(i,'rollout','baseline,gripper_1,background_1,cube_1')
 for i,r in enumerate(PLAN['cases']):
  if r['split']=='validation':invoke(i,'fixed')
 # Fixed-state validation is diagnostic only; strength=1 for phase ablations is frozen,
 # not selected from validation/test success. No automatic outcome-based tuning.
 for i,r in enumerate(PLAN['cases']):
  if r['split']=='test':invoke(i,'rollout')
 counts=collections.Counter()
 for p in (ROOT/'outputs_v1').glob('*/*/*/episode.json'):
  r=json.loads(p.read_text());counts[(r['case']['split'],r['mode'])]+=1
 assert counts[('test','rollout')]==600 and counts[('validation','fixed')]==320
 subprocess.run([PYTHON,str(ROOT/'summarize.py')],env=ENV,check=True)
 status(state='complete',record_counts={str(k):v for k,v in counts.items()})
 # Shut down only the policy daemon created for this experiment.
 (ROOT/'queue/STOP').touch()
try:main()
except Exception:
 status(state='failed',error=traceback.format_exc());traceback.print_exc();raise
