"""User-requested six-hour budget; preserve original frozen plan and raw records."""
import collections,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON=str(ROOT.parent/'.venv/bin/python')
PLAN=json.loads((ROOT/'plan.json').read_text());BUDGET=json.loads((ROOT/'six_hour_plan.json').read_text())
ENV=dict(os.environ,USE_TF='0',TOKENIZERS_PARALLELISM='false',PYTHONNOUSERSITE='1',VK_ICD_FILENAMES='/root/.config/vulkan/nvidia_egl_icd.json',XDG_RUNTIME_DIR='/root/.cache/xdg-runtime')
CHILD=None
class Deadline(Exception):pass

def status(**kw):
 p=ROOT/'status.json';old=json.loads(p.read_text()) if p.exists() else {};old.update(updated_at_unix=time.time(),**kw)
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(old,indent=2));tmp.replace(p)

def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False

def kill_owned_policy():
 ready=ROOT/'queue/ready.json'
 if ready.exists():
  pid=json.loads(ready.read_text())['pid']
  try:
   cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
   if b'gripper_experiment/policy.py' in cmd:os.kill(pid,signal.SIGTERM)
  except (FileNotFoundError,ProcessLookupError):pass

def invoke(index,mode,conditions):
 global CHILD
 remaining=BUDGET['deadline_unix']-time.time()-60
 if remaining<=0:raise Deadline()
 case=PLAN['cases'][index];log=ROOT/'logs'/f'bounded-{mode}-{index:03d}.log'
 status(state='running',budget='six_hours',deadline_unix=BUDGET['deadline_unix'],current=dict(index=index,case_id=case['case_id'],split=case['split'],mode=mode),log=str(log))
 with log.open('a') as f:
  CHILD=subprocess.Popen([PYTHON,'-u',str(ROOT/'worker.py'),'--case-index',str(index),'--mode',mode,'--conditions',','.join(conditions)],env=ENV,cwd=ROOT.parent,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  status(active_worker_pid=CHILD.pid)
  try:rc=CHILD.wait(timeout=remaining)
  except subprocess.TimeoutExpired:
   os.killpg(CHILD.pid,signal.SIGTERM)
   try:CHILD.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(CHILD.pid,signal.SIGKILL);CHILD.wait()
   CHILD=None;raise Deadline()
  CHILD=None
  if rc:raise RuntimeError(f'Worker {index} exited {rc}; see {log}')
 subprocess.run([PYTHON,str(ROOT/'summarize.py')],env=ENV,check=True,timeout=40)

def main():
 lock=(ROOT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((ROOT/'technical_gate.json').read_text())['passed']
 status(state='waiting_for_existing_worker',budget='six_hours',deadline_unix=BUDGET['deadline_unix'],active_worker_pid=BUDGET['existing_worker_pid'])
 while alive(BUDGET['existing_worker_pid']):
  if time.time()>=BUDGET['deadline_unix']-60:
   os.kill(BUDGET['existing_worker_pid'],signal.SIGTERM);raise Deadline()
  time.sleep(1)
 for i,r in enumerate(PLAN['cases']):
  if r['split']=='validation':invoke(i,'fixed',BUDGET['conditions'])
 for i in BUDGET['test_case_indices']:invoke(i,'rollout',BUDGET['conditions'])
 expected=len(BUDGET['test_case_indices'])*len(BUDGET['conditions'])
 actual=sum((ROOT/'outputs_v1/rollout'/hashlib.sha256(PLAN['cases'][i]['case_id'].encode()).hexdigest()[:16]/c/'episode.json').exists() for i in BUDGET['test_case_indices'] for c in BUDGET['conditions'])
 assert actual==expected
 return 'complete_reduced_plan'

state='failed';error=None
try:state=main()
except Deadline:state='time_limit_reached'
except Exception:error=traceback.format_exc();traceback.print_exc()
finally:
 kill_owned_policy()
 try:subprocess.run([PYTHON,str(ROOT/'summarize.py')],env=ENV,check=True,timeout=40)
 except Exception:traceback.print_exc()
 status(state=state,budget='six_hours',deadline_unix=BUDGET['deadline_unix'],active_worker_pid=None,error=error)
