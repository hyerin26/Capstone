"""Independent server-side watchdog; stop only this experiment's registered PIDs."""
import json,os,signal,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'six_hour_plan.json').read_text())
while time.time()<plan['deadline_unix']:time.sleep(max(0,min(5,plan['deadline_unix']-time.time())))
status=json.loads((ROOT/'status.json').read_text())
pids=[plan['launcher_pid'],plan['existing_worker_pid'],status.get('active_worker_pid')]
ready=ROOT/'queue/ready.json'
if ready.exists():pids.append(json.loads(ready.read_text())['pid'])
for sig in [signal.SIGTERM,signal.SIGKILL]:
 for pid in set(p for p in pids if p):
  try:
   cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
   if b'gripper_experiment/' in cmd:os.kill(pid,sig)
  except (FileNotFoundError,ProcessLookupError):pass
 if sig==signal.SIGTERM:time.sleep(2)
if status.get('state') not in ['complete_reduced_plan','time_limit_reached','failed']:
 status.update(state='time_limit_reached',updated_at_unix=time.time(),active_worker_pid=None)
 temp=ROOT/'status.deadline.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(ROOT/'status.json')
(ROOT/'deadline_guard_result.json').write_text(json.dumps(dict(deadline_unix=plan['deadline_unix'],finished_unix=time.time(),action='experiment processes stopped or already exited'),indent=2))
