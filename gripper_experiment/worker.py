"""Bounded-lifetime simulator worker; matched resets and dynamic actor-ID masks."""
import argparse,json,sys,time,hashlib,random,copy
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parent
sys.path[:0]=['/root/YeahPick/SimplerEnv','/root/YeahPick/ManiSkill']
import mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.put_on_in_scene_multi_v4 as task
from simpler_env.run import Args
from simpler_env.env.simpler_wrapper_v4 import SimplerWrapper

def atom(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)

def read_info(info):
    return {k:v[0].detach().cpu().tolist() for k,v in info.items() if torch.is_tensor(v) and v.numel() and k not in ['pi_0']}

def masks_of(env,info):
    sensor=env.get_obs(info=info)['sensor_data']['3rd_view_camera']
    rgb=sensor['rgb'][0].to(torch.uint8).cpu().numpy()
    seg=sensor['segmentation'][0,...,0].cpu().numpy()
    gripper_ids=[];robot_ids=[];names=[]
    for name,link in env.agent.robot.links_map.items():
        idx=int(link._objs[0].entity.per_scene_id);robot_ids.append(idx)
        if name in ['gripper_link','ee_gripper_link','left_finger_link','right_finger_link']:
            gripper_ids.append(idx);names.append(name)
    assert {'left_finger_link','right_finger_link'}<=set(names),names
    grip=np.isin(seg,gripper_ids)
    cube_actor=env.cubes[env._white_key]
    cube=seg==int(cube_actor.per_scene_id[0].item())
    masks={'gripper':grip,'cube':cube}
    for side in ['left','right']:
        actor=env.objs_board[getattr(env,'_current_'+side+'_names')[0]]
        masks[side]=seg==int(actor.per_scene_id[0].item())
    bg=~(np.isin(seg,robot_ids)|cube|masks['left']|masks['right'])
    # Translate the same visible gripper shape; maximize retained background occupancy.
    # Never wrap at frame boundaries. Ties use a fixed candidate order.
    h,w=seg.shape;ys,xs=np.nonzero(grip)
    best=np.zeros_like(grip);score=-1
    for dy in [-h//2,-h//4,h//4,h//2]:
        for dx in [-w//2,-w//4,w//4,w//2]:
            yy=ys+dy;xx=xs+dx;ok=(yy>=0)&(yy<h)&(xx>=0)&(xx<w)
            candidate=np.zeros_like(grip);candidate[yy[ok],xx[ok]]=True;candidate &= bg
            n=int(candidate.sum())
            if n>score:best=candidate;score=n
    masks['background']=best
    return rgb,masks,dict(gripper_links=names,mask_pixels={k:int(v.sum()) for k,v in masks.items()},background_area_ratio=float(best.sum()/max(grip.sum(),1)))

def request(rgb,masks,instruction,cond,path):
    inp=path.with_suffix('.input.npz');np.savez_compressed(inp,rgb=rgb,**masks)
    req=ROOT/'queue'/(hashlib.sha256(str(path).encode()).hexdigest()+'.request.json')
    atom(req,dict(input=str(inp),instruction=instruction,region=cond['region'],strength=cond['strength'],output=str(path)))
    start=time.monotonic()
    while not path.exists():
        if path.with_suffix('.error').exists():raise RuntimeError(path.with_suffix('.error').read_text())
        if time.monotonic()-start>600:raise TimeoutError('Policy request exceeded 600 seconds')
        time.sleep(.1)
    return json.loads(path.read_text())

def phase_name(env,info):
    grasp=bool(info.get('diag_gripper_state',torch.tensor([False]))[0])
    tcp=env.agent.tcp.pose.p[0].cpu().numpy()
    d=min(np.linalg.norm((tcp-env.objs_board[getattr(env,'_current_'+side+'_names')[0]].pose.p[0].cpu().numpy())[:2]) for side in ['left','right'])
    return 'placement' if d<.12 else ('transport' if grasp else 'ungrasped')

def physical(env):
    cube=env.cubes[env._white_key];p=cube.pose.p[0].cpu().numpy()
    grasp=bool(env.agent.is_grasping(cube)[0]);velocity=float(torch.linalg.vector_norm(cube.linear_velocity[0]).item())
    on={};tcp_over={};tcp=env.agent.tcp.pose.p[0].cpu().numpy()
    for side in ['left','right']:
        name=getattr(env,'_current_'+side+'_names')[0];board=env.objs_board[name]
        bp=board.pose.p[0].cpu().numpy();half=env.model_bbox_sizes[name].cpu().numpy()/2
        rotation=board.pose.to_transformation_matrix()[0,:3,:3].cpu().numpy();half=np.abs(rotation)@half
        on[side]=bool(np.all(np.abs(p[:2]-bp[:2])<=half[:2]+.015) and abs(p[2]-(bp[2]+half[2]+task.CUBE_HALF_SIZE))<=.02)
        tcp_over[side]=bool(np.all(np.abs(tcp[:2]-bp[:2])<=half[:2]+.015) and tcp[2]>=bp[2]+half[2]+.02)
    return dict(tcp_over=tcp_over,cube=p.tolist(),tcp=env.agent.tcp.pose.p[0].cpu().tolist(),qpos=env.agent.robot.get_qpos()[0].cpu().tolist(),grasped=grasp,velocity=velocity,on=on)

def main():
    p=argparse.ArgumentParser();p.add_argument('--case-index',type=int,required=True);p.add_argument('--mode',choices=['fixed','rollout'],required=True);p.add_argument('--conditions',default='');p.add_argument('--steps',type=int,default=80);a=p.parse_args()
    plan=json.loads((ROOT/'plan.json').read_text());case=plan['cases'][a.case_index]
    conditions=plan['conditions']
    if a.conditions:conditions=[c for c in conditions if c['name'] in a.conditions.split(',')]
    elif a.mode=='fixed':conditions=conditions[:8]
    task.CARROT_DATASET_DIR=Path('/root/Capstone/bundle/assets')
    args=Args();args.assets=case['frozen_asset'];args.ids=[int(case['question_id'])];args.num_envs=1;args.seed=0;args.do_swap=case['swap']=='1';args.instruction_condition='explicit_object';args.episode_len=80
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    wrapper=SimplerWrapper(args);env=wrapper.env.unwrapped
    base_state=None;snapshot=None
    case_key=hashlib.sha256(case['case_id'].encode()).hexdigest()[:16]
    snapshot_path=ROOT/'outputs_v1'/a.mode/case_key/'initial_state.pt'
    try:
        for cond in conditions:
            dest=ROOT/'outputs_v1'/a.mode/case_key/cond['name'];dest.mkdir(parents=True,exist_ok=True)
            if (dest/'episode.json').exists():continue
            # Stale per-step responses are reusable only with a persisted identical initial snapshot.
            wrapper.env.reset(seed=[0]);rgb,instructions,info=wrapper.reset('test')
            rendered_instruction=instructions[0]
            instructions=[case['instruction']]  # Exact recorded bundle instruction, no silent rephrasing.
            assert env.where_target()[0]==case['answer_side']
            if snapshot is None:
                if snapshot_path.exists():snapshot=torch.load(snapshot_path,map_location=env.device)
                else:
                    snapshot=copy.deepcopy(dict(sim=env.get_state_dict(),agent=env.agent.get_state(),elapsed=env._elapsed_steps,stats=env.episode_stats,info=info))
                    torch.save(snapshot,snapshot_path)
            if snapshot is not None:
                env.set_state_dict(copy.deepcopy(snapshot['sim']))
                env.agent.set_state(copy.deepcopy(snapshot['agent']))
                env._elapsed_steps=snapshot['elapsed'].clone();env.episode_stats=copy.deepcopy(snapshot['stats']);info=copy.deepcopy(snapshot['info'])
            initial=physical(env)
            if base_state is None:base_state=initial
            else:
                for key in ['cube','tcp','qpos']:assert np.allclose(initial[key],base_state[key],atol=1e-6),('Reset mismatch',key)
            state_hash=hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
            traces=[];frames=[];stable=0;max_stable=0;first_release=None;first_entry=None
            for step in range(1 if a.mode=='fixed' else a.steps):
                frame,masks,mask_meta=masks_of(env,info)
                if step==0:
                    Image.fromarray(frame).save(dest/'initial.png')
                    for name,mask in masks.items():Image.fromarray(mask.astype('uint8')*255).save(dest/(name+'_mask.png'))
                phase=phase_name(env,info)
                active=cond['phase']=='all' or cond['phase']==phase
                actual=dict(cond,strength=cond['strength'] if active else 0.)
                result=request(frame,masks,instructions[0],actual,dest/f'step_{step:03d}.json')
                entry=dict(step=step,phase=phase,active=active,mask_meta=mask_meta,policy=result,pre_state=physical(env))
                if not result['valid_action']:
                    traces.append(entry);break
                if a.mode=='rollout':
                    action=torch.tensor([result['action']],dtype=torch.float32,device=env.device)
                    # SimplerWrapper forces gripper closed for first 15 policy steps.
                    _,_,_,info=wrapper.step(action)
                    after=physical(env);entry['post_state']=after;entry['executed_action']=action[0].cpu().tolist();entry['info']=read_info(info)
                    if first_release is None and entry['pre_state']['grasped'] and not after['grasped']:
                        first_release=dict(step=step,correct=after['on'][case['answer_side']],state=after)
                    entered=[s for s in ['left','right'] if after['tcp_over'][s]]
                    if first_entry is None and len(entered)==1:first_entry=dict(step=step,side=entered[0],correct=entered[0]==case['answer_side'])
                    ok=after['on'][case['answer_side']] and not after['grasped'] and after['velocity']<.05
                    stable=stable+1 if ok else 0;max_stable=max(stable,max_stable)
                traces.append(entry);frames.append(frame)
                # Input npz no longer needed; preserve first and fixed-state images/masks.
                if step>0 and a.mode=='rollout':(dest/f'step_{step:03d}.input.npz').unlink()
            completed=len(traces)==(1 if a.mode=='fixed' else a.steps) and all(t['policy']['valid_action'] for t in traces)
            record=dict(rendered_instruction=rendered_instruction,case=case,condition=cond,mode=a.mode,completed=completed,steps=len(traces),initial_state=initial,initial_state_hash=state_hash,
                        released_stable_correct_5_steps=max_stable>=5 and completed,final_stable_correct_5_steps=stable>=5 and completed,first_release=first_release,first_entry=first_entry,
                        benchmark_success=bool(info.get('success',torch.tensor([False]))[0]) and completed,final_state=physical(env),traces=traces)
            atom(dest/'episode.json',record)
            if a.mode=='rollout' and frames:
                from mani_skill.utils.visualization.misc import images_to_video
                images_to_video(frames,str(dest),'rollout',fps=5,verbose=False)
            print(json.dumps(dict(case=case_key,condition=cond['name'],completed=completed,steps=len(traces),success=record['released_stable_correct_5_steps'])),flush=True)
    finally:wrapper.env.close()
if __name__=='__main__':main()
