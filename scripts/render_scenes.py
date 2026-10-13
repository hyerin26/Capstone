"""Render recorded pair layouts through the existing YeahPick simulator.

One subprocess per small batch bounds SAPIEN GPU lifetime. No robot actions beyond
its existing ten-step initial grasp are evaluated. RGBs are pre-policy views.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True)
    p.add_argument('--yeahpick', required=True)
    p.add_argument('--cases', default='outputs/audit/explicit_object_cases.jsonl')
    p.add_argument('--output', default='outputs/scenes')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--batch', type=int, default=5)
    p.add_argument('--worker', help='JSON file of cases, internal subprocess argument')
    a = p.parse_args()
    if not a.worker:
        rows = [json.loads(x) for x in Path(a.cases).read_text().splitlines()]
        if a.limit:
            rows = rows[:a.limit]
        groups = {}
        for r in rows:
            groups.setdefault((r['frozen_asset'], r['swap']), []).append(r)
        out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
        for group in groups.values():
            for start in range(0, len(group), a.batch):
                block = group[start:start+a.batch]
                job = out / ('job_' + hashlib.sha256(json.dumps(block).encode()).hexdigest()[:16] + '.json')
                job.write_text(json.dumps(block))
                subprocess.run([sys.executable, __file__, '--bundle', a.bundle, '--yeahpick', a.yeahpick,
                                '--output', str(out), '--worker', str(job)], check=True)
        return
    import numpy as np
    import torch
    from PIL import Image
    yp = Path(a.yeahpick).resolve()
    sys.path[:0] = [str(yp/'SimplerEnv'), str(yp/'ManiSkill')]
    import mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.put_on_in_scene_multi_v4 as task
    task.CARROT_DATASET_DIR = Path(a.bundle).resolve()/'assets'
    from simpler_env.run import Args
    from simpler_env.env.simpler_wrapper_v4 import SimplerWrapper
    rows = json.loads(Path(a.worker).read_text())
    args = Args()
    args.assets = rows[0]['frozen_asset']
    args.ids = [int(r['question_id']) for r in rows]
    args.num_envs = len(rows)
    args.seed = 0
    args.do_swap = rows[0]['swap'] == '1'
    args.instruction_condition = 'explicit_object'
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    wrapper = SimplerWrapper(args)
    try:
        rgb, instructions, info = wrapper.reset('test')
        env = wrapper.env.unwrapped
        obs = env.get_obs()
        sensor = obs['sensor_data']['3rd_view_camera']
        assert torch.equal(sensor['rgb'].to(torch.uint8), rgb), 'Mask and policy RGB disagree'
        segmentation = sensor['segmentation'].cpu().numpy()
        if segmentation.shape[-1] != 1:
            raise ValueError(f'Unexpected segmentation shape: {segmentation.shape}')
        for i, r in enumerate(rows):
            case = hashlib.sha256(r['case_id'].encode()).hexdigest()[:16]
            out = Path(a.output)/case; out.mkdir(parents=True, exist_ok=True)
            img = rgb[i].cpu().numpy()
            Image.fromarray(img).save(out/'rgb.png')
            centers = {}
            for side in ('left', 'right'):
                name = getattr(env, f'_current_{side}_names')[i]
                assert name == r[f'{side}_tile'], (name, r[f'{side}_tile'])
                actor_id = int(env.objs_board[name].per_scene_id[i].item())
                mask = segmentation[i, ..., 0] == actor_id
                if not mask.any():
                    raise ValueError(f'Empty {side} mask in {r["case_id"]}')
                ys, xs = np.nonzero(mask)
                centers[side] = float(xs.mean())
                Image.fromarray(mask.astype('uint8')*255).save(out/f'{side}_mask.png')
            assert centers['left'] < centers['right'], 'World side disagrees with screen side'
            metadata = dict(r, scene_id=case, rgb='rgb.png', left_mask='left_mask.png',
                            right_mask='right_mask.png', scene_seed=0, init_grasp_steps=10,
                            rendered_instruction=instructions[i],
                            rgb_sha256=hashlib.sha256((out/'rgb.png').read_bytes()).hexdigest(),
                            mask_scope='whole candidate tile, not inner pictured object',
                            simulator_commit=subprocess.check_output(['git','-C',str(yp),'rev-parse','HEAD'],text=True).strip())
            (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
            print('RENDERED', r['case_id'], flush=True)
    finally:
        wrapper.env.close()

if __name__ == '__main__':
    main()
