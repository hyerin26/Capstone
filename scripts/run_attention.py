"""Text-answer Magma grounding evaluation with last-prompt-position attention.

Uses the official Magma visual-token merge and reconstructs its crop grid with
row separators removed. This is a text-answer task, not an action rollout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM


def fix_layerscale(model, checkpoint, loading):
    """Restore timm gamma values if Transformers renamed checkpoint LayerScale keys."""
    from safetensors import safe_open
    keys = set(model.state_dict())
    remap = {}
    for file in Path(checkpoint).glob('*.safetensors'):
        with safe_open(file, framework='pt', device='cpu') as sf:
            for key in sf.keys():
                if 'vision_tower' in key and re.search(r'blocks\.\d+\.(weight|gamma)$', key):
                    target = re.sub(r'\.weight$', '.gamma', key)
                    if target in keys:
                        remap[target] = sf.get_tensor(key)
    if remap:
        model.load_state_dict(remap, strict=False)
    # Magma's dense encoder calls only trunk.stem and trunk.stages. The CLIP
    # pooled-classification heads are not in its checkpoint or inference path.
    unused={'vision_tower.clip_vision_model.head.proj.weight',
            'vision_tower.clip_vision_model.trunk.head.norm.weight',
            'vision_tower.clip_vision_model.trunk.head.norm.bias'}
    def reject_unused_head(module, inputs):
        raise RuntimeError('Untrained CLIP pooling head unexpectedly entered inference')
    model.vision_tower.clip_vision_model.head.register_forward_pre_hook(reject_unused_head)
    model.vision_tower.clip_vision_model.trunk.head.register_forward_pre_hook(reject_unused_head)
    loading['excluded_unused_head_keys']=sorted(set(loading.get('missing_keys',[])) & unused)
    missing = set(loading.get('missing_keys', [])) - set(remap) - unused
    unexpected=set(loading.get('unexpected_keys',[])) - {re.sub(r'\.gamma$', '.weight', k) for k in remap}
    if unexpected:raise RuntimeError(f'Unexpected checkpoint parameters: {unexpected}')
    if missing or loading.get('mismatched_keys'):
        raise RuntimeError(f'Unresolved checkpoint weights: {missing}, {loading.get("mismatched_keys")}')
    return sorted(remap)


class Probe:
    def __init__(self, model, processor):
        self.model, self.processor = model, processor
        self.rows = {}
        self.shape = None
        self.capture = False
        self.handles = []
        def project_hook(module, inputs, output):
            self.shape = tuple(output.shape)
        self.handles.append(model.multi_modal_projector.register_forward_hook(project_hook))
        for i, layer in enumerate(model.language_model.model.layers):
            def hook(module, inputs, output, index=i):
                if self.capture:
                    if output[1] is None:
                        raise RuntimeError('Attention backend did not return attention weights')
                    self.rows[index] = output[1][0, :, -1, :].detach().float().cpu().numpy()
                    # Discard quadratic matrices after copying the one required row.
                    return (output[0], None, *output[2:])
            self.handles.append(layer.self_attn.register_forward_hook(hook))

    def prepare(self, image, instruction):
        marker = '<image_start><image><image_end>' if self.model.config.mm_use_image_start_end else '<image>'
        question = ('Which tile is the target of the following instruction?\n'
                    + instruction + '\nAnswer with only LEFT or RIGHT, using the viewer\'s perspective.')
        messages = [{'role':'system','content':'You are agent that can see, talk and act.'},
                    {'role':'user','content':marker+'\n'+question}]
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        x = self.processor(images=[image], texts=prompt, return_tensors='pt')
        x['pixel_values'] = x['pixel_values'].unsqueeze(0)
        x['image_sizes'] = x['image_sizes'].unsqueeze(0)
        device = self.model.get_input_embeddings().weight.device
        x = {k:v.to(device=device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in x.items()}
        return x, prompt

    @torch.inference_mode()
    def forward(self, image, instruction, capture=False):
        x, prompt = self.prepare(image, instruction)
        self.rows = {}; self.capture = capture
        output = self.model(**x, use_cache=False, output_attentions=capture, return_dict=True)
        logits = output.logits[0,-1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite logits; do not silently sanitize inference')
        labels = {s:self.processor.tokenizer.encode(s, add_special_tokens=False) for s in ('LEFT','RIGHT')}
        if any(len(v)!=1 for v in labels.values()):
            raise ValueError(f'Expected single-token candidate labels, got {labels}')
        scores = {s:float(logits.log_softmax(-1)[v[0]].item()) for s,v in labels.items()}
        first_id = int(logits.argmax())
        raw = self.processor.tokenizer.decode([first_id], skip_special_tokens=True)
        data = dict(scores=scores, candidate_prediction=max(scores, key=scores.get).lower(),
                    first_token_id=first_id, first_token=raw, prompt=prompt,
                    input_ids=x['input_ids'][0].cpu().tolist(), image_sizes=x['image_sizes'].cpu().tolist())
        if capture:
            shape = self.shape
            if self.model.config.vision_config['img_anyres_strategy'] != 'crop' or len(shape)!=4:
                raise ValueError(f'Unvalidated visual layout: {shape}')
            gh, gw = map(int, x['image_sizes'][0,0].tolist())
            nc, ph, pw, _ = shape
            assert nc == gh*gw
            h, w = gh*ph, gw*pw
            positions = (x['input_ids'][0] == self.model.config.image_token_index).nonzero().flatten()
            assert len(positions)==1
            start = int(positions[0]); length=h*(w+1)
            rows = np.stack([self.rows[i] for i in range(len(self.model.language_model.model.layers))])
            expected = x['input_ids'].shape[1]-1+length
            assert rows.shape[-1] == expected, (rows.shape, expected)
            assert np.allclose(rows.sum(-1), 1, atol=.015)
            maps = rows[..., start:start+length].reshape(rows.shape[0],rows.shape[1],h,w+1)[...,:w]
            data.update(grid=[h,w], visual_start=start, visual_length_with_separators=length,
                        visual_mass_per_layer=maps.sum((-1,-2)).mean(1).tolist())
            self.capture = False
            return data, maps, x
        return data, None, x


def parse_side(raw):
    text = raw.strip().strip('.!').strip().upper()
    return text.lower() if text in ('LEFT','RIGHT') else None


def summarize_maps(maps, masks):
    # Fractional occupancy per visual cell; area normalization prevents large-tile bias.
    h,w = maps.shape[-2:]
    total = maps.sum((-1,-2))
    metrics = {}
    for side, mask in masks.items():
        occ = np.array(Image.fromarray(mask.astype('float32')).resize((w,h), Image.Resampling.BOX))
        mass = (maps*occ).sum((-1,-2))
        metrics[side+'_mass_by_layer'] = mass.mean(1).tolist()
        metrics[side+'_fraction_by_layer'] = (mass / np.maximum(total,1e-12)).mean(1).tolist()
        metrics[side+'_density_by_layer'] = (mass / max(float(occ.sum()),1e-12)).mean(1).tolist()
        peak = maps.mean(1).reshape(maps.shape[0],-1).argmax(-1)
        metrics[side+'_peak_hit_by_layer'] = (occ.reshape(-1)[peak] >= .5).tolist()
    return metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--revision',default='4159b1dedade8f88b5bbaaae7810299dee5228f5')
    p.add_argument('--scenes',default='outputs/scenes')
    p.add_argument('--output',default='outputs/attention')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--occlusion',action='store_true')
    p.add_argument('--validate-hook',action='store_true')
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    config=AutoConfig.from_pretrained(a.checkpoint,trust_remote_code=True,code_revision=a.revision)
    config._attn_implementation='eager';config.text_config._attn_implementation='eager'
    # Official image-tower code downloads CLIP even with require_pretrained=False.
    # Skip only that unused weight download; still load its exact architecture JSON.
    # All Magma weights are checked below, so random missing weights cannot pass.
    import open_clip.pretrained
    original_download=open_clip.pretrained.download_pretrained_from_hf
    def architecture_only(model_id, *args, **kwargs):
        if model_id=='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg' and not args and kwargs.get('filename') is None:
            return None
        return original_download(model_id,*args,**kwargs)
    open_clip.pretrained.download_pretrained_from_hf=architecture_only
    print('Loading complete Magma checkpoint (unused CLIP weight download skipped)',flush=True)
    model, loading = AutoModelForCausalLM.from_pretrained(a.checkpoint,config=config,
        trust_remote_code=True,code_revision=a.revision,torch_dtype=torch.bfloat16,attn_implementation='eager',
        device_map='auto',max_memory={i:('14GiB' if i==0 and torch.cuda.device_count()>1 else '20GiB') for i in range(torch.cuda.device_count())},
        output_loading_info=True)
    open_clip.pretrained.download_pretrained_from_hf=original_download
    restored=fix_layerscale(model,a.checkpoint,loading)
    print('Magma loaded; checkpoint keys validated',flush=True)
    model.eval()
    processor=AutoProcessor.from_pretrained(a.checkpoint,trust_remote_code=True,code_revision=a.revision)
    probe=Probe(model,processor)
    (out/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,
        loading=loading,restored_layerscale=restored,device_map=model.hf_device_map,
        checkpoint=a.checkpoint,revision=a.revision,config=config.to_dict()),indent=2,default=str))
    signature=hashlib.sha256(Path(__file__).read_bytes()+json.dumps(dict(config=config.to_dict(),occlusion=a.occlusion),sort_keys=True,default=str).encode()).hexdigest()
    files=sorted(Path(a.scenes).glob('*/metadata.json'))
    if a.limit: files=files[:a.limit]
    if not files: raise ValueError('No rendered scenes')
    for index,f in enumerate(files):
        meta=json.loads(f.read_text());dest=out/meta['scene_id'];dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():
            previous=json.loads((dest/'result.json').read_text())
            if previous.get('method_signature')!=signature:raise RuntimeError('Cannot mix different protocols in one output directory')
            continue
        begin=time.time()
        image=Image.open(f.parent/'rgb.png').convert('RGB')
        masks={s:np.array(Image.open(f.parent/f'{s}_mask.png'))>0 for s in ('left','right')}
        result,maps,x=probe.forward(image,meta['instruction'],capture=True)
        if a.validate_hook:
            normal,_,_=probe.forward(image,meta['instruction'],capture=False)
            if normal['first_token_id']!=result['first_token_id'] or any(abs(normal['scores'][k]-result['scores'][k])>.1 for k in result['scores']):
                raise RuntimeError('Attention collection changed model prediction')
            result['hook_validated']=True
        probe.capture=False
        with torch.inference_mode():
            generated=model.generate(**x,max_new_tokens=8,do_sample=False,use_cache=True,
                                     output_attentions=False,pad_token_id=processor.tokenizer.eos_token_id)
        raw=processor.tokenizer.decode(generated[0,x['input_ids'].shape[1]:],skip_special_tokens=True)
        prediction=parse_side(raw)
        result.update(meta,method_signature=signature,raw_response=raw,predicted_side=prediction,valid=prediction is not None,
                      correct=prediction==meta['answer_side'],candidate_correct=result['candidate_prediction']==meta['answer_side'])
        result.update(summarize_maps(maps,masks))
        np.savez_compressed(dest/'attention.npz',head_maps=maps.astype('float16'))
        # Fixed panel layout and shared scale for visual comparison across layers of this example.
        meanmaps=maps.mean(1)
        vmax=max(float(meanmaps.max()),1e-12)
        panels=[]
        for layer in (0,7,15,23,31):
            if layer>=len(meanmaps):continue
            heat=np.array(Image.fromarray(meanmaps[layer]).resize(image.size,Image.Resampling.BILINEAR))/vmax
            base=np.asarray(image).astype(float)
            color=np.zeros_like(base);color[...,0]=255
            alpha=(.65*heat)[...,None]
            panels.append(Image.fromarray(np.uint8(base*(1-alpha)+color*alpha)))
        canvas=Image.new('RGB',(image.width*len(panels),image.height))
        for i,panel in enumerate(panels):canvas.paste(panel,(i*image.width,0))
        canvas.save(dest/'layers_01_08_16_24_32.png')
        if a.occlusion:
            variants={s:mask for s,mask in masks.items()}
            # Equal pixel count background control. Deterministic and scattered, explicitly logged.
            bg=~(masks['left']|masks['right']); coords=np.flatnonzero(bg)
            n=int(masks[meta['answer_side']].sum())
            rng=np.random.default_rng(0); chosen=rng.choice(coords,min(n,len(coords)),replace=False)
            control=np.zeros(bg.size,dtype=bool);control[chosen]=True
            variants['background']=control.reshape(bg.shape)
            result['occlusion']={}
            for name,mask in variants.items():
                modified=np.array(image);modified[mask]=127
                score,_,_=probe.forward(Image.fromarray(modified),meta['instruction'])
                result['occlusion'][name]=dict(scores=score['scores'],masked_pixels=int(mask.sum()))
            result['background_control']='scattered equal-area gray pixels; topology differs from tile masks'
        result['elapsed_seconds']=time.time()-begin
        temp=dest/'result.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(dest/'result.json')
        print(json.dumps(dict(index=index+1,total=len(files),case_id=meta['case_id'],correct=result['correct'],
                              candidate_correct=result['candidate_correct'],seconds=result['elapsed_seconds'])),flush=True)

if __name__=='__main__':main()
