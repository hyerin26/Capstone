"""Frozen follow-up: case-aware scores, generated occlusion answers, layout/label controls."""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
from run_attention import fix_layerscale

REVISION='4159b1dedade8f88b5bbaaae7810299dee5228f5'


def load_model(checkpoint,device):
    import open_clip.pretrained
    from transformers.utils import logging
    logging.set_verbosity_error()
    original=open_clip.pretrained.download_pretrained_from_hf
    def architecture_only(model_id,*args,**kwargs):
        if model_id=='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg' and not args and kwargs.get('filename') is None:return None
        return original(model_id,*args,**kwargs)
    open_clip.pretrained.download_pretrained_from_hf=architecture_only
    config=AutoConfig.from_pretrained(checkpoint,trust_remote_code=True,code_revision=REVISION)
    config._attn_implementation='eager';config.text_config._attn_implementation='eager'
    try:
        model,loading=AutoModelForCausalLM.from_pretrained(checkpoint,config=config,trust_remote_code=True,code_revision=REVISION,torch_dtype=torch.bfloat16,attn_implementation='eager',device_map={'':device},output_loading_info=True)
    finally:open_clip.pretrained.download_pretrained_from_hf=original
    restored=fix_layerscale(model,checkpoint,loading);model.eval()
    processor=AutoProcessor.from_pretrained(checkpoint,trust_remote_code=True,code_revision=REVISION)
    return model,processor,dict(loading=loading,restored=restored,config=config.to_dict(),revision=REVISION)


def parse_response(raw,mode,mapping):
    text=raw.strip().strip('.!').strip().upper()
    if mode=='lr':return text.lower() if text in ('LEFT','RIGHT') else None
    if text not in ('A','B'):return None
    return ('left','right')[mapping.index(text)]


def logsumexp(values):
    m=max(values);return m+math.log(sum(math.exp(v-m) for v in values))


def score_sides(logprobs,mode,mapping):
    if mode=='lr':
        groups={'uppercase':('LEFT','RIGHT'),'titlecase':('Left','Right'),'lowercase':('left','right')}
    else:
        groups={'uppercase':tuple(mapping),'lowercase':tuple(mapping.lower())}
    scores={name:dict(zip(('left','right'),(logprobs[s] for s in labels))) for name,labels in groups.items()}
    scores['case_marginal']={side:logsumexp([scores[name][side] for name in groups]) for side in ('left','right')}
    return scores


@torch.inference_mode()
def evaluate(model,processor,image,instruction,mode,mapping,validate=False):
    marker='<image_start><image><image_end>' if model.config.mm_use_image_start_end else '<image>'
    ending="Answer with only LEFT or RIGHT, using the viewer's perspective." if mode=='lr' else 'Answer with only A or B, using the labels shown in the image.'
    question='Which tile is the target of the following instruction?\n'+instruction+'\n'+ending
    messages=[dict(role='system',content='You are agent that can see, talk and act.'),dict(role='user',content=marker+'\n'+question)]
    prompt=processor.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    x=processor(images=[image],texts=prompt,return_tensors='pt')
    x['pixel_values']=x['pixel_values'].unsqueeze(0);x['image_sizes']=x['image_sizes'].unsqueeze(0)
    device=model.get_input_embeddings().weight.device
    x={k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in x.items()}
    labels=('LEFT','RIGHT','Left','Right','left','right') if mode=='lr' else ('A','B','a','b')
    ids={s:processor.tokenizer.encode(s,add_special_tokens=False) for s in labels}
    assert all(len(v)==1 for v in ids.values()) and len({v[0] for v in ids.values()})==len(labels)
    generated=model.generate(**x,max_new_tokens=8,do_sample=False,use_cache=True,output_attentions=False,pad_token_id=processor.tokenizer.eos_token_id,return_dict_in_generate=True,output_scores=True)
    first=generated.scores[0][0].float()
    if not torch.isfinite(first).all():raise RuntimeError('Nonfinite generation scores')
    lp=first.log_softmax(-1);probs={s:float(lp[v[0]]) for s,v in ids.items()}
    output_ids=generated.sequences[0,x['input_ids'].shape[1]:].cpu().tolist()
    raw=processor.tokenizer.decode(output_ids,skip_special_tokens=True)
    predicted=parse_response(raw,mode,mapping)
    scores=score_sides(probs,mode,mapping)
    result=dict(raw_response=raw,predicted_side=predicted,valid=predicted is not None,model_prompt=prompt,
        input_ids=x['input_ids'][0].cpu().tolist(),output_ids=output_ids,image_sizes=x['image_sizes'].cpu().tolist(),pixel_values_shape=list(x['pixel_values'].shape),
        token_ids={s:v[0] for s,v in ids.items()},first_token_logprobs=probs,scores=scores,
        score_predictions={name:max(values,key=values.get) for name,values in scores.items()},
        restricted_label_mass=sum(math.exp(v) for v in probs.values()))
    if validate:
        logits=model(**x,use_cache=False,return_dict=True).logits[0,-1].float()
        delta=float((logits-first).abs().max())
        if delta>1e-4 or int(logits.argmax())!=output_ids[0]:raise RuntimeError(f'Generation scores differ from raw forward: {delta}')
        result['score_validation_max_abs_difference']=delta
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='weights/magma');p.add_argument('--inputs',default='outputs/followup_inputs');p.add_argument('--scenes',default='outputs/scenes');p.add_argument('--previous',default='outputs/attention');p.add_argument('--output',default='outputs/followup');p.add_argument('--device',default='cuda:0');p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    files=sorted(Path(a.inputs).glob('*/metadata.json'))
    if a.pilot:
        bycat={}
        for f in files:bycat.setdefault(json.loads(f.read_text())['category'],f)
        files=list(bycat.values())
    files=[f for i,f in enumerate(files) if i%a.workers==a.worker]
    model,processor,runtime=load_model(a.checkpoint,a.device)
    source=Path(__file__).read_bytes()+Path(__file__).resolve().parents[1].joinpath('run_attention.py').read_bytes()
    signature=hashlib.sha256(source+json.dumps(runtime['config'],sort_keys=True,default=str).encode()+Path(a.inputs).joinpath('manifest.json').read_bytes()).hexdigest()
    runtime.update(device=a.device,torch=torch.__version__,method_signature=signature,worker=a.worker)
    (out/f'runtime_{a.worker}.json').write_text(json.dumps(runtime,indent=2,default=str))
    start=time.time()
    for index,f in enumerate(files):
        meta=json.loads(f.read_text());dest=out/meta['scene_id'];dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():
            old=json.loads((dest/'result.json').read_text());assert old['method_signature']==signature;assert old['source']['image_sha256']==meta['image_sha256'];continue
        begin=time.time();records=[];images={}
        for name,sha in meta['image_sha256'].items():
            path=f.parent/f'{name}.png';assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
            images[name]=Image.open(path).convert('RGB')
        variants=[('original_lr','original','lr','', 'none'),('equal_lr','equal','lr','','none')]
        for layout in ('original','equal'):
            for mapping in ('AB','BA'):
                for mode in ('lr','ab'):
                    condition=f'{layout}_marked_lr' if mode=='lr' else f'{layout}_ab'
                    variants.append((condition,f'{layout}_{mapping}',mode,mapping,'none'))
        masks={s:np.array(Image.open(Path(a.scenes)/meta['scene_id']/f'{s}_mask.png'))>0 for s in ('left','right')}
        bg=~(masks['left']|masks['right']);coords=np.flatnonzero(bg);n=int(masks[meta['answer_side']].sum())
        chosen=np.random.default_rng(0).choice(coords,min(n,len(coords)),replace=False)
        control=np.zeros(bg.size,dtype=bool);control[chosen]=True;masks['background']=control.reshape(bg.shape)
        for name,mask in masks.items():
            pixels=np.array(images['original']);pixels[mask]=127
            images[f'occluded_{name}']=Image.fromarray(pixels);variants.append(('original_lr',f'occluded_{name}','lr','',name))
        for condition,image_name,mode,mapping,intervention in variants:
            r=evaluate(model,processor,images[image_name],meta['instruction'],mode,mapping,validate=a.pilot and image_name=='original')
            r.update(condition=condition,mapping=mapping,intervention=intervention,correct=r['predicted_side']==meta['answer_side'],
                     image_name=image_name,image_rgb_sha256=hashlib.sha256(images[image_name].tobytes()).hexdigest())
            if intervention!='none':r['masked_pixels']=int(masks[intervention].sum())
            records.append(r)
        previous=json.loads((Path(a.previous)/meta['scene_id']/'result.json').read_text());base=records[0]
        agreement=base['raw_response']==previous['raw_response']
        baseline_delta=max(abs(base['scores']['uppercase'][s]-previous['scores'][s.upper()]) for s in ('left','right'))
        if a.pilot and (not agreement or baseline_delta>.1):raise RuntimeError(f'Baseline did not reproduce: {meta["case_id"]}, {baseline_delta}')
        result=dict(source=meta,records=records,method_signature=signature,baseline_raw_agreement=agreement,baseline_uppercase_max_difference=baseline_delta,elapsed_seconds=time.time()-begin)
        tmp=dest/'result.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(dest/'result.json')
        status=dict(worker=a.worker,completed=index+1,total=len(files),generations=(index+1)*13,elapsed_seconds=time.time()-start,last_case=meta['case_id'],baseline_agreement=agreement,peak_gpu_mb=torch.cuda.max_memory_allocated(a.device)/1024**2)
        (out/f'status_{a.worker}.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    (out/f'complete_{a.worker}.json').write_text(json.dumps(dict(complete=True,cases=len(files),method_signature=signature)))

if __name__=='__main__':main()
