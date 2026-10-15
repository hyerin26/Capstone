"""Magma action policy with real pre-softmax visual-key intervention."""
import sys,json,hashlib,time,os,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from transformers import AutoConfig,AutoProcessor,AutoModelForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_attention import fix_layerscale

class Policy:
    def __init__(self,checkpoint):
        a=SimpleNamespace(checkpoint=checkpoint,revision='4159b1dedade8f88b5bbaaae7810299dee5228f5')
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
        self.model=model;self.processor=processor;self.strength=0.;self.regions={};self.records=[]
        self.loading=loading;self.enabled=True;self.verified=False
        self.model.multi_modal_projector.register_forward_hook(self.projected)
        for i,layer in enumerate(model.language_model.model.layers):
            layer.self_attn.register_forward_pre_hook(self.before,with_kwargs=True)
            layer.self_attn.register_forward_hook(self.after(i),with_kwargs=True)
        self.stats={'q01':[-.02872725307941437,-.04170349963009357,-.026093858778476715,-.08092105075716972,-.09288699507713317,-.20718276381492615,0.],
                    'q99':[.028309678435325586,.040855254605412394,.040161586627364146,.08192047759890528,.07792850524187081,.20382574498653397,1.]}

    def projected(self,module,args,output):
        nc,ph,pw,_=output.shape
        gh,gw=self.crop_grid;assert nc==gh*gw
        self.h,self.w=gh*ph,gw*pw
        self.length=self.h*(self.w+1);self.query_start=self.input_len-1+self.length-1
        self.occs={}
        for name,mask in self.regions.items():
            # Native wrapper resizes source RGB to square 256 before the processor.
            square=Image.fromarray(mask.astype('float32')).resize((256,256),Image.Resampling.NEAREST)
            occ=np.asarray(square.resize((self.w,self.h),Image.Resampling.BOX)).copy()
            self.occs[name]=occ
        grid=np.zeros((self.h,self.w+1),dtype='float32')
        if self.region in self.occs:grid[:,:self.w]=self.occs[self.region]
        self.penalty=grid.reshape(-1)

    def before(self,module,args,kw):
        if not self.enabled:return args,kw
        hidden=kw.get('hidden_states',args[0] if args else None)
        mask=kw.get('attention_mask')
        assert mask is not None and mask.ndim==4,'Eager additive causal mask required'
        if self.strength:
            mask=mask.clone()
            positions=kw['cache_position']
            selected=(positions>=self.query_start).to(mask.dtype)
            bias=torch.as_tensor(self.penalty,device=mask.device,dtype=mask.dtype)*self.strength
            mask[:,:,:,self.start:self.start+self.length]-=selected.reshape(1,1,-1,1)*bias.reshape(1,1,1,-1)
            kw['attention_mask']=mask
        kw['output_attentions']=True
        return args,kw

    def after(self,index):
        def hook(module,args,kw,output):
            if not self.enabled:return output
            row=output[1][0,:,-1,:].detach().float()
            assert torch.isfinite(row).all()
            maps=row[:,self.start:self.start+self.length].reshape(row.shape[0],self.h,self.w+1)[...,:self.w]
            self.layer_maps[index]=maps.cpu().numpy()
            if index==len(self.model.language_model.model.layers)-1:
                arr=np.stack([self.layer_maps[i] for i in range(index+1)])
                self.records.append(arr)
            return output[0],None,*output[2:]
        return hook

    @torch.inference_mode()
    def act(self,rgb,masks,instruction,region,strength,max_new_tokens=128):
        self.region=region;self.strength=strength;self.regions=masks;self.records=[];self.layer_maps={}
        marker='<image_start><image><image_end>' if self.model.config.mm_use_image_start_end else '<image>'
        # Use canonical native step_one system/user ordering, action prompt and image resize.
        messages=[{'role':'system','content':'You are agent that can see, talk and act.'},
                  {'role':'user','content':marker+'\nWhat action should the robot take to '+instruction+'?'}]
        prompt=self.processor.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        image=Image.fromarray(rgb).resize((256,256))
        x=self.processor(images=[image],texts=prompt,return_tensors='pt')
        x['pixel_values']=x['pixel_values'].unsqueeze(0);x['image_sizes']=x['image_sizes'].unsqueeze(0)
        device=self.model.get_input_embeddings().weight.device
        x={k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in x.items()}
        self.input_len=x['input_ids'].shape[1];self.crop_grid=tuple(map(int,x['image_sizes'][0,0].tolist()))
        pos=(x['input_ids'][0]==self.model.config.image_token_index).nonzero().flatten();assert len(pos)==1
        self.start=int(pos[0])
        ids=self.model.generate(**x,do_sample=False,max_new_tokens=max_new_tokens,use_cache=True,output_attentions=False,pad_token_id=self.processor.tokenizer.eos_token_id)
        if not self.verified and strength==0:
            self.enabled=False
            plain=self.model.generate(**x,do_sample=False,max_new_tokens=max_new_tokens,use_cache=True,output_attentions=False,pad_token_id=self.processor.tokenizer.eos_token_id)
            self.enabled=True
            assert torch.equal(ids,plain),'Zero-intervention hooks changed generated sequence'
            self.verified=True
        gen=ids[0,self.input_len:].cpu().tolist();eos=self.processor.tokenizer.eos_token_id
        has_eos=eos in gen;e=gen.index(eos) if has_eos else len(gen)
        action_ids=gen[max(0,e-7):e]
        vocab=self.processor.tokenizer.vocab_size
        valid=has_eos and len(action_ids)==7 and all(vocab-256<=v<vocab for v in action_ids)
        action=None
        if valid:
            bins=(np.linspace(-1,1,256)[:-1]+np.linspace(-1,1,256)[1:])/2
            norm=bins[np.clip(vocab-np.asarray(action_ids)-1,0,254)]
            low=np.asarray(self.stats['q01']);high=np.asarray(self.stats['q99'])
            raw=.5*(norm+1)*(high-low)+low;raw[6]=norm[6]
            action=raw.copy();action[6]=1. if raw[6]>.5 else -1.
        # First row predicts first generated token; last seven rows before EOS predict actions.
        stack=np.stack(self.records)
        action_maps=stack[max(0,e-7):e] if valid else stack[:1]
        fractions={}
        total=action_maps.sum((-1,-2))
        for name,occ in self.occs.items():
            fractions[name]=float(((action_maps*occ).sum((-1,-2))/np.maximum(total,1e-12)).mean())
        result=dict(zero_hook_sequence_validated=self.verified,valid_action=valid,action=None if action is None else action.tolist(),generated_ids=gen,action_ids=action_ids,
                    raw_response=self.processor.tokenizer.decode(gen,skip_special_tokens=True),prompt=prompt,input_ids=x['input_ids'][0].cpu().tolist(),
                    region=region,strength=strength,grid=[self.h,self.w],region_fractions=fractions,
                    attention_scope='all seven action-token prediction queries' if valid else 'first generated query only; invalid action',
                    vocab_size=vocab,ended_with_eos=has_eos)
        # Save action-query mean per layer/head plus first query; compact, unmodified weights.
        return result,action_maps.mean(0).astype('float16'),stack[0].astype('float16')


def daemon():
    root=Path(__file__).resolve().parent;queue=root/'queue';queue.mkdir(exist_ok=True)
    policy=Policy(str(root.parent/'weights/magma'))
    (queue/'ready.json').write_text(json.dumps({'pid':os.getpid(),'loading':policy.loading},default=str))
    while not (queue/'STOP').exists():
        requests=sorted(queue.glob('*.request.json'))
        if not requests:time.sleep(.05);continue
        p=requests[0];request=json.loads(p.read_text());out=Path(request['output'])
        try:
            data=np.load(request['input']);masks={k:data[k] for k in data.files if k!='rgb'}
            result,maps,first=policy.act(data['rgb'],masks,request['instruction'],request['region'],request['strength'])
            out.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out.with_suffix('.npz'),action_query_mean=maps,first_query=first)
            temp=out.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(out)
        except Exception:
            out.with_suffix('.error').write_text(traceback.format_exc());traceback.print_exc()
        p.unlink()
if __name__=='__main__':daemon()
