"""Six imperative prompts; last-prompt-position attention, not robot execution."""
import sys, json, hashlib
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_attention import Probe, fix_layerscale
import numpy as np
from PIL import Image
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM

class CommandProbe(Probe):
    def prepare(self,image,instruction):
        marker='<image_start><image><image_end>' if self.model.config.mm_use_image_start_end else '<image>'
        messages=[{'role':'system','content':'You are agent that can see, talk and act.'},
                  {'role':'user','content':marker+'\n'+instruction}]
        prompt=self.processor.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        x=self.processor(images=[image],texts=prompt,return_tensors='pt')
        x['pixel_values']=x['pixel_values'].unsqueeze(0)
        x['image_sizes']=x['image_sizes'].unsqueeze(0)
        device=self.model.get_input_embeddings().weight.device
        return {k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in x.items()},prompt

def main():
    root=Path(__file__).resolve().parent
    out=root/'results';out.mkdir(exist_ok=True)
    a=SimpleNamespace(checkpoint=str(root.parent/'weights/magma'),revision='4159b1dedade8f88b5bbaaae7810299dee5228f5')
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
    probe=CommandProbe(model,processor)
    image=Image.open(root/'desk.png').convert('RGB')
    records=[]
    for obj in ['headphones','speaker','clock','eyeglasses','camera','mobile phone']:
        instruction='Pick up the '+obj+'.'
        data,maps,x=probe.forward(image,instruction,capture=True)
        normal,_,_=probe.forward(image,instruction,capture=False)
        assert normal['first_token_id']==data['first_token_id']
        assert all(abs(normal['scores'][k]-data['scores'][k])<.1 for k in normal['scores'])
        # LEFT/RIGHT scores inherited from the probe are irrelevant to this task.
        for key in ['scores','candidate_prediction']:data.pop(key,None)
        with torch.inference_mode():
            generated=model.generate(**x,max_new_tokens=64,do_sample=False,use_cache=True,output_attentions=False,pad_token_id=processor.tokenizer.eos_token_id)
        data.update(object=obj,instruction=instruction,raw_response=processor.tokenizer.decode(generated[0,x['input_ids'].shape[1]:],skip_special_tokens=True),hook_validated=True,revision=a.revision,image_sha256=hashlib.sha256((root/'desk.png').read_bytes()).hexdigest())
        name=obj.replace(' ','_');dest=out/name;dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/'attention.npz',head_maps=maps.astype('float16'))
        (dest/'result.json').write_text(json.dumps(data,indent=2))
        records.append(data)
        print(json.dumps({'object':obj,'response':data['raw_response'],'grid':data['grid']}),flush=True)
    (out/'responses.json').write_text(json.dumps(records,indent=2))
    (out/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,revision=a.revision,loading=loading,device_map=model.hf_device_map),indent=2,default=str))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
