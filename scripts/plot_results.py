"""Scientific plots from completed results; no model rerun or metric changes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',default='outputs/attention')
    p.add_argument('--scenes',default='outputs/scenes')
    a=p.parse_args();root=Path(a.results);out=root/'figures';out.mkdir(exist_ok=True)
    summary=json.loads((root/'summary.json').read_text());cats=summary['categories']
    names=[r['category'] for r in cats];y=np.arange(len(names))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    ax=axes[0]
    values=np.array([r['accuracy'] for r in cats])*100
    lo=np.array([r['accuracy_ci_low'] for r in cats])*100
    hi=np.array([r['accuracy_ci_high'] for r in cats])*100
    ax.barh(y,values,color='#3478b9',alpha=.85)
    ax.errorbar(values,y,xerr=np.stack([values-lo,hi-values]),fmt='none',ecolor='#182b41',capsize=3)
    ax.set(yticks=y,yticklabels=[f"{n} (n={r['n']})" for n,r in zip(names,cats)],xlim=(0,103),xlabel='Selection accuracy (%)',title='Free-response accuracy · pair-bootstrap 95% CI')
    ax.invert_yaxis();ax.axvline(50,color='.6',ls='--',lw=1)
    ax=axes[1]
    width=.25
    for j,(key,label,color) in enumerate([
        ('occlusion_target_margin_drop','Target tile','#e88442'),
        ('occlusion_distractor_margin_drop','Distractor tile','#3478b9'),
        ('occlusion_background_margin_drop','Background pixels','#929ca6')]):
        ax.barh(y+(j-1)*width,[r.get(key,float('nan')) for r in cats],height=width,label=label,color=color)
    ax.set(yticks=y,yticklabels=names,xlabel='Mean reduction in target-vs-distractor log-probability margin',title='Gray occlusion diagnostic (no causal claim)')
    ax.invert_yaxis();ax.axvline(0,color='.4',lw=1);ax.legend(loc='best',fontsize=9)
    fig.suptitle(f"Magma explicit-object grounding · {summary['completed']} static scenes",fontsize=15)
    fig.savefig(out/'accuracy_occlusion.png',dpi=180);fig.savefig(out/'accuracy_occlusion.pdf');plt.close(fig)
    with (root/'layers.csv').open(encoding='utf-8-sig') as f: layers=list(csv.DictReader(f))
    differences=np.array([[float(r['target_density'])-float(r['distractor_density']) for r in layers if r['category']==name] for name in names])
    vmax=max(float(np.abs(differences).max()),1e-12)
    fig,ax=plt.subplots(figsize=(13,5),layout='constrained')
    im=ax.imshow(differences,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    ax.set(yticks=y,yticklabels=names,xticks=np.arange(0,differences.shape[1],3),xticklabels=np.arange(1,differences.shape[1]+1,3),xlabel='Language-model layer',title='Target minus distractor attention density · last prompt position')
    fig.colorbar(im,ax=ax,label='Attention probability per occupied visual cell')
    fig.savefig(out/'layer_attention_density.png',dpi=180);fig.savefig(out/'layer_attention_density.pdf');plt.close(fig)
    # One example per category, selected by scene ID rather than correctness.
    chosen={}
    for f in sorted(root.glob('*/result.json')):
        r=json.loads(f.read_text());chosen.setdefault(r['category'],(f,r))
    for category,(f,r) in chosen.items():
        scene=Path(a.scenes)/r['scene_id'];image=Image.open(scene/'rgb.png')
        maps=np.load(f.parent/'attention.npz')['head_maps'].astype('float32').mean(1)
        selected=(0,7,15,23,31)
        fig,axes=plt.subplots(2,5,figsize=(16,6),layout='constrained')
        rawmax=max(float(maps[list(selected)].max()),1e-12)
        for j,layer in enumerate(selected):
            m=maps[layer];relative=m/max(float(m.sum()),1e-12)
            for row,values,label,limit in ((0,m,'raw attention',rawmax),(1,relative,'spatial distribution',max(float(relative.max()),1e-12))):
                ax=axes[row,j];ax.imshow(image)
                im=ax.imshow(values,extent=(0,image.width,image.height,0),cmap='magma',alpha=.65,vmin=0,vmax=limit,interpolation='nearest')
                for side,color in [('left','cyan'),('right','lime')]:
                    mask=np.array(Image.open(scene/f'{side}_mask.png'))>0
                    ax.contour(mask,levels=[.5],colors=[color],linewidths=.6)
                ax.set_title(f'Layer {layer+1} · {label}',fontsize=10);ax.axis('off')
                fig.colorbar(im,ax=ax,fraction=.03,pad=.01)
        fig.suptitle(f"{category} | target={r['answer_side']} | response={r['raw_response']!r}\n{r['instruction']}\nTop: shared raw scale. Bottom: per-layer visual normalization and scale. Tile outlines: left cyan, right green.",fontsize=11)
        fig.savefig(out/f'example_{category}.png',dpi=160);plt.close(fig)
    print(f'Saved overview, all-layer heatmap and {len(chosen)} fixed-selection examples to {out}')

if __name__=='__main__':main()
