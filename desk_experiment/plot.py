"""Shared-scale image-conditional attention; all heads/layers, no outcome selection."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
image=Image.open(root/'desk.png').convert('RGB')
records=json.loads((root/'results/responses.json').read_text())
arrays=[]
for record in records:
    folder=root/'results'/record['object'].replace(' ','_')
    maps=np.load(folder/'attention.npz')['head_maps'].astype('float32')
    assert np.isfinite(maps).all() and maps.shape[:2]==(32,32)
    fractions=maps/np.maximum(maps.sum((-1,-2),keepdims=True),1e-12)
    arrays.append(fractions.mean((0,1)))
vmax=max(a.max() for a in arrays)
fig,axes=plt.subplots(2,3,figsize=(15,11),layout='constrained')
def overlay(ax,heat):
    ax.imshow(image)
    im=ax.imshow(heat,cmap='inferno',alpha=.65,vmin=0,vmax=vmax,extent=(-.5,image.width-.5,image.height-.5,-.5),interpolation='bilinear')
    ax.axis('off');return im
for ax,record,heat in zip(axes.flat,records,arrays):
    im=overlay(ax,heat);ax.set_title(record['instruction'])
    folder=root/'results'/record['object'].replace(' ','_')
    one,oa=plt.subplots(figsize=(8,8),layout='constrained');oi=overlay(oa,heat)
    oa.set_title(record['instruction']+'\nMean over 32 layers x 32 heads')
    one.colorbar(oi,ax=oa,shrink=.75,label='Image-conditional attention per visual cell')
    one.savefig(folder/'heatmap.png',dpi=160);plt.close(one)
fig.colorbar(im,ax=list(axes.flat),shrink=.8,label='Image-conditional attention per visual cell (shared scale)')
fig.suptitle('Magma: attention before the first response token\nSame image; six commands; all layers and heads averaged',fontsize=15)
fig.savefig(root/'results/overview.png',dpi=160);plt.close(fig)
for record in records:
    folder=root/'results'/record['object'].replace(' ','_')
    maps=np.load(folder/'attention.npz')['head_maps'].astype('float32')
    maps=maps/np.maximum(maps.sum((-1,-2),keepdims=True),1e-12)
    lm=maps.mean(1);lv=max(lm[i].max() for i in [0,7,15,23,31])
    fig,axes=plt.subplots(1,5,figsize=(20,5),layout='constrained')
    for ax,i in zip(axes,[0,7,15,23,31]):
        ax.imshow(image);im=ax.imshow(lm[i],cmap='inferno',alpha=.65,vmin=0,vmax=lv,extent=(-.5,image.width-.5,image.height-.5,-.5),interpolation='bilinear');ax.axis('off');ax.set_title('Layer '+str(i+1))
    fig.suptitle(record['instruction']+' — image-conditional attention (shared scale within panel)')
    fig.colorbar(im,ax=list(axes),shrink=.7)
    fig.savefig(folder/'layers.png',dpi=140);plt.close(fig)
print('Verified six finite 32 x 32 attention arrays; figures complete.')
