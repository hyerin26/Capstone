"""Render figures from verified follow-up aggregates (no inference)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('outputs/followup/report')
s=json.loads((root/'summary.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
labels=['Original\nL/R','Equalized\nL/R','Original\n+ marks\nL/R','Equalized\n+ marks\nL/R','Original\nA/B','Equalized\nA/B']
x=np.arange(6);cs=s['conditions']
fig,ax=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
y=np.array([r['accuracy']*100 for r in cs]);lo=np.array([r['ci_low']*100 for r in cs]);hi=np.array([r['ci_high']*100 for r in cs])
ax[0].bar(x,y,color=['#74899b','#378fab','#74899b','#378fab','#74899b','#378fab'])
ax[0].errorbar(x,y,yerr=[y-lo,hi-y],fmt='none',color='black',capsize=3)
for i,v in enumerate(y):ax[0].text(i,hi[i]+1,f'{v:.1f}',ha='center')
ax[0].set(title='Actual generated-answer accuracy',ylabel='Accuracy (%)',ylim=(0,102),xticks=x,xticklabels=labels)
for key,label,color in [('left_target_accuracy','Target on left','#3977b8'),('right_target_accuracy','Target on right','#dd8835')]:
 ax[1].plot(x,[r[key]*100 for r in cs],'o-',label=label,color=color)
ax[1].set(title='Performance by target position',ylabel='Accuracy (%)',ylim=(0,102),xticks=x,xticklabels=labels);ax[1].legend(loc='lower right')
fig.suptitle('Magma: presentation and response controls\n500 scenes / 250 original pairs; 95% paired-bootstrap intervals')
for ext in ['png','pdf']:fig.savefig(root/f'layout_results.{ext}')
plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
base=[r for r in s['scoring_occlusion'] if r['intervention']=='none']
y=[r['score_accuracy']*100 for r in base]
ax[0].bar(range(4),y,color='#74899b');ax[0].axhline(74.2,color='#b63b3b',linestyle='--',label='Actual generation: 74.2%')
for i,v in enumerate(y):ax[0].text(i,v+1,f'{v:.1f}',ha='center')
ax[0].set(xticks=range(4),xticklabels=['UPPERCASE','Titlecase','lowercase','Case marginal'],ylim=(0,100),ylabel='Accuracy (%)',title='First-token scoring is spelling-sensitive');ax[0].legend(loc='lower left')
occ=[r for r in s['scoring_occlusion'] if r['scoring']=='titlecase']
y=[r['free_accuracy']*100 for r in occ];ax[1].bar(range(4),y,color=['#74899b','#b63b3b','#dd8835','#378fab'])
for i,v in enumerate(y):ax[1].text(i,v+1,f'{v:.1f}',ha='center')
ax[1].set(xticks=range(4),xticklabels=['None','Target tile','Distractor tile','Background'],ylim=(0,100),ylabel='Actual-answer accuracy (%)',title='Gray occlusion: newly generated answers')
for ext in ['png','pdf']:fig.savefig(root/f'scoring_occlusion.{ext}')
plt.close(fig)
cats=sorted({r['category'] for r in s['categories']});names=[r['condition'] for r in cs]
z=np.array([[next(r['accuracy']*100 for r in s['categories'] if r['category']==cat and r['condition']==cond) for cond in names] for cat in cats])
fig,ax=plt.subplots(figsize=(10,6),layout='constrained');im=ax.imshow(z,vmin=0,vmax=100,cmap='Blues')
for i in range(len(cats)):
 for j in range(6):ax.text(j,i,f'{z[i,j]:.1f}',ha='center',va='center',color='white' if z[i,j]>65 else 'black')
ax.set(xticks=range(6),xticklabels=labels,yticks=range(len(cats)),yticklabels=cats,title='Category accuracy (%) — repeated mappings averaged');fig.colorbar(im,ax=ax,shrink=.8)
for ext in ['png','pdf']:fig.savefig(root/f'category_results.{ext}')
