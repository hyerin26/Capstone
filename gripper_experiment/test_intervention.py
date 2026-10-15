"""Check intervention rows, image-key boundaries and softmax odds analytically."""
import numpy as np
import torch
from policy import Policy
p=Policy.__new__(Policy);p.enabled=True;p.strength=1.;p.start=1;p.length=2;p.query_start=3;p.penalty=np.array([1.,0.],dtype='float32')
mask=torch.zeros((1,1,4,5));hidden=torch.zeros((1,4,8))
_,kw=p.before(None,(),dict(hidden_states=hidden,attention_mask=mask,cache_position=torch.arange(4)))
actual=kw['attention_mask'];assert actual[0,0,3,1]==-1
expected=torch.zeros_like(actual);expected[0,0,3,1]=-1
assert torch.equal(actual,expected) and torch.equal(mask,torch.zeros_like(mask))
probs=actual[0,0,3].softmax(-1)
assert torch.allclose(probs[1]/probs[0],torch.exp(torch.tensor(-1.)))
p.strength=0
_,kw=p.before(None,(),dict(hidden_states=hidden,attention_mask=mask,cache_position=torch.arange(4)))
assert torch.equal(kw['attention_mask'],mask)
p.strength=2
_,kw=p.before(None,(),dict(hidden_states=hidden[:,:1],attention_mask=mask[:,:,:1],cache_position=torch.tensor([4])))
assert kw['attention_mask'][0,0,0,1]==-2
print('PASS: zero identity, last-prompt/generated query scope, visual-key scope, exp(-lambda) odds, input mask unchanged')
