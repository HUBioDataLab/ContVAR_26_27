"""Four frozen-encoder decoder experiments: output strategy x pair processing."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS = {
    'parallel_difference': dict(strategy='parallel', processing='score_difference',
        title='Parallel + Score Difference', section='01_Parallel_ScoreDifference'),
    'parallel_concat': dict(strategy='parallel', processing='joint_concat',
        title='Parallel + Joint Concat', section='02_Parallel_JointConcat'),
    'three_class_difference': dict(strategy='three_class', processing='score_difference',
        title='Three-Class + Score Difference', section='03_ThreeClass_ScoreDifference'),
    'three_class_concat': dict(strategy='three_class', processing='joint_concat',
        title='Three-Class + Joint Concat', section='04_ThreeClass_JointConcat'),
}
FORMAT = 'contvar_factorial_decoder_v1'
DECISION = ('Two independent binary logits. Both <= 0 -> NEUTRAL; otherwise the larger logit wins. '
            'Positive LOF/GOF ties choose LOF. No validation/test threshold fitting.')


class SharedPairScorer(nn.Module):
    """The cascade's paired-scoring mechanism, versioned with this experiment.

    Keeping it here avoids changing an old cascade checkpoint implementation or
    depending on which historical scorer version happens to be on the server.
    """
    def __init__(self,protein_dim,go_dim,hidden_dims,dropout):
        super().__init__()
        dims=[protein_dim+go_dim,*hidden_dims]
        self.blocks=nn.ModuleList(nn.Sequential(nn.Linear(a,b),nn.LayerNorm(b),nn.SiLU())
                                  for a,b in zip(dims,dims[1:]))
        self.output=nn.Linear(dims[-1],1)
        self.dropout=dropout

    def forward(self,wt,var,go):
        left,right=torch.cat((wt,go),dim=-1),torch.cat((var,go),dim=-1)
        for block in self.blocks:
            left,right=block(left),block(right)
            if self.training and self.dropout:
                mask=F.dropout(torch.ones_like(left),self.dropout,training=True)
                left,right=left*mask,right*mask
        s_wt=self.output(left).squeeze(-1).sigmoid()
        s_var=self.output(right).squeeze(-1).sigmoid()
        return {'s_wt':s_wt,'s_var':s_var,'delta':s_wt-s_var}


class PairDecisionModel(nn.Module):
    def __init__(self, processing, protein_dim, go_dim, output_dim, projection_dim=128,
                 hidden_dims=(128,64), dropout=.2, normalization=None):
        super().__init__()
        if processing not in ('score_difference','joint_concat') or output_dim not in (1,3):
            raise ValueError('Unknown pair processing or output dimension')
        if min(protein_dim,go_dim,projection_dim,*hidden_dims)<1 or not hidden_dims or not 0<=dropout<1:
            raise ValueError('Invalid model dimensions/dropout')
        self.spec=dict(processing=processing,protein_dim=protein_dim,go_dim=go_dim,output_dim=output_dim,
                       projection_dim=projection_dim,hidden_dims=list(hidden_dims),dropout=dropout)
        self.processing,self.output_dim=processing,output_dim
        for name,dim,default in (('protein_mean',protein_dim,0),('protein_std',protein_dim,1),
                                ('go_mean',go_dim,0),('go_std',go_dim,1)):
            value=normalization[name].detach().cpu().clone() if normalization is not None else torch.full((dim,),float(default))
            self.register_buffer(name,value)
        # Identical WT/VAR transformations; no independent dropout before comparison.
        self.projection=nn.Sequential(nn.Linear(protein_dim,projection_dim),nn.LayerNorm(projection_dim),nn.SiLU())
        if processing=='score_difference':
            self.scorer=SharedPairScorer(projection_dim,go_dim,hidden_dims,dropout)
            self.raw_scale=nn.Parameter(torch.tensor(math.log(math.expm1(10.))))
            self.classifier=nn.Linear(1,output_dim)
        else:
            dim=2*projection_dim+go_dim; layers=[]
            for width in hidden_dims:
                layers.extend((nn.Linear(dim,width),nn.LayerNorm(width),nn.SiLU(),nn.Dropout(dropout)))
                dim=width
            self.classifier=nn.Sequential(*layers,nn.Linear(dim,output_dim))

    def representations(self,wt,var,go):
        if wt.ndim!=2 or wt.shape!=var.shape or go.ndim!=2 or len(go)!=len(wt):
            raise ValueError('Expected paired WT/VAR [B,D] and GO [B,G]')
        w=self.projection((wt-self.protein_mean)/self.protein_std)
        v=self.projection((var-self.protein_mean)/self.protein_std)
        g=(go-self.go_mean)/self.go_std
        return w,v,g

    def comparison(self,wt,var,go):
        if self.processing!='score_difference':
            raise ValueError('Joint Concat has no separate scalar score difference')
        return self.scorer(*self.representations(wt,var,go))

    def forward(self,wt,var,go):
        if self.processing=='score_difference':
            delta=self.comparison(wt,var,go)['delta']
            features=(delta*(F.softplus(self.raw_scale)+1e-4)).unsqueeze(-1)
        else:
            features=torch.cat(self.representations(wt,var,go),dim=-1)
        result=self.classifier(features)
        return result.squeeze(-1) if self.output_dim==1 else result


def parallel_decisions(logits):
    """Fixed OvR decision; returned three-class scores are NOT calibrated posteriors."""
    if logits.ndim!=2 or logits.shape[1]!=2 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite [B,2] independent LOF/GOF logits')
    joint=torch.cat((logits,torch.zeros_like(logits[:,:1])),dim=1)
    pred=joint.argmax(1)
    pred[(logits<=0).all(1)]=2
    return pred,joint.softmax(-1)


def binary_targets(labels,target_class):
    if target_class not in (0,1) or labels.ndim!=1 or not torch.isin(labels,labels.new_tensor([0,1,2])).all():
        raise ValueError('Expected LOF=0 / GOF=1 / NEUTRAL=2 labels')
    return labels.eq(target_class).float()


def load_model(path,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('format')!=FORMAT or checkpoint.get('labels')!=['LOF','GOF','NEUTRAL']:
        raise ValueError('Not a four-experiment decoder checkpoint')
    model=PairDecisionModel(**checkpoint['model_spec'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(),checkpoint
