"""有限物理源分量的谱系划分。选择不改变全池证据，叶不是UAV位置。"""
from copy import deepcopy
import math
import numpy as np
from .branch_belief import gaussian_logpdf,logsumexp,normalized


class SpatialTree:
    def __init__(self,state,received_count=12):
        self.components=deepcopy(state['components']);self.logs=np.array(state['log_weights'],float)
        self.group_ids=sorted({c['group_id'] for c in self.components})
        self.nodes={'A':dict(id='A',parent_id=None,children=['R'],groups=self.group_ids.copy(),born_count=received_count,scored_count=0,label='persistent root',falsifier='not an environmental claim'),
                    'R':dict(id='R',parent_id='A',children=[],groups=self.group_ids.copy(),born_count=received_count,scored_count=0,label='unselected alternatives',falsifier='remaining physical candidates')}
        self.active=[];self.count=received_count;self.event=-1

    def leaves(self):return [k for k,n in self.nodes.items() if not n['children']]

    def indices(self,node):
        groups=set(self.nodes[node]['groups'])
        return [i for i,c in enumerate(self.components) if c['group_id'] in groups]

    def mass(self,node):return float(np.exp(self.logs[self.indices(node)]).sum()) if self.indices(node) else 0.

    def validate(self):
        leaves=self.leaves();groups=[g for k in leaves for g in self.nodes[k]['groups']]
        if sorted(groups)!=self.group_ids or len(groups)!=len(set(groups)):raise ValueError('leaves must partition all candidates')
        if len(self.nodes)>9 or len(leaves)>6 or len(self.active)>3 or len(set(self.active))!=len(self.active):raise ValueError('tree/active budget')
        if any(k not in leaves or k=='R' for k in self.active):raise ValueError('active must reference non-reserve leaves')
        for k,n in self.nodes.items():
            if n['children']:
                child_groups=[g for c in n['children'] for g in self.nodes[c]['groups']]
                if sorted(child_groups)!=sorted(n['groups']) or len(child_groups)!=len(set(child_groups)):raise ValueError('parent partition')
            depth=0;p=k
            while self.nodes[p]['parent_id'] is not None:depth+=1;p=self.nodes[p]['parent_id']
            if depth>3:raise ValueError('depth budget')
        if abs(logsumexp(self.logs))>1e-8:raise ValueError('unnormalized evidence')

    def initialize(self,hypotheses):
        if len(self.nodes)!=2 or not isinstance(hypotheses,list) or len(hypotheses)!=3:raise ValueError('three initialization branches once')
        if any(not isinstance(h['group_ids'],list) or not all(isinstance(g,str) for g in h['group_ids']) for h in hypotheses):raise ValueError('group ID lists required')
        groups=[g for h in hypotheses for g in h['group_ids']]
        if len(groups)!=len(set(groups)) or not set(groups)<=set(self.group_ids):raise ValueError('overlap or unknown group')
        for key,h in zip(('B','C','D'),hypotheses):
            if not h['group_ids']:raise ValueError('empty branch')
            self.nodes[key]=dict(id=key,parent_id='A',children=[],groups=sorted(h['group_ids']),born_count=self.count,scored_count=0,label=h['label'],falsifier=h['falsifier'])
        self.nodes['R']['groups']=sorted(set(self.group_ids)-set(groups));self.nodes['A']['children']=['B','C','D','R'];self.active=['B','C','D'];self.validate()

    def split(self,parent,groups,label,falsifier):
        if not isinstance(groups,list) or not all(isinstance(g,str) for g in groups) or parent not in self.leaves() or not set(groups)<set(self.nodes[parent]['groups']) or not groups or len(groups)!=len(set(groups)):raise ValueError('strict unique nonempty subset required')
        if len(self.nodes)+2>9:raise ValueError('node budget')
        depth=parent.count('.')+1
        if depth>=3:raise ValueError('depth budget')
        children=[parent+'.0',parent+'.1'];remaining=sorted(set(self.nodes[parent]['groups'])-set(groups))
        self.nodes[parent]['children']=children
        for key,subset,text in zip(children,(sorted(groups),remaining),(label,'complement of '+label)):
            self.nodes[key]=dict(id=key,parent_id=parent,children=[],groups=subset,born_count=self.count,scored_count=0,label=text,falsifier=falsifier)
        if parent in self.active:self.active.remove(parent)
        self.validate();return children

    def focus(self,ids):
        if not isinstance(ids,list) or not 1<=len(ids)<=3 or not all(isinstance(k,str) for k in ids):raise ValueError('one to three active leaf IDs')
        before=self.active;self.active=list(ids)
        try:self.validate()
        except Exception:self.active=before;raise

    def update(self,event_id,means,sigma,y):
        if event_id<=self.event or not math.isfinite(y) or len(means)!=len(self.logs):raise ValueError('ordered matching source readout')
        self.logs=normalized(self.logs+np.array([gaussian_logpdf(y,m,sigma) for m in means]));self.event=event_id;self.count+=1
        for k in self.leaves():self.nodes[k]['scored_count']+=1
        self.validate()

    def snapshot(self):
        return dict(nodes=[dict(deepcopy(n),mass=self.mass(k),is_leaf=k in self.leaves()) for k,n in self.nodes.items()],active=self.active.copy(),
                    received_count=self.count,last_event=self.event,component_log_weights=self.logs.tolist(),root_is_not_uav_position=True)

    def distribution(self,means,sigma,gp,full=False):
        indices=list(range(len(self.components))) if full else sorted({i for k in self.active for i in self.indices(k)})
        logs=self.logs[indices];mass=float(np.exp(logs).sum());mu=np.asarray(means)[indices].tolist();sd=[float(sigma)]*len(indices);weights=np.exp(logs).tolist()
        if not full and mass<1.:
            mu.append(float(gp['mean_dbm']));sd.append(float(gp['std_db']));weights.append(max(0.,1-mass))
        weights=np.array(weights);weights/=weights.sum()
        return dict(means=mu,stds=sd,weights=weights.tolist(),retained_source_mass=mass)


def parse(value,request_id,tree,initial=False):
    common={'request_id','action','rationale'}
    if not isinstance(value,dict) or value.get('request_id')!=request_id or not isinstance(value.get('rationale'),str) or not 1<=len(value['rationale'])<=3000:raise ValueError('identity/rationale')
    obj=deepcopy(tree)
    if initial:
        if set(value)!=common|{'hypotheses'} or value['action']!='initialize':raise ValueError('initialize schema')
        for h in value['hypotheses']:
            if set(h)!= {'group_ids','label','falsifier'} or not all(isinstance(h[k],str) and 1<=len(h[k])<=1200 for k in ('label','falsifier')):raise ValueError('hypothesis schema')
        obj.initialize(value['hypotheses'])
    else:
        if set(value)!=common|{'parent_id','group_ids','label','falsifier','active_ids'}:raise ValueError('refinement schema')
        if value['action']=='split':
            if not all(isinstance(value[k],str) and 1<=len(value[k])<=1200 for k in ('label','falsifier')):raise ValueError('labels')
            obj.split(value['parent_id'],value['group_ids'],value['label'],value['falsifier'])
        elif value['action']=='keep':
            if any(value[k] is not None for k in ('parent_id','group_ids','label','falsifier')):raise ValueError('keep null operands')
        else:raise ValueError('split/keep only')
        obj.focus(value['active_ids'])
    return obj


def numeric_initial(tree):
    regions={}
    for c in tree.components:
        if c['source_count']==1:regions.setdefault(c['sources'][0]['region_id'],set()).add(c['group_id'])
    ranked=sorted(regions,key=lambda r:(-sum(math.exp(tree.logs[i]) for i,c in enumerate(tree.components) if c['group_id'] in regions[r]),r))[:3]
    tree.initialize([dict(group_ids=sorted(regions[r]),label=r+' all single-source modes',falsifier='subsequent point likelihood') for r in ranked])


def numeric_refine(tree):
    eligible=[k for k in tree.leaves() if len(tree.nodes[k]['groups'])>1 and k.count('.')<2]
    if eligible:
        parent=sorted(eligible,key=lambda k:(-tree.mass(k),k))[0]
        groups=tree.nodes[parent]['groups'];best=sorted(groups,key=lambda g:(-sum(math.exp(tree.logs[i]) for i,c in enumerate(tree.components) if c['group_id']==g),g))[0]
        tree.split(parent,[best],'largest evidence atom','subsequent point likelihood')
    tree.focus(sorted((k for k in tree.leaves() if k!='R'),key=lambda k:(-tree.mass(k),k))[:3])
