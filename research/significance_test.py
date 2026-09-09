import numpy as np, pandas as pd, math
rng=np.random.default_rng(1)
ev=pd.read_csv('data/events.csv',parse_dates=['timestamp']);it=pd.read_csv('data/items.csv');us=pd.read_csv('data/users.csv')
POS={'play','like','complete','save'}; ev['pos']=ev.event_type.isin(POS)
users=sorted(us.user_id);items=sorted(it.item_id);uix={u:i for i,u in enumerate(users)};iix={v:i for i,v in enumerate(items)}
nU,nI=len(users),len(items)
def run(seed):
    r=np.random.default_rng(seed)
    pos=ev[ev.pos].sample(frac=1,random_state=seed)
    test=pos.groupby('user_id').tail(1); train=ev.drop(test.index); trp=train[train.pos]
    M=np.zeros((nU,nI))
    for u,i in zip(trp.user_id,trp.item_id): M[uix[u],iix[i]]=1
    seen=M>0
    def ev_(s):
        s=s.copy();s[seen]=-np.inf;o=np.argsort(-s,1);h=[]
        for u,i in zip(test.user_id,test.item_id):
            ui,ii=uix[u],iix[i]
            if seen[ui,ii]:continue
            h.append(int(np.where(o[ui]==ii)[0][0])<10)
        return np.array(h)
    out={}
    out['random']=ev_(r.random((nU,nI)))
    out['popularity']=ev_(np.tile(M.sum(0),(nU,1)))
    X=M/np.maximum(np.linalg.norm(M,axis=0),1e-9);S=X.T@X;np.fill_diagonal(S,0)
    S2=S.copy();thr=np.sort(S2,1)[:,-50][:,None];S2[S2<thr]=0
    out['itemknn']=ev_(M@S2)
    Xu=M/np.maximum(np.linalg.norm(M,axis=1,keepdims=True),1e-9);Su=Xu@Xu.T;np.fill_diagonal(Su,0)
    out['userknn']=ev_(Su@M)
    return out
agg={}
for s in range(10):
    for k,v in run(s).items(): agg.setdefault(k,[]).append(v.mean())
print('HR@10 over 10 random leave-one-out splits (mean ± sd):')
for k,v in agg.items():
    v=np.array(v);print(f'  {k:12s} {v.mean():.4f} ± {v.std():.4f}   [{v.min():.4f}, {v.max():.4f}]')
# paired bootstrap on one split
o=run(0); base=o['random']
for k in ['popularity','itemknn','userknn']:
    d=o[k].astype(float)-base.astype(float)
    bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(5000)]
    lo,hi=np.percentile(bs,[2.5,97.5])
    print(f'{k} - random: diff={d.mean():+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]  {"SIGNIFICANT" if lo>0 else "not significant"}')
