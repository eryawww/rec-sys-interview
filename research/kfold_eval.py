"""5-fold CV (80/20) over positive interactions, all algorithms, paired stats."""
import numpy as np, pandas as pd, math
from sklearn.decomposition import TruncatedSVD

ev=pd.read_csv('data/events.csv',parse_dates=['timestamp'])
it=pd.read_csv('data/items.csv'); us=pd.read_csv('data/users.csv')
POS={'play','like','complete','save'}
pos=ev[ev.event_type.isin(POS)].drop_duplicates(['user_id','item_id']).reset_index(drop=True)
users=sorted(us.user_id); items=sorted(it.item_id)
uix={u:i for i,u in enumerate(users)}; iix={v:i for i,v in enumerate(items)}
nU,nI=len(users),len(items)
feat=pd.get_dummies(it.set_index('item_id')[['genre','content_type']]).reindex(items).values.astype(float)
featn=feat/np.maximum(np.linalg.norm(feat,axis=1,keepdims=True),1e-9)

def models(M,rng):
    out={}
    out['random']=rng.random((nU,nI))
    out['popularity']=np.tile(M.sum(0),(nU,1))
    X=M/np.maximum(np.linalg.norm(M,axis=0),1e-9); S=X.T@X; np.fill_diagonal(S,0)
    out['itemknn_full']=M@S
    S2=S.copy(); thr=np.sort(S2,1)[:,-50][:,None]; S2[S2<thr]=0
    out['itemknn_top50']=M@S2
    Xu=M/np.maximum(np.linalg.norm(M,axis=1,keepdims=True),1e-9); Su=Xu@Xu.T; np.fill_diagonal(Su,0)
    out['userknn']=Su@M
    for f in (16,64):
        sv=TruncatedSVD(n_components=f,random_state=0); Z=sv.fit_transform(M)
        out[f'svd_{f}']=Z@sv.components_
    p=M@featn; p=p/np.maximum(np.linalg.norm(p,axis=1,keepdims=True),1e-9)
    out['content']=p@featn.T
    pop=M.sum(0); out['hybrid']=0.5*out['content']+0.5*np.tile(pop/max(pop.max(),1),(nU,1))
    return out

rng=np.random.default_rng(0)
idx=rng.permutation(len(pos)); folds=np.array_split(idx,5)
hits={}; ndcg={}
for f,teidx in enumerate(folds):
    tr=pos.drop(pos.index[teidx]); te=pos.iloc[teidx]
    M=np.zeros((nU,nI))
    for u,i in zip(tr.user_id,tr.item_id): M[uix[u],iix[i]]=1
    seen=M>0
    for name,S in models(M,np.random.default_rng(100+f)).items():
        s=S.copy(); s[seen]=-np.inf; order=np.argsort(-s,1)
        rank_of=np.empty_like(order); 
        for r in range(nU): rank_of[r,order[r]]=np.arange(nI)
        h=[];g=[]
        for u,i in zip(te.user_id,te.item_id):
            ui,ii=uix[u],iix[i]
            if seen[ui,ii]: continue
            rk=rank_of[ui,ii]; h.append(rk<10); g.append(1/math.log2(rk+2) if rk<10 else 0.0)
        hits.setdefault(name,[]).extend(h); ndcg.setdefault(name,[]).extend(g)

rows=[]
base=np.array(hits['random'],float)
for name in hits:
    h=np.array(hits[name],float); g=np.array(ndcg[name])
    d=h-base
    bs=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(4000)])
    lo,hi=np.percentile(bs,[2.5,97.5])
    rows.append(dict(model=name,n=len(h),HR10=h.mean(),NDCG10=g.mean(),
                     vs_random=d.mean(),ci_lo=lo,ci_hi=hi,
                     sig='YES' if lo>0 else 'no'))
df=pd.DataFrame(rows).sort_values('NDCG10',ascending=False)
pd.set_option('display.width',200)
print(df.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
