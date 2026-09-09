import numpy as np, pandas as pd, math
from collections import defaultdict
rng = np.random.default_rng(0)

ev = pd.read_csv('data/events.csv', parse_dates=['timestamp'])
it = pd.read_csv('data/items.csv'); us = pd.read_csv('data/users.csv')

POS = {'play','like','complete','save'}
NEG = {'skip','pause'}
ev['pos'] = ev.event_type.isin(POS)

users = sorted(us.user_id); items = sorted(it.item_id)
uix = {u:i for i,u in enumerate(users)}; iix = {v:i for i,v in enumerate(items)}
nU, nI = len(users), len(items)
title = dict(zip(it.item_id, it.title)); genre = dict(zip(it.item_id, it.genre)); ctype = dict(zip(it.item_id, it.content_type))

# temporal leave-last-out on positive events
ev = ev.sort_values('timestamp')
pos = ev[ev.pos]
test = pos.groupby('user_id').tail(1)
train = ev.drop(test.index)

def build(df, weight):
    M = np.zeros((nU,nI))
    for u,i,w in zip(df.user_id, df.item_id, weight):
        M[uix[u], iix[i]] += w
    return M

tr_pos = train[train.pos]
Mbin = build(tr_pos, np.ones(len(tr_pos)))
Mbin = (Mbin>0).astype(float)
Mws  = build(tr_pos, tr_pos.watch_seconds.values)
Mlog = np.log1p(Mws)
seen = Mbin>0   # exclude from recommendation

def evaluate(score, name, K=10):
    s = score.copy()
    s[seen] = -np.inf
    hits=0; ndcg=0; r20=0; n=0; rec_items=set()
    order = np.argsort(-s, axis=1)
    for u,i in zip(test.user_id, test.item_id):
        ui, ii = uix[u], iix[i]
        if seen[ui,ii]: continue     # target already in train -> skip
        row = order[ui]
        rank = int(np.where(row==ii)[0][0])
        n+=1
        if rank < K: hits+=1; ndcg += 1/math.log2(rank+2)
        if rank < 20: r20+=1
        rec_items.update(row[:K].tolist())
    return dict(model=name, n=n, HR10=hits/n, NDCG10=ndcg/n, HR20=r20/n, cov=len(rec_items)/nI)

res=[]
# 1 random
res.append(evaluate(rng.random((nU,nI)), 'random'))
# 2 popularity by event count
res.append(evaluate(np.tile(Mbin.sum(0),(nU,1)), 'popularity_count'))
# 3 popularity by total watch_seconds
res.append(evaluate(np.tile(Mws.sum(0),(nU,1)), 'popularity_watchsec'))
# 3b popularity by completes only
cm = build(train[train.event_type=='complete'], np.ones((train.event_type=='complete').sum()))
res.append(evaluate(np.tile(cm.sum(0),(nU,1)), 'popularity_completes'))
# 3c popularity with recency decay (half-life 14d)
tmax = train.timestamp.max()
hl = 14*24*3600
w = 0.5**((tmax-tr_pos.timestamp).dt.total_seconds()/hl)
Mr = build(tr_pos, w.values)
res.append(evaluate(np.tile(Mr.sum(0),(nU,1)), 'popularity_recency'))

def norm(M):
    n = np.linalg.norm(M,axis=0); n[n==0]=1; return M/n
# 4 item-kNN cosine binary
for M,lab in [(Mbin,'bin'),(Mlog,'logws')]:
    X = norm(M); S = X.T@X; np.fill_diagonal(S,0)
    res.append(evaluate(M@S, f'itemknn_cosine_{lab}'))
    # top-50 neighbor truncation
    S2 = S.copy(); thr = np.sort(S2,axis=1)[:,-50][:,None]; S2[S2<thr]=0
    res.append(evaluate(M@S2, f'itemknn_top50_{lab}'))
# 5 user-kNN
Xu = Mbin/np.maximum(np.linalg.norm(Mbin,axis=1,keepdims=True),1e-9)
Su = Xu@Xu.T; np.fill_diagonal(Su,0)
res.append(evaluate(Su@Mbin, 'userknn_cosine'))
# 6 SVD
from sklearn.decomposition import TruncatedSVD
for f in [8,16,32,64]:
    sv = TruncatedSVD(n_components=f, random_state=0)
    Z = sv.fit_transform(Mbin); res.append(evaluate(Z@sv.components_, f'svd_{f}'))
# 7 implicit ALS
def als(C, f=32, reg=0.1, alpha=40, iters=15):
    P=(C>0).astype(float); Cf=1+alpha*C
    U=rng.normal(0,.01,(nU,f)); V=rng.normal(0,.01,(nI,f))
    for _ in range(iters):
        for (A,B,Pm,Cm) in ((U,V,P,Cf),(V,U,P.T,Cf.T)):
            BtB=B.T@B+reg*np.eye(f)
            for i in range(A.shape[0]):
                ci=Cm[i]; Bi=B*(ci-1)[:,None]
                A[i]=np.linalg.solve(BtB+B.T@Bi, (B*ci[:,None]).T@Pm[i])
    return U@V.T
res.append(evaluate(als(Mbin,32), 'als_implicit_f32'))
# 8 content-based: user profile over genre+content_type
feat = pd.get_dummies(it.set_index('item_id')[['genre','content_type']]).reindex(items).values.astype(float)
featn = feat/np.maximum(np.linalg.norm(feat,axis=1,keepdims=True),1e-9)
prof = Mbin@featn
profn = prof/np.maximum(np.linalg.norm(prof,axis=1,keepdims=True),1e-9)
res.append(evaluate(profn@featn.T, 'content_genre_ctype'))
# 9 demographic popularity (region)
uu = us.set_index('user_id').reindex(users)
for col in ['region','gender']:
    S=np.zeros((nU,nI))
    for g,grp in uu.groupby(col):
        rows=[uix[u] for u in grp.index]
        S[rows]=Mbin[rows].sum(0)
    res.append(evaluate(S, f'popularity_by_{col}'))
# 10 hybrid content*popularity
pop = Mbin.sum(0); popn = pop/pop.max()
res.append(evaluate(0.5*(profn@featn.T)+0.5*np.tile(popn,(nU,1)), 'hybrid_content_pop'))

df=pd.DataFrame(res).sort_values('NDCG10',ascending=False)
pd.set_option('display.width',200)
print(df.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
print('\nrandom-baseline HR10 expectation = %.4f'%(10/ (nI - Mbin.sum(1).mean())))
