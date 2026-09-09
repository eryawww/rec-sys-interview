"""Test the proposal: predict watch_seconds from denormalized features with a GBDT."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import CountVectorizer

ev=pd.read_csv('data/events.csv',parse_dates=['timestamp'])
it=pd.read_csv('data/items.csv'); us=pd.read_csv('data/users.csv')
df=ev.merge(it,on='item_id').merge(us,on='user_id')

# title unigrams
cv=CountVectorizer(lowercase=True); T=cv.fit_transform(df.title).toarray()
print('title unigram vocab size:',T.shape[1],'over',it.item_id.nunique(),'items')

def enc(cols):
    return pd.get_dummies(df[cols].astype(str)).values.astype(float)

base=dict(
  demo=enc(['age','gender','region']),
  item=enc(['content_type','genre']),
  etype=enc(['event_type']),
  time=np.c_[df.timestamp.dt.dayofweek, df.timestamp.dt.hour, df.timestamp.astype('int64')/1e9],
  title=T)
y=df.watch_seconds.values

def cvr2(X,y):
    s=[]
    for tr,te in KFold(5,shuffle=True,random_state=0).split(X):
        m=HistGradientBoostingRegressor(random_state=0).fit(X[tr],y[tr])
        s.append(r2_score(y[te],m.predict(X[te])))
    return np.mean(s),np.std(s)

sets={
 'event_type ONLY':['etype'],
 'everything EXCEPT event_type':['demo','item','time','title'],
 'full proposal (all features)':['demo','item','etype','time','title'],
}
print('\n5-fold CV R^2 predicting watch_seconds:')
for name,keys in sets.items():
    X=np.hstack([base[k] for k in keys])
    m,s=cvr2(X,y); print(f'  {name:32s} R2 = {m:+.4f} ± {s:.4f}   ({X.shape[1]} features)')

# serving-time reality: event_type is unknown for an unobserved pair. Fix it to 'play'.
play=df[df.event_type=='play']
Xp=np.hstack([pd.get_dummies(play[c].astype(str)).values.astype(float) for c in ['age','gender','region','content_type','genre']])
Xp=np.hstack([Xp, cv.transform(play.title).toarray()])
m,s=cvr2(Xp,play.watch_seconds.values)
print(f'\nServing scenario (event_type fixed to "play", n={len(play)}):')
print(f'  R2 = {m:+.4f} ± {s:.4f}   <- what the model can actually score an unseen pair with')
print(f'  std of watch_seconds within play = {play.watch_seconds.std():.0f}s (mean {play.watch_seconds.mean():.0f}s)')
