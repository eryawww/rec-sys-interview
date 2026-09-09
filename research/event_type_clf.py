"""(age,gender,region,title-unigrams,content_type,genre) -> binary mapping of event_type"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

ev=pd.read_csv('data/events.csv'); it=pd.read_csv('data/items.csv'); us=pd.read_csv('data/users.csv')
df=ev.merge(it,on='item_id').merge(us,on='user_id')
cv=CountVectorizer(lowercase=True)
X=np.hstack([pd.get_dummies(df[['age','gender','region','content_type','genre']].astype(str)).values.astype(float),
             cv.fit_transform(df.title).toarray()])
print(f'n={len(df)}  features={X.shape[1]} (incl. {len(cv.vocabulary_)} title unigrams)\n')

def run(y,groups=None,splitter=None):
    a=[];acc=[]
    sp = splitter.split(X,y,groups) if groups is not None else splitter.split(X)
    for tr,te in sp:
        if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2: continue
        m=HistGradientBoostingClassifier(random_state=0).fit(X[tr],y[tr])
        p=m.predict_proba(X[te])[:,1]
        a.append(roc_auc_score(y[te],p)); acc.append(accuracy_score(y[te],p>.5))
    return np.mean(a),np.std(a),np.mean(acc)

MAPS={
 'engaged {complete,like,save} vs rest':{'complete','like','save'},
 'positive {play,like,complete,save} vs {skip,pause}':{'play','like','complete','save'},
 'complete vs rest':{'complete'},
 'like vs rest':{'like'},
 'skip vs rest':{'skip'},
 'save vs rest':{'save'},
}
rows=[]
rng=np.random.default_rng(0)
for name,pos in MAPS.items():
    y=df.event_type.isin(pos).values.astype(int)
    p=y.mean(); maj=max(p,1-p)
    a,s,acc=run(y,splitter=KFold(5,shuffle=True,random_state=0))
    ag,_,_=run(y,groups=df.item_id.values,splitter=GroupKFold(5))
    au,_,_=run(y,groups=df.user_id.values,splitter=GroupKFold(5))
    ash,_,_=run(rng.permutation(y),splitter=KFold(5,shuffle=True,random_state=0))
    rows.append(dict(mapping=name,pos_rate=p,majority=maj,AUC=a,sd=s,acc=acc,
                     AUC_grp_item=ag,AUC_grp_user=au,AUC_shuffled=ash))
d=pd.DataFrame(rows); pd.set_option('display.width',250); pd.set_option('display.max_colwidth',55)
print(d.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
print('\nAUC 0.500 = chance. acc must beat `majority`.')
