import pandas as pd, numpy as np
from collections import Counter,defaultdict
ev=pd.read_csv('data/events.csv',parse_dates=['timestamp']);it=pd.read_csv('data/items.csv').set_index('item_id')
ev=ev.sort_values(['user_id','timestamp'])
# genre transition matrix vs marginal
g=it.genre.to_dict()
trans=Counter();marg=Counter()
for u,grp in ev.groupby('user_id'):
    s=[g[i] for i in grp.item_id]
    for a,b in zip(s,s[1:]): trans[(a,b)]+=1
    for x in s: marg[x]+=1
N=sum(marg.values())
chi=0;tot=sum(trans.values())
for (a,b),c in trans.items():
    exp=tot*marg[a]/N*marg[b]/N
    chi+=(c-exp)**2/exp
dof=(len(marg)-1)**2
print(f'genre transition chi2={chi:.1f} dof={dof} ratio={chi/dof:.2f}')
# repeat-consumption rate
d=ev.duplicated(['user_id','item_id']).sum(); print('repeat interactions:',d,'of',len(ev), f'({d/len(ev):.1%})')
# session structure: gaps
gaps=[]
for u,grp in ev.groupby('user_id'):
    gaps+= list(grp.timestamp.diff().dt.total_seconds().dropna()/3600)
gaps=np.array(gaps); print('inter-event gap hours: median %.1f  p10 %.1f  frac<1h %.2f'%(np.median(gaps),np.percentile(gaps,10),(gaps<1).mean()))
# does user's past genre predict next genre?
hit=0;n=0
for u,grp in ev.groupby('user_id'):
    s=[g[i] for i in grp.item_id]
    for k in range(3,len(s)):
        top=Counter(s[:k]).most_common(1)[0][0]
        hit+= (top==s[k]); n+=1
base=max(marg.values())/N
print(f'next-genre predicted by user history mode: {hit/n:.3f}  vs global-mode baseline {base:.3f}')
