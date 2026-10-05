"""Small discrete NB/TAN, with exact tree marginalization. No file/data loading."""
import itertools
import numpy as np
import pandas as pd
from scipy.special import logsumexp

RAW = ['transaction_amount','transaction_hour','merchant_category','country','transaction_channel',
       'transactions_last_24h','spend_last_24h','account_age','new_device','transactions_last_1h']
CAT = {'merchant_category','country','transaction_channel'}
MENU = [('NB4',None,'marginal'),('TAN4-M20',20.,'marginal'),
        ('TAN4-M80',80.,'marginal'),('TAN4-S20',20.,'state')]

def fit_encoder(frame, mode):
    state=[]
    for name in RAW:
        s=frame[name]; item={'name':name,'missing_mode':mode}
        if name in CAT:
            levels=sorted(s.dropna().astype(str).unique().tolist())
            item.update(kind='category',levels=levels,value_states=max(1,len(levels)))
        elif name=='transaction_hour':item.update(kind='hour',value_states=4)
        elif name=='new_device':item.update(kind='binary',value_states=2)
        else:
            values=pd.to_numeric(s,errors='coerce').to_numpy(float)
            values=values[np.isfinite(values)]
            cuts=np.unique(np.quantile(values,[.25,.5,.75],method='linear')) if len(values) else np.array([])
            if len(values):cuts=cuts[cuts>values.min()]
            if len(values) and len(cuts):
                occupied=np.unique(np.searchsorted(cuts,values,side='right'))
                cuts=cuts[occupied[1:]-1]
                np.testing.assert_array_equal(np.unique(np.searchsorted(cuts,values,side='right')),np.arange(len(cuts)+1))
            item.update(kind='numeric',cuts=cuts,value_states=len(cuts)+1)
        item['states']=item['value_states']+(mode=='state');state.append(item)
    return state

def encode(frame,state):
    out=np.full((len(frame),len(state)),-1,dtype=np.int64)
    for i,item in enumerate(state):
        s=frame[item['name']];missing=s.isna().to_numpy();kind=item['kind']
        if kind=='category':
            mapping={v:j for j,v in enumerate(item['levels'])}
            out[:,i]=s.map(lambda v:mapping.get(str(v),-1) if not pd.isna(v) else -1).to_numpy(int)
        else:
            a=pd.to_numeric(s,errors='coerce').to_numpy(float);ok=np.isfinite(a)
            if kind=='numeric':out[ok,i]=np.searchsorted(item['cuts'],a[ok],side='right')
            elif kind=='hour':
                ok &= (a>=0)&(a<24);out[ok,i]=np.floor(((a[ok]+3)%24)/6).astype(int)
            else:
                ok &= (a==0)|(a==1);out[ok,i]=a[ok].astype(int)
        if item['missing_mode']=='state':out[missing,i]=item['value_states']
    return out

def marginals(x,y,ks):
    result=[]
    for i,k in enumerate(ks):
        ok=x[:,i]>=0
        count=np.bincount(y[ok]*k+x[ok,i],minlength=2*k).reshape(2,k)
        result.append((count+1./k)/(count.sum(1,keepdims=True)+1.))
    return result

def learn_tree(x,y,ks,m,prior):
    d=len(ks);edges=[];support={}
    for i in range(d):
        for j in range(i+1,d):
            ok=(x[:,i]>=0)&(x[:,j]>=0)
            n=np.bincount(y[ok]*ks[i]*ks[j]+x[ok,i]*ks[j]+x[ok,j],minlength=2*ks[i]*ks[j]).reshape(2,ks[i],ks[j])
            q=n+20.*m[i][:,:,None]*m[j][:,None,:]
            q/=q.sum((1,2),keepdims=True)
            information=(q*np.log(q/(q.sum(2,keepdims=True)*q.sum(1,keepdims=True)))).sum((1,2))
            score=float(np.dot(prior,information));edges.append((score,i,j))
            support[f'{i},{j}']={'pair_observed_class_counts':n.sum((1,2)).tolist(),'pair_table_counts':n,'weight':score}
    component=list(range(d));chosen=[]
    def find(a):
        while component[a]!=a:a=component[a]
        return a
    for weight,i,j in sorted(edges,key=lambda row:(-row[0],row[1],row[2])):
        a,b=find(i),find(j)
        if a!=b:component[b]=a;chosen.append((i,j))
        if len(chosen)==d-1:break
    adjacency=[[] for _ in range(d)]
    for i,j in chosen:adjacency[i].append(j);adjacency[j].append(i)
    parent=np.full(d,-2,int);parent[1]=-1;queue=[1]
    for i in queue:
        for j in sorted(adjacency[i]):
            if parent[j]==-2:parent[j]=i;queue.append(j)
    assert (parent!=-2).all()
    return parent,support

def postorder(parent):
    children=[[] for _ in parent]
    for i,p in enumerate(parent):
        if p>=0:children[p].append(i)
    order=[]
    def visit(i):
        for j in children[i]:visit(j)
        order.append(i)
    for i,p in enumerate(parent):
        if p<0:visit(i)
    assert len(order)==len(parent) and len(set(order))==len(parent)
    return order,children

def validate(model):
    k=model['ks'];parent=model['parent'];postorder(parent)
    assert len(k)==len(parent)==len(model['probability'])
    np.testing.assert_allclose(model['prior'].sum(),1.,rtol=0,atol=1e-14)
    assert np.all(model['prior']>0)
    for i,t in enumerate(model['probability']):
        assert t.shape==((2,k[i]) if parent[i]<0 else (2,k[parent[i]],k[i]))
        assert np.isfinite(t).all() and np.all(t>0)
        np.testing.assert_allclose(t.sum(-1),1.,rtol=0,atol=1e-13)

def log_evidence(model,x):
    """Exact evidence likelihood for arbitrary observed subsets and either class."""
    x=np.asarray(x,dtype=np.int64);ks=model['ks'];parent=model['parent']
    assert x.ndim==2 and x.shape[1]==len(ks)
    assert all(np.all((x[:,i]>=-1)&(x[:,i]<k)) for i,k in enumerate(ks))
    order,children=postorder(parent);answer=np.zeros((len(x),2))
    for c in range(2):
        messages={}
        for i in order:
            local=np.zeros((len(x),ks[i]));known=x[:,i]>=0
            local[known,:]=-np.inf;local[np.flatnonzero(known),x[known,i]]=0.
            for child in children[i]:local+=messages[child]
            t=np.log(model['probability'][i][c])
            if parent[i]<0:answer[:,c]+=logsumexp(local+t,axis=1)
            else:messages[i]=logsumexp(local[:,None,:]+t[None,:,:],axis=2)
    return answer

def predict_encoded(model,x):
    lp=log_evidence(model,x)+np.log(model['prior'])
    return np.exp(lp-logsumexp(lp,axis=1,keepdims=True))

def predict_proba(model,frame):
    return predict_encoded(model,encode(frame,model['encoder']))

def complete_log_likelihood(model,x,y):
    out=np.zeros(len(x))
    for i,p in enumerate(model['parent']):
        t=model['probability'][i]
        out+=np.log(t[y,x[:,i]] if p<0 else t[y,x[:,p],x[:,i]])
    return out

def expand_missing(x,ks,max_rows=100000):
    count=np.ones(len(x),dtype=int)
    for i,k in enumerate(ks):count*=np.where(x[:,i]<0,k,1)
    if count.sum()>max_rows:raise RuntimeError('Exact training completion resource cap exceeded')
    original=np.repeat(np.arange(len(x)),count);expanded=x[original].copy()
    start=np.r_[0,np.cumsum(count)[:-1]]
    for row in np.flatnonzero(count>1):
        missing=np.flatnonzero(x[row]<0)
        states=np.array(list(itertools.product(*(range(ks[j]) for j in missing))),int)
        expanded[start[row]:start[row]+count[row],missing]=states
    # A missing constant-state variable needs its single possible value too.
    expanded[expanded<0]=0
    return expanded,original,start

def e_step(model,expanded,original,start,y):
    lp=complete_log_likelihood(model,expanded,y[original])
    maximum=np.maximum.reduceat(lp,start)
    scaled=np.exp(lp-maximum[original]);normalizer=np.add.reduceat(scaled,start)
    weight=scaled/normalizer[original]
    likelihood=float(np.sum(maximum+np.log(normalizer)+np.log(model['prior'][y])))
    return weight,likelihood

def m_step(model,expanded,y,weight,tau,centers):
    tables=[];support=[]
    for i,p in enumerate(model['parent']):
        ki=model['ks'][i]
        if p<0:
            count=np.bincount(y*ki+expanded[:,i],weights=weight,minlength=2*ki).reshape(2,ki)
            table=(count+1./ki)/(count.sum(1,keepdims=True)+1.)
        else:
            kp=model['ks'][p]
            count=np.bincount(y*kp*ki+expanded[:,p]*ki+expanded[:,i],weights=weight,minlength=2*kp*ki).reshape(2,kp,ki)
            table=(count+tau*centers[i][:,None,:])/(count.sum(2,keepdims=True)+tau)
        tables.append(table);support.append(count)
    return tables,support

def regularizer(model,tau,centers):
    value=0.
    for i,p in enumerate(model['parent']):
        log=np.log(model['probability'][i])
        value+=float(np.sum(log/model['ks'][i])) if p<0 else float(np.sum(tau*centers[i][:,None,:]*log))
    return value

def fit_encoded(x,y,ks,structure,tau=20.,iterations=20):
    """Fit only supplied encoded data; callers must enforce outer isolation."""
    x=np.asarray(x,int);y=np.asarray(y,int);ks=np.asarray(ks,int)
    assert set(np.unique(y))=={0,1}
    prior=np.bincount(y,minlength=2)/len(y);centers=marginals(x,y,ks)
    parent,scores=(np.full(len(ks),-1,int),{}) if structure=='nb' else learn_tree(x,y,ks,centers,prior)
    model={'format':'tan_discrete_v1','positive_class':1,'ks':ks,'parent':parent,'prior':prior,
           'probability':[],'pair_scores':scores,'prior_centers':centers,'tau':tau,'trace':[]}
    model['observed_state_counts']=[]
    for i,k in enumerate(ks):
        ok=x[:,i]>=0
        model['observed_state_counts'].append(np.bincount(y[ok]*k+x[ok,i],minlength=2*k).reshape(2,k))
    for i,p in enumerate(parent):
        if p<0:model['probability'].append(centers[i].copy());continue
        ki,kp=ks[i],ks[p];ok=(x[:,i]>=0)&(x[:,p]>=0)
        count=np.bincount(y[ok]*kp*ki+x[ok,p]*ki+x[ok,i],minlength=2*kp*ki).reshape(2,kp,ki)
        model['probability'].append((count+tau*centers[i][:,None,:])/(count.sum(2,keepdims=True)+tau))
    if structure!='nb':
        expanded,original,start=expand_missing(x,ks)
        weights,ll=e_step(model,expanded,original,start,y)
        objective=ll+regularizer(model,tau,centers)
        model['trace'].append({'iteration':0,'log_likelihood':ll,'regularized_objective':objective})
        for step in range(iterations):
            tables,support=m_step(model,expanded,y[original],weights,tau,centers)
            delta=max(float(np.max(np.abs(a-b))) for a,b in zip(tables,model['probability']))
            model['probability']=tables
            weights,ll=e_step(model,expanded,original,start,y)
            new=ll+regularizer(model,tau,centers)
            assert new>=objective-1e-6,(step,objective,new)
            model['trace'].append({'iteration':step+1,'log_likelihood':ll,'regularized_objective':new,'max_parameter_change':delta})
            objective=new
        model['final_m_step_support']=support if iterations else None
        _,support=m_step(model,expanded,y[original],weights,tau,centers)
        model['expected_support']=support
        model['expanded_training_rows']=len(expanded)
    validate(model)
    return model

def fit(frame,y,spec):
    name,tau,mode=spec;encoder=fit_encoder(frame,mode);x=encode(frame,encoder)
    structure='nb' if name=='NB4' else 'tan'
    model=fit_encoded(x,y,[s['states'] for s in encoder],structure,tau or 20.,20 if mode=='marginal' and structure=='tan' else 0)
    model.update(name=name,encoder=encoder)
    return model
