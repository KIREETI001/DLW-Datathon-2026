import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
TARGETS=['transaction_amount','transactions_last_1h']
LOG=['transaction_amount','spend_last_24h','account_age','transactions_last_24h','transactions_last_1h']
CAT=['merchant_category','country','transaction_channel']
PARAMS={'loss':'quantile','learning_rate':.05,'max_iter':100,'max_leaf_nodes':7,'max_depth':3,'min_samples_leaf':150,'l2_regularization':10.,'max_bins':63,'early_stopping':False,'random_state':8603}
def target_values(X,target):
    a=pd.to_numeric(X[target],errors='coerce').to_numpy(float);a=np.where(np.isfinite(a)&(a>=0),a,np.nan);return np.log1p(a)
def features(X,target):
    F=X[CAT].copy()
    for col in LOG:
        if col!=target:F['log_'+col]=target_values(X,col)
    F['new_device']=pd.to_numeric(X.new_device,errors='coerce').replace([np.inf,-np.inf],np.nan)
    hr=pd.to_numeric(X.transaction_hour,errors='coerce').to_numpy(float);hr=np.where(np.isfinite(hr)&(hr>=0)&(hr<24),hr,np.nan)
    F['hour_sin']=np.sin(2*np.pi*hr/24);F['hour_cos']=np.cos(2*np.pi*hr/24)
    return F
def prep(F):
    return ColumnTransformer([('num',SimpleImputer(strategy='median',keep_empty_features=True),[c for c in F if c not in CAT]),
      ('cat',Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='__MISSING__',keep_empty_features=True)),('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CAT)])
def donor_fit(X,y,training,query,check):
    quantiles=np.empty((len(query),2,2));states=[]
    for j,target in enumerate(TARGETS):
        check();tv=target_values(X,target);donors=training[(y[training]==0)&np.isfinite(tv[training])];assert len(donors)>150 and not np.intersect1d(donors,query).size
        F=features(X,target);pre=prep(F);A=pre.fit_transform(F.iloc[donors]);V=pre.transform(F.iloc[query]);models=[]
        for k,q in enumerate([.5,.9]):
            check();model=HistGradientBoostingRegressor(quantile=q,**PARAMS);model.fit(A,tv[donors]);quantiles[:,j,k]=model.predict(V);models.append(model);check()
        states.append({'target':target,'donors':donors,'preprocess':pre,'models':models,'query':query.copy(),'training':training.copy(),'feature_columns':list(F.columns)})
    return quantiles,states
def crossfit(X,y,outer_train,outer_validation,check):
    qtrain=np.full((len(X),2,2),np.nan);records=[];coverage=np.zeros(len(X),int)
    for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=8603).split(X.iloc[outer_train],y[outer_train])):
        tr=outer_train[a];va=outer_train[b];assert not np.intersect1d(tr,outer_validation).size and not np.intersect1d(va,outer_validation).size
        q,states=donor_fit(X,y,tr,va,check);qtrain[va]=q;coverage[va]+=1;records.append({'inner_fold':k,'training':tr,'query':va,'states':states})
    assert np.all(coverage[outer_train]==1) and not coverage[outer_validation].any()
    qval,full=donor_fit(X,y,outer_train,outer_validation,check);assert np.isfinite(qtrain[outer_train]).all() and np.isfinite(qval).all()
    return qtrain[outer_train],qval,{'inner':records,'full':full,'outer_train':outer_train,'outer_validation':outer_validation}
def residual_features(X,q,kind):
    t=np.column_stack([target_values(X,col) for col in TARGETS]);assert q.shape==(len(X),2,2)
    if kind=='MEDIAN':return t-q[:,:,0]
    assert kind=='UPPER';return np.clip(np.maximum((t-q[:,:,1])/np.maximum(q[:,:,1]-q[:,:,0],.25),0),0,10)
def fit_basis(N,A):
    imp=SimpleImputer(strategy='median',keep_empty_features=True);Z=imp.fit_transform(N);D=np.column_stack([np.ones(len(A)),A]);projection=np.linalg.lstsq(D,Z,rcond=1e-12)[0]
    R=Z-D@projection;keep=R.std(0)>1e-8;scaler=StandardScaler().fit(R[:,keep]) if keep.any() else None
    return (scaler.transform(R[:,keep]) if scaler else np.empty((len(A),0))),{'imputer':imp,'projection':projection,'keep':keep,'scaler':scaler}
def transform_basis(N,A,b):
    if b['scaler'] is None:return np.empty((len(A),0))
    R=b['imputer'].transform(N)-np.column_stack([np.ones(len(A)),A])@b['projection'];return b['scaler'].transform(R[:,b['keep']])
