"""Independent oracle for the removal audit.

Retrains one model per deleted training row with sklearn's own
StandardScaler + GaussianNB chain and compares flip counts with the fast
closed-form implementation in backend/src/services/removal_compute.py.

Usage:  python removal_oracle.py [path-to-backend]
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

BACKEND=Path(sys.argv[1] if len(sys.argv)>1 else __file__).resolve().parents[2]/'backend'
sys.path.insert(0,str(BACKEND))
from src.services.removal_compute import compute_fold  # noqa: E402


def retrain_counts(X,y,train,validation,base,smoothing):
    out=[]
    for p in range(len(train)):
        keep=np.ones(len(train),bool);keep[p]=False
        kept=train[keep]
        scaler=StandardScaler().fit(X[kept])
        model=GaussianNB(var_smoothing=smoothing).fit(scaler.transform(X[kept]),y[kept])
        out.append(int((model.predict(scaler.transform(X[validation]))!=base[validation]).sum()))
    return np.array(out)


def baseline(X,y,folds,smoothing):
    base=np.empty(len(y),int)
    for train,validation in folds:
        scaler=StandardScaler().fit(X[train])
        model=GaussianNB(var_smoothing=smoothing).fit(scaler.transform(X[train]),y[train])
        base[validation]=model.predict(scaler.transform(X[validation]))
    return base


def synth(n,classes,scale,means,seed,offset=(0,0,0),outlier_share=0.0):
    rng=np.random.default_rng(seed)
    y=np.array([i%classes for i in range(n)])
    X=np.array(means)[:classes][y]+rng.normal(0,scale,size=(n,len(scale)))+np.array(offset)
    if outlier_share:
        rows=rng.choice(n,int(n*outlier_share),replace=False)
        X[rows]+=rng.normal(0,np.array(scale)*6,size=(len(rows),len(scale)))
    return X,y


def check(name,X,y,classes,smoothing=1e-9,folds=3):
    split=list(StratifiedKFold(min(folds,classes*4),shuffle=True,random_state=42).split(X,y))
    base=baseline(X,y,split,smoothing)
    labels=list(range(classes));mismatch=0;retrained=0
    for train,validation in split:
        fast=compute_fold(X,y,labels,train,validation,base,smoothing)
        got=np.array([row['flips'] for row in fast['rows']])
        expected=retrain_counts(X,y,train,validation,base,smoothing)
        usable=np.array([not row['missing'] for row in fast['rows']])
        mismatch+=int((got[usable]!=expected[usable]).sum())
        retrained+=fast['retrained']
    print(f'{name:38s} bad={mismatch} retrained={retrained}')
    return mismatch


if __name__=='__main__':
    three=[[1.2,900,2.8],[3.0,1600,5.2],[2.0,2300,3.9]]
    four=three+[[1.8,2600,6.1]]
    two=[[1.2,900,2.8],[2.6,1700,4.4]]
    bad=0
    bad+=check('baseline 240x3',*synth(240,3,(.35,170,.45),three,2),3)
    bad+=check('720x3 demo scale',*synth(720,3,(.35,170,.45),three,4),3)
    bad+=check('720x3 5% outliers',*synth(720,3,(.35,170,.45),three,5,outlier_share=.05),3)
    bad+=check('centroid offset 1e6',*synth(240,3,(.35,170,.45),three,3,offset=(0,1e6,0)),3)
    bad+=check('4 classes smoothing 1e-3',*synth(240,4,(.35,170,.45),four,6),4,smoothing=1e-3)
    bad+=check('2 classes smoothing 1e-3',*synth(240,2,(.35,170,.45),two,7),2,smoothing=1e-3)
    bad+=check('smoothing 1e-4',*synth(240,3,(.35,170,.45),three,9),3,smoothing=1e-4)
    X,y=synth(240,3,(.35,170,.45),three,10);X[:,2]=7.5
    bad+=check('constant kurtosis column',X,y,3)
    for seed in range(40):  # randomised shapes: duplicates, rounding, rare classes
        rng=np.random.default_rng(seed)
        classes=int(rng.integers(2,5));n=int(rng.integers(30,220))
        scale=np.exp(rng.uniform(np.log(1e-3),np.log(1e3),size=3))
        means=rng.normal(0,1,size=(classes,3))*scale*rng.uniform(.2,3)
        X,y=synth(n,classes,scale,means,seed,offset=rng.normal(0,1,size=3)*scale*rng.choice([0,1,1e3,1e6]))
        if seed%5==1 and n>20:X[n//3:n//3+5]=X[0]
        if seed%5==2:X[:,1]=X[0,1]
        if seed%5==3:
            y[:]=0;y[0]=1;y[1]=2
        if seed%5==4:X=np.round(X,0)
        if min(np.bincount(y,minlength=classes))<2:continue
        folds=min(int(rng.integers(2,5)),int(np.bincount(y,minlength=classes).max()))
        bad+=check(f'random shape seed={seed}',X,y,classes,smoothing=float(10**rng.uniform(-9,-3)),folds=folds)
    print('TOTAL MISMATCHES',bad)
    sys.exit(1 if bad else 0)
