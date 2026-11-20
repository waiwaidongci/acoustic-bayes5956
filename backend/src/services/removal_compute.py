import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# A deletion is settled by an actual retrain only when its winning margin is
# below this tolerance, i.e. when float64 rounding could change the argmax.
# Continuous acoustic features essentially never get this close; the count of
# such cases is reported alongside the published table.
TIE_MARGIN=1e-6
# Peak working memory allowed for one batch of candidate deletions.
BATCH_BYTES=32*1024*1024


def _batch_size(classes,validation,features):
    per_candidate=8*(classes*validation*(2*features+2))
    return max(1,min(128,int(BATCH_BYTES/per_candidate)))


def compute_fold(X,y,classes,train,validation,base,smoothing):
    """Flip counts for deleting each training row of one fold, without refits.

    Deleting one training row changes the scaler mean/variance and every class
    mean/variance in closed form from raw first and second moments. The whole
    validation fold is then scored in batches, which bounds peak memory.
    Arithmetic mirrors sklearn's StandardScaler + GaussianNB chain: argmax over
    standardised Gaussian joint log-likelihoods, ties to the lowest class index
    exactly as ``predict`` does. Values are centred on the fold mean first so
    large raw magnitudes keep their precision.
    """
    train=np.asarray(train,dtype=np.int64);validation=np.asarray(validation,dtype=np.int64)
    k=len(classes);m=int(train.size);v=int(validation.size);f=X.shape[1]
    if m<2:raise ValueError('训练折样本不足，无法做剔除复核')
    if k<2:raise ValueError('剔除复核至少需要两种类型')
    npr=m-1
    offset=X[train].mean(axis=0)
    Xl=X-offset
    Xt=Xl[train];yt=y[train]
    Xv=Xl[validation];basev=base[validation]
    s1=Xt.sum(axis=0);s2=(Xt*Xt).sum(axis=0)
    counts=np.bincount(yt,minlength=k).astype(np.float64)
    cs1=np.zeros((k,f));cs2=np.zeros((k,f))
    for c in range(k):
        hit=yt==c
        if hit.any():cs1[c]=Xt[hit].sum(axis=0);cs2[c]=(Xt[hit]**2).sum(axis=0)
    flips=np.zeros(m,dtype=np.int64);margin=np.full(m,np.inf);missing=np.zeros(m,dtype=bool)
    eps0=np.finfo(np.float64).eps
    step=_batch_size(k,v,f)
    for lo in range(0,m,step):
        pidx=np.arange(lo,min(lo+step,m));xi=Xt[pidx];yi=yt[pidx];size=pidx.size
        mu=(s1[None,:]-xi)/npr
        var_total=np.maximum((s2[None,:]-xi*xi)/npr-mu*mu,0.0)
        scale2=var_total.copy()
        # sklearn pins the scale of near-constant features to 1; the test uses
        # the raw-space mean, hence the offset.
        raw_mean=mu+offset
        constant=var_total<=npr*eps0*var_total+(npr*raw_mean*eps0)**2
        scale2[constant]=1.0
        eps=smoothing*(var_total/scale2).max(axis=1)
        inv=np.sqrt(scale2)
        xv=(Xv[None,:,:]-mu[:,None,:])/inv[:,None,:]
        scores=np.full((size,k,v),-np.inf)
        for c in range(k):
            hit=(yi==c).astype(np.float64)
            left=counts[c]-hit
            kept=left>0
            div=np.where(kept,left,1.0)[:,None]
            mu_c=(cs1[c][None,:]-hit[:,None]*xi)/div
            var_c=np.maximum((cs2[c][None,:]-hit[:,None]*xi*xi)/div-mu_c*mu_c,0.0)/scale2
            denom=var_c+eps[:,None]
            log_prior=np.where(kept,np.log(np.where(kept,left/npr,1.0)),-np.inf)
            mu_sc=(mu_c-mu)/inv
            diff=xv-mu_sc[:,None,:]
            scores[:,c,:]=log_prior[:,None]-0.5*np.log(2.0*np.pi*denom).sum(axis=1)[:,None]-0.5*(diff*diff/denom[:,None,:]).sum(axis=2)
            missing[pidx[~kept]]=True
        prediction=scores.argmax(axis=1)
        flips[pidx]=(prediction!=basev[None,:]).sum(axis=1)
        # A deletion that empties a class cannot be scored under the original
        # rule; its score above is a model without that class, so discard it and
        # report the candidate separately from the ranking.
        flips[pidx[missing[pidx]]]=0
        top=np.partition(scores,k-2,axis=1)
        best=top[:,k-1,:];runner_up=top[:,k-2,:]
        # Rounding on either side scales with the score magnitude, so compare
        # the margin relatively: only a genuinely undecidable row is retrained.
        margin[pidx]=(best-runner_up).min(axis=1)/np.maximum(1.0,np.abs(best).max(axis=1))
    ambiguous=np.where((margin<TIE_MARGIN)&~missing)[0]
    for p in ambiguous:
        keep=np.ones(m,dtype=bool);keep[p]=False
        tr=train[keep]
        scaler=StandardScaler().fit(X[tr])
        model=GaussianNB(var_smoothing=smoothing).fit(scaler.transform(X[tr]),y[tr])
        flips[p]=int((model.predict(scaler.transform(X[validation]))!=basev).sum())
    rows=[{'index':int(train[p]),'label':str(classes[y[train[p]]]),'flips':int(flips[p]),'missing':bool(missing[p]),'settled_by_retrain':bool(not missing[p] and margin[p]<TIE_MARGIN)} for p in range(m)]
    return {'rows':rows,'validation':v,'retrained':int(ambiguous.size),'missing':int(missing.sum()),'batch_size':int(step)}
