from fastapi import HTTPException
from sqlalchemy import select
from redis import Redis
from ..config.settings import REDIS_URL, CONFIG
from ..models.run import Run
from ..models.removal import RemovalAudit, RemovalFold
from ..utils.serialization import row, now
from .domain_validation import validate_parameters
from .removal_compute import TIE_MARGIN
from .audit import log
queue=Redis.from_url(REDIS_URL)
QUEUE_NAME='removal:queue'
def ready_run(db,run_id):
    r=db.get(Run,run_id)
    if not r:raise HTTPException(404,'未找到分析任务')
    if r.state!='completed' or not r.result:raise HTTPException(409,'只有已完成的研究可以做剔除复核')
    if CONFIG['algorithm']!='gaussian':raise HTTPException(409,'当前研究方向不支持剔除复核')
    if not (r.result.get('rows') or []):raise HTTPException(409,'原结果缺少逐行预测，无法做剔除复核')
    folds=r.result.get('folds') or []
    if len(folds)<2:raise HTTPException(409,'原研究折数不足，无法沿用原分折')
    seen=set()
    for f in folds:
        tr=f.get('training_indices');va=f.get('validation_indices')
        if not isinstance(tr,list) or not isinstance(va,list) or not tr or not va:raise HTTPException(409,'原结果缺少分折索引，无法沿用原分折')
        if set(tr)&set(va):raise HTTPException(409,'原结果分折索引相互重叠，无法沿用原分折')
        seen.update(tr);seen.update(va)
    n=len(r.snapshot.get('records',[]))
    if seen!=set(range(n)):raise HTTPException(409,'原结果分折索引与冻结样本不一致，无法沿用原分折')
    return r
def audit_for(db,run_id):
    return db.scalar(select(RemovalAudit).where(RemovalAudit.run_id==run_id).order_by(RemovalAudit.id.desc()))
def _summary(a):
    return row(a)
def enqueue(db,user,run_id):
    """Queue the audit; never duplicates a running job and never reopens a completed one."""
    r=ready_run(db,run_id)
    params=validate_parameters(r.parameters)
    folds=r.result['folds']
    a=audit_for(db,run_id)
    if a and a.state=='completed':return _summary(a),'completed'
    if not a:
        a=RemovalAudit(run_id=r.id,state='queued',parameters={**params,'tie_margin':TIE_MARGIN},fold_total=len(folds),created_at=now(),updated_at=now())
        db.add(a);db.flush();log(db,user,'removal.queued',a.id,r.name);db.commit()
    elif a.state=='failed':
        a.state='queued';a.error='';a.updated_at=now();log(db,user,'removal.requeued',a.id,r.name);db.commit()
    else:
        # Already queued or running: re-announce the same audit so a message lost
        # with a restarted worker does not strand it. Compute folds are keyed by
        # (audit, fold), so repeated delivery cannot double count.
        log(db,user,'removal.requeued',a.id,r.name);db.commit()
    try:queue.rpush(QUEUE_NAME,a.id)
    except Exception:
        a.state='failed';a.error='剔除复核队列暂不可用，请稍后重试';a.updated_at=now();db.commit()
        raise HTTPException(503,a.error)
    db.refresh(a)
    return _summary(a),'queued'
def status(db,run_id):
    a=audit_for(db,run_id)
    if not a:return {'audit':None,'published':False,'progress':{'completed':0,'total':0},'ranking':[],'excluded_missing':[],'totals':None,'folds':[]}
    folds=db.scalars(select(RemovalFold).where(RemovalFold.audit_id==a.id).order_by(RemovalFold.fold)).all()
    total=a.fold_total or 0;done=len(folds)
    out={'audit':_summary(a),'progress':{'completed':done,'total':total},'published':a.state=='completed' and total>0 and done==total,'ranking':[],'excluded_missing':[],'totals':None,'folds':[]}
    if not out['published']:return out
    run=db.get(Run,a.run_id)
    refs=[rec.get('reference','') for rec in run.snapshot.get('records',[])]
    merged={}
    for f in folds:
        for item in f.payload['rows']:
            cur=merged.get(item['index'])
            if cur is None:
                merged[item['index']]={'index':item['index'],'reference':refs[item['index']] if item['index']<len(refs) else '','label':item['label'],'flips':int(item['flips']),'missing':bool(item['missing']),'retrained':bool(item['settled_by_retrain']),'folds':f.fold}
            else:
                if cur['label']!=item['label']:
                    raise HTTPException(500,'剔除复核分折记录不一致，请重新执行复核')
                cur['flips']+=int(item['flips'])
                cur['missing']=cur['missing'] or bool(item['missing'])
                cur['retrained']=cur['retrained'] or bool(item['settled_by_retrain'])
                cur['folds']=f'{cur["folds"]},{f.fold}'
    items=list(merged.values())
    ranked=sorted([x for x in items if not x['missing']],key=lambda x:(-x['flips'],x['index']))
    # Equal flip counts are ordered by the original input order, so the list is
    # reproducible; the rank column then numbers every candidate once.
    for pos,x in enumerate(ranked,1):x['rank']=pos
    excluded=sorted([x for x in items if x['missing']],key=lambda x:(-x['flips'],x['index']))
    out['ranking']=ranked;out['excluded_missing']=excluded
    out['totals']={'candidates':len(items),'flips':sum(x['flips'] for x in items),'missing_candidates':len(excluded),'retrained':sum(1 for x in items if x['retrained']),'max_flips':ranked[0]['flips'] if ranked else 0}
    out['folds']=[{'fold':f.fold,'candidates':f.candidate_count,'flips':f.flip_total,'missing':f.missing_count,'retrained':f.retrained_count,'validation':f.payload.get('validation')} for f in folds]
    return out
