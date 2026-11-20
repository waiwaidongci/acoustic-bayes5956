import json,time,logging
from redis import Redis
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from .config.database import Session
from .config.settings import REDIS_URL,ARTIFACT_DIR
from .models.run import Run
from .models.removal import RemovalAudit,RemovalFold
from .models import source,record,user,note,audit
from .services.algorithm import calculate,FEATURES
from .services.removal_compute import compute_fold
from .services.audit import log
from .services.export import csv_response
from .utils.serialization import now
queue=Redis.from_url(REDIS_URL)
ARTIFACT_DIR.mkdir(parents=True,exist_ok=True)
def execute(id):
    with Session() as db:
        r=db.get(Run,id)
        if not r or r.state!='queued':return
        r.state='running';db.commit()
        try:
            result=calculate(r.snapshot['records'],r.parameters)
            json.dumps(result,allow_nan=False)
            r.result=result;r.state='completed'
            response=csv_response(result,'run.csv')
            (ARTIFACT_DIR/f'run-{r.id}.csv').write_bytes(response.body)
            log(db,'worker','run.completed',r.id)
        except Exception as e:
            logging.exception('analysis failed');r.state='failed';r.error=str(e)[:2000]
            log(db,'worker','run.failed',r.id,r.error)
        db.commit()
def _arrays(run):
    import numpy as np
    records=run.snapshot['records']
    X=np.array([[rec['values'][f] for f in FEATURES] for rec in records],float)
    labels=np.unique([rec['values']['label'] for rec in records]).tolist()
    index={label:i for i,label in enumerate(labels)}
    y=np.array([index[rec['values']['label']] for rec in records])
    base=np.array([index[row['predicted']] for row in run.result['rows']])
    return X,y,labels,base
def execute_removal(audit_id):
    import numpy as np
    with Session() as db:
        a=db.get(RemovalAudit,audit_id)
        if not a or a.state not in ('queued','running'):return
        run=db.get(Run,a.run_id)
        if not run or run.state!='completed' or not run.result:
            a.state='failed';a.error='原研究结果不可用';a.updated_at=now();db.commit();return
        a.state='running';a.updated_at=now();db.commit()
        try:
            X,y,labels,base=_arrays(run)
            smoothing=float(a.parameters['smoothing'])
            done={f.fold for f in db.scalars(select(RemovalFold).where(RemovalFold.audit_id==a.id))}
            for fold in run.result['folds']:
                number=int(fold['fold'])
                if number in done:continue
                payload=compute_fold(X,y,labels,np.array(fold['training_indices']),np.array(fold['validation_indices']),base,smoothing)
                db.add(RemovalFold(audit_id=a.id,fold=number,candidate_count=len(payload['rows']),flip_total=sum(r['flips'] for r in payload['rows']),missing_count=payload['missing'],retrained_count=payload['retrained'],payload=payload,computed_at=now()))
                try:db.commit()
                except IntegrityError:
                    db.rollback()
            a=db.get(RemovalAudit,audit_id)
            present=db.scalars(select(RemovalFold.fold).where(RemovalFold.audit_id==a.id)).all()
            if len(set(present))>=int(a.fold_total or 0):
                a.state='completed';a.error='';a.updated_at=now();log(db,'worker','removal.completed',a.id,f'{len(set(present))}折')
            else:
                a.state='failed';a.error='剔除复核未完成全部折，可重新发起续跑';a.updated_at=now()
            db.commit()
        except Exception as e:
            logging.exception('removal audit failed')
            a=db.get(RemovalAudit,audit_id)
            a.state='failed';a.error=str(e)[:2000];a.updated_at=now();db.commit()
def recover():
    with Session() as db:
        for a in db.scalars(select(RemovalAudit).where(RemovalAudit.state.in_(('queued','running')))).all():
            queue.rpush('removal:queue',a.id)
def main():
    try:recover()
    except Exception:logging.exception('removal recovery scan failed')
    while True:
        (ARTIFACT_DIR/'heartbeat').write_text(str(time.time()))
        value=queue.blpop(['analysis:queue','removal:queue'],timeout=2)
        if not value:continue
        key,raw=value
        if isinstance(key,bytes):key=key.decode()
        if isinstance(raw,bytes):raw=raw.decode()
        if key=='analysis:queue':execute(int(raw))
        else:execute_removal(int(raw))
if __name__=='__main__':main()
