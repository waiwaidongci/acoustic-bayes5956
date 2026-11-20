from sqlalchemy import Column,Integer,String,JSON,Text,ForeignKey,UniqueConstraint
from ..config.database import Base
from ..config.settings import CONFIG
class RemovalAudit(Base):
    __tablename__='removal_audits'
    id=Column(Integer,primary_key=True)
    run_id=Column(Integer,ForeignKey(CONFIG['run_table']+'.id'),nullable=False)
    state=Column(String(20),default='queued')
    parameters=Column(JSON,nullable=False)
    fold_total=Column(Integer,nullable=False,default=0)
    error=Column(Text,default='')
    created_at=Column(String(40),nullable=False)
    updated_at=Column(String(40),nullable=False)
class RemovalFold(Base):
    __tablename__='removal_fold_results'
    __table_args__=(UniqueConstraint('audit_id','fold',name='uq_removal_fold'),)
    id=Column(Integer,primary_key=True)
    audit_id=Column(Integer,ForeignKey('removal_audits.id'),nullable=False)
    fold=Column(Integer,nullable=False)
    candidate_count=Column(Integer,nullable=False,default=0)
    flip_total=Column(Integer,nullable=False,default=0)
    missing_count=Column(Integer,nullable=False,default=0)
    retrained_count=Column(Integer,nullable=False,default=0)
    payload=Column(JSON,nullable=False)
    computed_at=Column(String(40),nullable=False)
