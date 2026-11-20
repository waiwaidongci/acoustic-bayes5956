from fastapi import APIRouter,Depends
from ..config.database import db_session
from ..services.auth import current_user,writer
from ..controllers.removal import RemovalController as C
router=APIRouter()
@router.get('/runs/{id}/removal')
def removal_status(id:int,db=Depends(db_session),u=Depends(current_user)):return C.status(db,id)
@router.post('/runs/{id}/removal')
def removal_create(id:int,db=Depends(db_session),u=Depends(writer)):
    audit,_=C.create(db,u,id)
    return {'audit':audit}
