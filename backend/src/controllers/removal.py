from ..services import removal
class RemovalController:
    status=staticmethod(removal.status)
    create=staticmethod(removal.enqueue)
