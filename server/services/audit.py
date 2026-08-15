"""审计日志：记录关键管理操作，便于追溯。"""
from datetime import datetime

from flask import request
from flask_login import current_user

from extensions import db
from db_models import AuditLog


def audit_log(action, detail=None):
    """写一条审计记录。action 为动词短语，detail 为人类可读描述。"""
    try:
        user_id = current_user.id if current_user and current_user.is_authenticated else None
    except Exception:
        user_id = None
    try:
        ip = request.remote_addr or ''
    except Exception:
        ip = ''
    try:
        db.session.add(AuditLog(user_id=user_id, action=action, detail=(detail or '')[:1000], ip=ip))
        db.session.commit()
    except Exception:
        db.session.rollback()
