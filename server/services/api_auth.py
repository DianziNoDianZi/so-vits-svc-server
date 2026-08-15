"""对外 API 的 API Key 鉴权。"""
import hashlib
import secrets
from functools import wraps
from datetime import datetime

from flask import g, request, jsonify

from extensions import db
from db_models import ApiToken


def hash_token(token):
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def generate_token(user, name='default'):
    """生成一次性明文 Key（只存 hash），返回明文供用户保存。"""
    token = secrets.token_urlsafe(32)
    db.session.add(ApiToken(user_id=user.id, token_hash=hash_token(token), name=name))
    db.session.commit()
    return token


def revoke_token(user, token_id):
    t = ApiToken.query.filter_by(id=token_id, user_id=user.id).first()
    if not t:
        return False
    t.revoked_at = datetime.utcnow()
    db.session.commit()
    return True


def get_user_from_request():
    """从请求头取 Key：优先 X-API-Key，其次 Authorization: Bearer（兼容旧用法）。"""
    token = ''
    api_key = request.headers.get('X-API-Key', '')
    if api_key:
        token = api_key.strip()
    else:
        auth = request.headers.get('Authorization', '')
        if auth.startswith('Bearer '):
            token = auth[len('Bearer '):].strip()
    if not token:
        return None
    t = ApiToken.query.filter_by(token_hash=hash_token(token)).first()
    if not t or t.revoked_at:
        return None
    return t.user


def require_api_token(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = get_user_from_request()
        if not user:
            return jsonify({'error': '未授权或 token 无效'}), 401
        g.api_user = user
        return f(*args, **kwargs)
    return wrapper
