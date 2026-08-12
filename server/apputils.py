"""服务器通用工具（独立命名，避免与仓库根目录上游的 utils.py 冲突）。"""
import os
import re
import time
import uuid
import secrets
import string

from datetime import datetime

from flask import session, request, current_app
from werkzeug.utils import secure_filename

from extensions import db
from db_models import Task


def generate_random_password(length=12):
    chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))


def _csrf_token():
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_hex(32)
    return session['_csrf_token']


# ========== 登录限速 ==========
_login_attempts = {}
_login_lock = __import__('threading').Lock()
_LOGIN_MAX_FAILS = 5
_LOGIN_WINDOW = 900


def _login_blocked(remote_addr):
    with _login_lock:
        info = _login_attempts.get(remote_addr)
        return bool(info and info.get('blocked_until') and time.time() < info['blocked_until'])


def _record_login_failure(remote_addr):
    with _login_lock:
        now = time.time()
        info = _login_attempts.get(remote_addr)
        if not info or now - info.get('ts', 0) > _LOGIN_WINDOW:
            info = {'count': 0, 'ts': now, 'blocked_until': 0}
        info['count'] = info.get('count', 0) + 1
        info['ts'] = now
        if info['count'] >= _LOGIN_MAX_FAILS:
            info['blocked_until'] = now + _LOGIN_WINDOW
            info['count'] = 0
        _login_attempts[remote_addr] = info


def _clear_login_failures(remote_addr):
    with _login_lock:
        _login_attempts.pop(remote_addr, None)


# ========== 注册限速 ==========
_register_attempts = {}
_register_lock = __import__('threading').Lock()
_REGISTER_MAX = 5
_REGISTER_WINDOW = 3600


def _register_blocked(remote_addr):
    with _register_lock:
        info = _register_attempts.get(remote_addr)
        if info and info.get('count', 0) >= _REGISTER_MAX and time.time() < info.get('ts', 0) + _REGISTER_WINDOW:
            return True
        return False


def _record_register(remote_addr):
    with _register_lock:
        now = time.time()
        info = _register_attempts.get(remote_addr)
        if not info or now - info.get('ts', 0) > _REGISTER_WINDOW:
            info = {'count': 0, 'ts': now}
        info['count'] += 1
        info['ts'] = now
        _register_attempts[remote_addr] = info


def allowed_file(name):
    return name.lower().endswith(('.pth', '.pt', '.json', '.yaml', '.yml'))


def flash_errors(form_errors):
    from flask import flash
    for e in form_errors:
        flash(e, 'danger')


def save_uploaded(file, subdir):
    """保存上传文件到 uploads/<subdir>/，用唯一前缀避免冲突。"""
    ext = os.path.splitext(file.filename)[1]
    unique = uuid.uuid4().hex[:12]
    filename = f'{unique}_{secure_filename(file.filename)}'
    dest_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], subdir)
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, filename)
    file.save(path)
    return filename


def safe_join(base, *parts):
    base_abs = os.path.abspath(base)
    full = os.path.abspath(os.path.join(base_abs, *parts))
    if not full.startswith(base_abs + os.sep):
        raise ValueError('非法路径')
    return full


def dir_size_mb(path):
    total = 0
    if os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total / 1048576


def today_cutoff():
    now = datetime.utcnow()
    return datetime(now.year, now.month, now.day)


_STATUS_LABELS = {
    'pending': '排队中', 'claimed': '待执行', 'running': '运行中',
    'cancel_requested': '正在停止', 'done': '已完成', 'failed': '失败',
    'stopped': '已停止', 'expired': '已过期',
}


def status_label(status):
    return _STATUS_LABELS.get(status, status)


def global_queue_position(task):
    """该任务在全部 pending/claimed 中的全局位置（按 created_at,id 近似公平先后）。"""
    return Task.query.filter(
        Task.status.in_(['pending', 'claimed']),
        db.or_(
            Task.created_at < task.created_at,
            db.and_(Task.created_at == task.created_at, Task.id < task.id),
        ),
    ).count()


def arch_label_from_cfg(cfg):
    m = cfg.get('model') or {}
    arch = m.get('arch', '')
    if arch == 'rvc-flow':
        fm = m.get('flow_mode', 'a2')
        uni = m.get('use_unified_flow', False)
        if fm == 'a1':
            return 'A1 特征先验流', 'a1'
        return 'A2 后验流' + (' + 统一流' if uni else ''), 'a2'
    if arch == 'rvc':
        return 'RVC 轻量架构', 'rvc'
    return 'SoVITS v1', 'sovits-v1'


def read_model_cfg(cfg_path):
    if not cfg_path:
        return {}
    cfg_p = os.path.join(current_app.config['UPLOAD_FOLDER'], 'configs', cfg_path)
    if not os.path.exists(cfg_p):
        return {}
    try:
        import json
        with open(cfg_p, 'r', encoding='utf-8') as f:
            return json.load(f) or {}
    except Exception:
        return {}
