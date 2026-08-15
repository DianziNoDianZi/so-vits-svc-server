"""数据库自动备份：用 SQLite 在线 backup API（WAL 安全），定时 + 保留策略。"""
import os
import sqlite3
import time
from datetime import datetime

from extensions import db
from services.quota import get_setting, set_setting

_app = None


def init_app(app):
    global _app
    _app = app


def backup_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backups')


def do_backup():
    """在线备份 data.db 到 backups/backup_<ts>.db，返回文件名；失败返回 None。"""
    os.makedirs(backup_dir(), exist_ok=True)
    db_path = _app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
    if not db_path or not os.path.exists(db_path):
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data.db')
    if not os.path.exists(db_path):
        return None
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    dest = os.path.join(backup_dir(), f'backup_{ts}.db')
    try:
        src = sqlite3.connect(db_path)
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
    except Exception:
        try:
            if os.path.exists(dest):
                os.remove(dest)
        except OSError:
            pass
        return None
    _prune_backups()
    return os.path.basename(dest)


def _prune_backups():
    keep = int(get_setting('backup_keep', 10) or 10)
    if keep <= 0:
        return
    files = sorted(
        (os.path.join(backup_dir(), f) for f in os.listdir(backup_dir()) if f.startswith('backup_') and f.endswith('.db')),
        key=lambda p: os.path.getmtime(p),
    )
    for old in files[:-keep]:
        try:
            os.remove(old)
        except OSError:
            pass


def _backup_daemon():
    with _app.app_context():
        last = 0.0
        while True:
            try:
                interval_hours = float(get_setting('backup_interval_hours', 24) or 24)
                interval = interval_hours * 3600
                if interval <= 0:
                    time.sleep(3600)
                    continue
                now = time.time()
                if now - last >= interval:
                    fname = do_backup()
                    if fname:
                        from services.audit import audit_log
                        audit_log('backup', f'自动备份完成: {fname}')
                    last = now
                time.sleep(300)
            except Exception:
                time.sleep(300)
