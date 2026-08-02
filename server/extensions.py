from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from sqlalchemy import event

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'login'


def init_sqlite_pragmas():
    """为 SQLite 开启 WAL 与忙等待，降低多线程写库冲突。"""
    try:
        @event.listens_for(db.engine, 'connect')
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            try:
                cursor = dbapi_connection.cursor()
                cursor.execute('PRAGMA journal_mode=WAL')
                cursor.execute('PRAGMA busy_timeout=30000')
                cursor.execute('PRAGMA synchronous=NORMAL')
                cursor.close()
            except Exception:
                pass
    except Exception:
        pass
