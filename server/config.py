import os
import secrets

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _load_or_create_secret():
    """优先使用环境变量；否则持久化一个随机密钥，避免每次重启会话失效。"""
    env_key = os.environ.get('SECRET_KEY')
    if env_key:
        return env_key
    key_path = os.path.join(BASE_DIR, 'secret_key.txt')
    try:
        if os.path.exists(key_path):
            with open(key_path, 'r', encoding='utf-8') as f:
                key = f.read().strip()
                if key:
                    return key
        key = secrets.token_hex(32)
        with open(key_path, 'w', encoding='utf-8') as f:
            f.write(key)
        try:
            os.chmod(key_path, 0o600)
        except OSError:
            pass
        return key
    except OSError:
        # 无法写文件时退回临时密钥（每次重启会话失效，但不会使用公开默认值）
        return secrets.token_hex(32)


class Config:
    SECRET_KEY = _load_or_create_secret()
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        f'sqlite:///{os.path.join(BASE_DIR, "data.db")}'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    TRUST_PROXY = os.environ.get('TRUST_PROXY', '0') == '1'
    MAX_CONTENT_LENGTH = 4 * 1024 * 1024 * 1024  # 4GB max upload（whisper large ~3GB）
    TEMPLATES_AUTO_RELOAD = True  # 模板文件修改后自动重载，无需重启服务
    # SQLite 连接等待锁超时（秒），缓解多线程写库的 "database is locked"
    SQLALCHEMY_ENGINE_OPTIONS = {
        'connect_args': {'timeout': 30},
    }
