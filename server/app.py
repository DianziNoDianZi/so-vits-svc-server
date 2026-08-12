"""So-VITS-SVC 多用户推理服务 — 应用工厂与入口。"""
import hmac
import json
import os
import secrets
import string
import sys

from flask import Flask, session, request, abort, url_for
from werkzeug.routing.exceptions import BuildError

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
# 路径顺序千万别手贱换：上游算法（models.py 等）要 `from utils import f0_to_coarse`，
# 指的就是仓库根目录那个 utils.py。之前我图省事在 server/ 底下也放了个 utils.py，
# 结果一上传模型推理就 ImportError，排查了半天才发现是俩模块抢名字。
# 所以服务端工具都叫 apputils，绝不跟上游撞名；SERVER_DIR 放最后插，保证它排在最前。
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SERVER_DIR)

from config import Config
from extensions import db, login_manager, init_sqlite_pragmas
from db_models import User, UserQuota
from authorization import is_active_user
from apputils import _csrf_token, generate_random_password
import services.scheduler as scheduler
import services.training as training


def migrate_db():
    # SQLite 不支持 ALTER 加约束，只能这样一列一列补。看着啰嗦，
    # 但至少老库升级不会炸，也不用指望 Flask-Migrate 那套自动生成的东西。
    import sqlalchemy as sa
    inspector = sa.inspect(db.engine)
    tcols = [c['name'] for c in inspector.get_columns('task')]
    if 'params_json' not in tcols:
        db.session.execute(sa.text('ALTER TABLE task ADD COLUMN params_json TEXT'))
    ucols = [c['name'] for c in inspector.get_columns('user')]
    user_cols = [('role', 'VARCHAR(20)'), ('is_active', 'BOOLEAN'), ('must_change_password', 'BOOLEAN'),
                 ('email', 'VARCHAR(200)'), ('notify_email', 'VARCHAR(200)'),
                 ('email_notify', 'BOOLEAN'), ('smtp_user', 'VARCHAR(200)'), ('smtp_pwd', 'VARCHAR(200)'),
                 ('smtp_host', 'VARCHAR(200)'), ('smtp_port', 'INTEGER'),
                 ('report_interval', 'INTEGER'), ('infer_notify', 'BOOLEAN')]
    for col, typ in user_cols:
        if col not in ucols:
            db.session.execute(sa.text(f'ALTER TABLE user ADD COLUMN {col} {typ}'))
    db.create_all()
    mcols = [c['name'] for c in inspector.get_columns('model')]
    model_cols = [('visibility', 'VARCHAR(20)'), ('status', 'VARCHAR(20)'), ('description', 'VARCHAR(500)'),
                  ('version', 'VARCHAR(100)'), ('review_note', 'VARCHAR(500)'), ('reviewed_at', 'DATETIME'),
                  ('tags', 'VARCHAR(500)'), ('public_requested', 'BOOLEAN')]
    for col, typ in model_cols:
        if col not in mcols:
            db.session.execute(sa.text(f'ALTER TABLE model ADD COLUMN {col} {typ}'))
    try:
        qcols = [c['name'] for c in inspector.get_columns('user_quota')]
        for col, typ in [('max_daily_tasks', 'INTEGER'), ('max_cpu_cores', 'INTEGER')]:
            if col not in qcols:
                db.session.execute(sa.text(f'ALTER TABLE user_quota ADD COLUMN {col} {typ}'))
    except Exception:
        pass
    tcols = [c['name'] for c in inspector.get_columns('task')]
    task_cols = [('model_id', 'INTEGER'), ('input_bytes', 'INTEGER'), ('input_duration', 'FLOAT'),
                 ('attempt_count', 'INTEGER'), ('priority_snapshot', 'INTEGER'), ('quota_snapshot_json', 'TEXT'),
                 ('lease_expires_at', 'DATETIME'), ('heartbeat_at', 'DATETIME'), ('claimed_by', 'VARCHAR(100)'),
                 ('cancel_requested_at', 'DATETIME'), ('cancel_reason', 'VARCHAR(500)'),
                 ('result_expires_at', 'DATETIME')]
    for col, typ in task_cols:
        if col not in tcols:
            db.session.execute(sa.text(f'ALTER TABLE task ADD COLUMN {col} {typ}'))
    try:
        db.session.execute(sa.text("UPDATE user SET is_active = 1 WHERE is_active IS NULL"))
        db.session.execute(sa.text("UPDATE user SET role = 'user' WHERE role IS NULL"))
        db.session.execute(sa.text("UPDATE user SET role = 'admin' WHERE username = 'admin'"))
        db.session.execute(sa.text("UPDATE user SET must_change_password = 0 WHERE must_change_password IS NULL"))
    except Exception:
        pass
    db.session.commit()


def init_admin():
    admin = User.query.filter_by(username='admin').first()
    if admin:
        if getattr(admin, 'role', 'user') != 'admin':
            admin.role = 'admin'
            db.session.commit()
        return False, None
    password = generate_random_password()
    from werkzeug.security import generate_password_hash
    admin = User(username='admin', password_hash=generate_password_hash(password),
                 role='admin', is_active=True, must_change_password=True)
    db.session.add(admin)
    db.session.commit()
    if not UserQuota.query.filter_by(user_id=admin.id).first():
        db.session.add(UserQuota(user_id=admin.id, priority=10, max_queued_tasks=10, max_running_tasks=1,
                                 max_daily_tasks=10 ** 9, storage_quota_bytes=10 ** 12,
                                 max_model_bytes=10 ** 12, max_private_models=100, results_retention_days=7))
        db.session.commit()
    return True, password


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    login_manager.init_app(app)

    from blueprints.auth import bp as auth_bp
    from blueprints.dashboard import bp as dash_bp
    from blueprints.models import bp as models_bp
    from blueprints.configs import bp as configs_bp
    from blueprints.inference import bp as infer_bp
    from blueprints.tasks import bp as tasks_bp
    from blueprints.announcements import bp as ann_bp
    from blueprints.admin import bp as admin_bp
    from blueprints.training import bp as train_bp
    for _b in (auth_bp, dash_bp, models_bp, configs_bp, infer_bp, tasks_bp, ann_bp, admin_bp, train_bp):
        app.register_blueprint(_b)

    # 模板里用无前缀 endpoint（url_for('login') 等），蓝图端点实际带前缀，这里做回退映射
    _blueprint_prefixes = ['auth', 'dashboard', 'models', 'configs', 'inference', 'tasks', 'announcements', 'admin', 'training']

    def _resolve_blueprint_endpoint(error, endpoint, values):
        if isinstance(error, BuildError) and '.' not in endpoint:
            for prefix in _blueprint_prefixes:
                try:
                    return url_for(f'{prefix}.{endpoint}', **values)
                except BuildError:
                    continue
        raise error  # 找不到就报错，别静默跳根目录掩盖问题

    app.url_build_error_handlers.append(_resolve_blueprint_endpoint)

    with app.app_context():
        init_sqlite_pragmas()
        for _sub in ('models', 'configs', 'audio', 'results', 'dataset_zips', 'train_logs'):
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], _sub), exist_ok=True)
        db.create_all()
        migrate_db()
        created, pwd = init_admin()
        if created:
            print(f'\n{"=" * 50}')
            print('  首次启动！管理员账号已创建')
            print('  用户名: admin')
            print(f'  初始密码:   {pwd}')
            print('  登录后将强制修改密码')
            print(f'{"=" * 50}\n')

    @app.template_filter('from_json')
    def from_json_filter(s):
        try:
            return json.loads(s) if s else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    @app.before_request
    def _protect_csrf():
        if request.method == 'POST':
            token = session.get('_csrf_token')
            form_token = request.form.get('_csrf_token', '')
            if not token or not form_token or not hmac.compare_digest(token, form_token):
                abort(400, description='CSRF 校验失败，请刷新页面后重试')

    @app.context_processor
    def _inject():
        return {'csrf_token': _csrf_token, 'resource_warning': scheduler.resource_warning}

    @login_manager.user_loader
    def load_user(user_id):
        try:
            user = db.session.get(User, int(user_id))
            if user and not is_active_user(user):
                return None
            return user
        except Exception:
            return None

    scheduler.init_app(app)
    training.init_app(app)
    return app


app = create_app()


if __name__ == '__main__':
    import atexit
    atexit.register(scheduler.stop_inference_daemon)
    scheduler.ensure_worker()
    scheduler._recover_tasks()
    training.ensure_training_worker()
    port = int(os.environ.get('PORT', 5000))
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=port, threads=8)
    except ImportError:
        app.run(host='0.0.0.0', port=port, threaded=True)
