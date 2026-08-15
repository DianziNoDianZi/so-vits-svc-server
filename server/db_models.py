from datetime import datetime

from extensions import db
from flask_login import UserMixin


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='user')
    is_active = db.Column(db.Boolean, default=True)
    must_change_password = db.Column(db.Boolean, default=False)
    device_pref = db.Column(db.String(10), default='auto')
    memory_limit = db.Column(db.Float, default=0)  # GB, 0=无限制
    email = db.Column(db.String(200), nullable=True)
    notify_email = db.Column(db.String(200), nullable=True)  # 结果接收邮箱（可不同于账号邮箱）
    email_notify = db.Column(db.Boolean, default=False)
    smtp_user = db.Column(db.String(200), nullable=True)
    smtp_pwd = db.Column(db.String(200), nullable=True)
    smtp_host = db.Column(db.String(200), nullable=True)
    smtp_port = db.Column(db.Integer, nullable=True)
    report_interval = db.Column(db.Integer, default=0)   # 训练进度报告间隔（步数，0=关闭）
    infer_notify = db.Column(db.Boolean, default=False)  # 推理完成时邮件通知
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    models = db.relationship('Model', backref='owner', lazy='dynamic')
    configs = db.relationship('InferenceConfig', backref='owner', lazy='dynamic')
    quotas = db.relationship('UserQuota', backref='user', lazy='dynamic')


class UserQuota(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    enabled = db.Column(db.Boolean, default=True)
    max_queued_tasks = db.Column(db.Integer, default=4)
    max_running_tasks = db.Column(db.Integer, default=1)
    max_input_seconds = db.Column(db.Integer, default=600)
    max_daily_tasks = db.Column(db.Integer, default=50)   # 每日可提交任务数（不是音频秒数！）
    max_cpu_cores = db.Column(db.Integer, default=0)      # 0=不限制，免得小白管理员真去数 CPU
    storage_quota_bytes = db.Column(db.BigInteger, default=10 * 1024 * 1024 * 1024)
    max_model_bytes = db.Column(db.BigInteger, default=4 * 1024 * 1024 * 1024)
    max_private_models = db.Column(db.Integer, default=3)
    priority = db.Column(db.Integer, default=1)
    results_retention_days = db.Column(db.Integer, default=7)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    visibility = db.Column(db.String(20), default='private')
    status = db.Column(db.String(20), default='ready')
    description = db.Column(db.String(500), nullable=True)
    version = db.Column(db.String(100), nullable=True)
    review_note = db.Column(db.String(500), nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    public_requested = db.Column(db.Boolean, default=False)

    # File paths (relative to uploads/)
    model_path = db.Column(db.String(500), nullable=False)       # G_*.pth
    config_path = db.Column(db.String(500), nullable=False)      # config.json
    diff_model_path = db.Column(db.String(500), nullable=True)   # model_*.pt（扩散模型）
    diff_config_path = db.Column(db.String(500), nullable=True)  # diffusion.yaml（扩散配置）
    cluster_path = db.Column(db.String(500), nullable=True)      # kmeans_*.pt
    tags = db.Column(db.String(500), nullable=True)              # 用户自定义标签，逗号分隔

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    configs = db.relationship('InferenceConfig', backref='model', lazy='dynamic')


class InferenceConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey('model.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    params_json = db.Column(db.Text, nullable=False, default='{}')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    config_id = db.Column(db.Integer, db.ForeignKey('inference_config.id'), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey('model.id'), nullable=True)
    params_json = db.Column(db.Text, nullable=True)  # 本次推理合并后的参数（覆盖配置默认值）
    status = db.Column(db.String(20), default='pending')
    progress_msg = db.Column(db.String(500), default='')
    audio_filename = db.Column(db.String(500), nullable=False)
    result_filename = db.Column(db.String(500), nullable=True)
    error_msg = db.Column(db.String(1000), nullable=True)
    device_pref = db.Column(db.String(10), default='auto')
    memory_limit = db.Column(db.Float, default=0)
    input_bytes = db.Column(db.BigInteger, default=0)
    input_duration = db.Column(db.Float, default=0)
    attempt_count = db.Column(db.Integer, default=0)
    priority_snapshot = db.Column(db.Integer, default=1)
    quota_snapshot_json = db.Column(db.Text, nullable=True)
    lease_expires_at = db.Column(db.DateTime, nullable=True)
    heartbeat_at = db.Column(db.DateTime, nullable=True)
    claimed_by = db.Column(db.String(100), nullable=True)
    cancel_requested_at = db.Column(db.DateTime, nullable=True)
    cancel_reason = db.Column(db.String(500), nullable=True)
    result_expires_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    done_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship('User', backref='tasks')
    config = db.relationship('InferenceConfig')
    model = db.relationship('Model')


class StoredFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    owner_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    task_id = db.Column(db.Integer, db.ForeignKey('task.id'), nullable=True)
    model_id = db.Column(db.Integer, db.ForeignKey('model.id'), nullable=True)
    kind = db.Column(db.String(40), nullable=False)
    storage_key = db.Column(db.String(500), nullable=False, unique=True)
    original_filename = db.Column(db.String(255), nullable=True)
    content_type = db.Column(db.String(100), nullable=True)
    size_bytes = db.Column(db.BigInteger, default=0)
    sha256 = db.Column(db.String(64), nullable=True)
    status = db.Column(db.String(20), default='ready')
    expires_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class TaskEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey('task.id'), nullable=False)
    event_type = db.Column(db.String(40), nullable=False)
    message = db.Column(db.String(500), nullable=True)
    payload_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ServerSetting(db.Model):
    key = db.Column(db.String(100), primary_key=True)
    value = db.Column(db.Text, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Announcement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_pinned = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    created_by_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    author = db.relationship('User')


class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    action = db.Column(db.String(60), nullable=False)
    detail = db.Column(db.String(1000), nullable=True)
    ip = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User')


class TrainingTask(db.Model):
    """训练任务（可选功能：仅管理员使用，服务端开关控制）。"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    dataset_zip = db.Column(db.String(500), nullable=False)
    speaker = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(20), default='pending')
    model_type = db.Column(db.String(20), default='sovits')
    batch_size = db.Column(db.Integer, default=4)
    total_steps = db.Column(db.Integer, default=4200)
    keep_ckpts = db.Column(db.Integer, default=3)
    params_json = db.Column(db.Text, nullable=True)
    progress_msg = db.Column(db.String(500), default='')
    model_path = db.Column(db.String(500), nullable=True)
    config_path = db.Column(db.String(500), nullable=True)
    diff_model_path = db.Column(db.String(500), nullable=True)
    diff_config_path = db.Column(db.String(500), nullable=True)
    error_msg = db.Column(db.String(1000), nullable=True)
    log_path = db.Column(db.String(500), nullable=True)
    resume_from_id = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    done_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship('User', backref='training_tasks')


DEFAULT_PARAMS = {
    'device': 'auto',
    'memory_limit': 0,
    'f0_predictor': 'pm',
    'k_step': 100,
    'vc_transform': 0,
    'cluster_ratio': 0,
    'slice_db': -40,
    'noise_scale': 0.4,
    'pad_seconds': 0.5,
    'auto_f0': False,
    'enhancer': False,
    'second_encoding': False,
    'loudness_envelope': 0,
    'output_format': 'wav',
    'hybrid_mode': 'auto',
}
