from extensions import db
from flask_login import UserMixin
from datetime import datetime


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    device_pref = db.Column(db.String(10), default='auto')
    memory_limit = db.Column(db.Float, default=0)  # GB, 0=无限制
    email = db.Column(db.String(200), nullable=True)
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


class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)

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
    params_json = db.Column(db.Text, nullable=True)  # 本次推理合并后的参数（覆盖配置默认值）
    status = db.Column(db.String(20), default='pending')
    progress_msg = db.Column(db.String(500), default='')
    audio_filename = db.Column(db.String(500), nullable=False)
    result_filename = db.Column(db.String(500), nullable=True)
    error_msg = db.Column(db.String(1000), nullable=True)
    device_pref = db.Column(db.String(10), default='auto')
    memory_limit = db.Column(db.Float, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    done_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship('User', backref='tasks')
    config = db.relationship('InferenceConfig')


class TrainingTask(db.Model):
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
    anomaly_token = db.Column(db.String(64), nullable=True)   # 训练异常确认 token
    anomaly_state = db.Column(db.String(20), default='')      # '' / pending / confirmed
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
