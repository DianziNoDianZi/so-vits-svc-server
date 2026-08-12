import smtplib
import os
from email.mime.text import MIMEText


def send(recipient, smtp_user, smtp_pwd, subject, body, host=None, port=None, use_ssl=True, from_addr=None):
    if not recipient or not smtp_user or not smtp_pwd:
        return False
    host = host or 'smtp.qq.com'
    port = int(port or 465)
    try:
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['Subject'] = subject
        # QQ 等要求 From 必须是合法邮箱（RFC5322），若“发件人”填了无 @ 的昵称会 550
        from_addr = from_addr or smtp_user
        if from_addr and '@' not in from_addr:
            from_addr = smtp_user
        msg['From'] = from_addr
        msg['To'] = recipient
        # 用 sendmail 发原始字节，别用 send_message：
        # 某些 Python 版本 + 中文内容时 send_message 会崩 "'utf8' is an invalid keyword argument for Compat32"
        raw = msg.as_string().encode('utf-8')
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=10) as s:
                s.login(smtp_user, smtp_pwd)
                s.sendmail(smtp_user, [recipient], raw)
        else:
            with smtplib.SMTP(host, port, timeout=10) as s:
                s.starttls()
                s.login(smtp_user, smtp_pwd)
                s.sendmail(smtp_user, [recipient], raw)
        return True
    except Exception:
        return False


_SMTP_ENV_MAP = {
    'smtp_host': 'SMTP_HOST',
    'smtp_port': 'SMTP_PORT',
    'smtp_user': 'SMTP_USER',
    'smtp_pass': 'SMTP_PASS',
    'mail_from': 'MAIL_FROM',
}


def _server_smtp_config():
    """读取服务器统一 SMTP 配置（ServerSetting 优先，环境变量可覆盖）。未配置/无上下文返回 None。"""
    from db_models import ServerSetting
    from extensions import db

    def get(key):
        env = _SMTP_ENV_MAP.get(key)
        if env and os.environ.get(env):
            return os.environ[env]
        try:
            row = db.session.get(ServerSetting, key)
            return row.value if row and row.value else None
        except Exception:
            return None

    host = get('smtp_host')
    user = get('smtp_user')
    pwd = get('smtp_pass')
    if not (host and user and pwd):
        return None
    return {
        'host': host,
        'port': int(get('smtp_port') or '465'),
        'user': user,
        'pwd': pwd,
        'from': get('mail_from') or user,
    }


def send_via_server(recipient, subject, body):
    """用服务器统一 SMTP 发送；未配置返回 False。"""
    cfg = _server_smtp_config()
    if not cfg:
        return False
    return send(recipient, cfg['user'], cfg['pwd'], subject, body,
                host=cfg['host'], port=cfg['port'], from_addr=cfg['from'])


def notify_inference_complete(task, server_url):
    """推理任务完成/失败时的邮件通知：优先服务器 SMTP，收件人为 notify_email。"""
    user = task.user if task else None
    if not user or not user.infer_notify:
        return False
    recipient = getattr(user, 'notify_email', None) or user.email
    if not recipient:
        return False
    status_cn = '成功' if task.status == 'done' else '失败'
    try:
        model_name = task.config.model.name if task.config and task.config.model else '-'
    except Exception:
        model_name = '-'
    result_link = f'{server_url}/tasks/{task.id}/result' if task.result_filename else '无'
    body = f"""推理任务 {status_cn}

任务: #{task.id}
模型: {model_name}
状态: {status_cn}
结果: {result_link}
进度: {task.progress_msg or '-'}

--- So-VITS-SVC 推理服务 ---
"""
    cfg = _server_smtp_config()
    if cfg:
        return send(recipient, cfg['user'], cfg['pwd'],
                    f'[SoVITS] 推理 {status_cn}: {model_name}', body,
                    host=cfg['host'], port=cfg['port'], from_addr=cfg['from'])
    # 回退：每用户自己的 SMTP
    return send(recipient, user.smtp_user, user.smtp_pwd,
                f'[SoVITS] 推理 {status_cn}: {model_name}', body,
                host=user.smtp_host or None, port=user.smtp_port or None)


def notify_welcome(user):
    """注册欢迎邮件：仅在服务器 SMTP 已配置时发送，失败静默。"""
    if not user:
        return False
    recipient = getattr(user, 'notify_email', None) or user.email
    if not recipient:
        return False
    body = f"""欢迎使用 So-VITS-SVC 推理服务

账号: {user.username}
接收邮箱: {recipient}

您已注册成功，可以上传模型或使用平台提供的模型进行推理。
推理完成后结果会发送到本邮箱（含下载链接）。

--- So-VITS-SVC 推理服务 ---
"""
    return send_via_server(recipient, '[SoVITS] 欢迎使用', body)


