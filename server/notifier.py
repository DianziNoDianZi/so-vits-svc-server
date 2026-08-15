import json
import re
import smtplib
import os
from email.mime.text import MIMEText


# 邮件模板默认值；管理员可在后台覆盖（存 ServerSetting['email_templates']）
DEFAULT_TEMPLATES = {
    'infer_done': {
        'subject': '[SoVITS] 推理成功: {model}',
        'body': '推理任务成功\n\n任务: #{task_id}\n模型: {model}\n结果: {result_link}\n进度: {progress}\n\n--- So-VITS-SVC 推理服务 ---',
    },
    'infer_failed': {
        'subject': '[SoVITS] 推理失败: {model}',
        'body': '推理任务失败\n\n任务: #{task_id}\n模型: {model}\n错误: {error}\n\n--- So-VITS-SVC 推理服务 ---',
    },
    'welcome': {
        'subject': '[SoVITS] 欢迎使用',
        'body': '欢迎使用 So-VITS-SVC 推理服务\n\n账号: {username}\n接收邮箱: {recipient}\n\n您已注册成功，可以上传模型或使用平台提供的模型进行推理。\n推理完成后结果会发送到本邮箱（含下载链接）。\n\n--- So-VITS-SVC 推理服务 ---',
    },
    'resource': {
        'subject': '[SoVITS] 服务器资源紧张',
        'body': '{message}\n\n--- So-VITS-SVC 推理服务 ---',
    },
    'announcement': {
        'subject': '[SoVITS] {title}',
        'body': '{title}\n\n{content}\n\n--- So-VITS-SVC 推理服务 ---',
    },
    'train_progress': {
        'subject': '[SoVITS] 训练进度: {speaker} (step {step})',
        'body': '训练进度报告\n\n说话人: {speaker}\n类型: {model_type}\n当前: step {step} / {total_steps}\n阶段: {stage}\nETA: {eta}\n最新 Loss: {losses}\n\n--- So-VITS-SVC 推理服务 ---',
    },
    'train_done': {
        'subject': '[SoVITS] 训练{status_cn}: {speaker}',
        'body': '训练任务已完成\n\n说话人: {speaker}\n类型: {model_type}\n状态: {status_cn}\n用时: {duration}\n步数: {total_steps}\n\nSoVITS 模型: {model_link}\nSoVITS 配置: {cfg_link}\n扩散模型: {diff_link}\n扩散配置: {diff_cfg_link}\n\n--- So-VITS-SVC 推理服务 ---',
    },
    'train_failed': {
        'subject': '[SoVITS] 训练失败: {speaker}',
        'body': '训练任务失败\n\n说话人: {speaker}\n类型: {model_type}\n状态: {status_cn}\n错误: {error}\n\n--- So-VITS-SVC 推理服务 ---',
    },
}


def _render(tpl, **kw):
    """把 {占位符} 替换为传入的值；未提供的占位符原样保留。"""
    return re.sub(r'\{(\w+)\}', lambda m: str(kw.get(m.group(1), m.group(0))), tpl or '')


def _get_templates():
    """读管理员自定义模板；无则空 dict（用默认）。"""
    try:
        from db_models import ServerSetting
        from extensions import db
        row = db.session.get(ServerSetting, 'email_templates')
        if row and row.value:
            data = json.loads(row.value)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def render_email(key, **kw):
    """返回 (subject, body)，优先管理员自定义模板，否则用默认。"""
    t = _get_templates().get(key) or DEFAULT_TEMPLATES[key]
    return _render(t.get('subject'), **kw), _render(t.get('body'), **kw)


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
    try:
        model_name = task.config.model.name if task.config and task.config.model else '-'
    except Exception:
        model_name = '-'
    result_link = f'{server_url}/tasks/{task.id}/result' if task.result_filename else '无'
    key = 'infer_done' if task.status == 'done' else 'infer_failed'
    subject, body = render_email(key, task_id=task.id, model=model_name,
                                 result_link=result_link, progress=task.progress_msg or '-',
                                 error=task.error_msg or '-', username=user.username)
    cfg = _server_smtp_config()
    if cfg:
        return send(recipient, cfg['user'], cfg['pwd'], subject, body,
                    host=cfg['host'], port=cfg['port'], from_addr=cfg['from'])
    # 回退：每用户自己的 SMTP
    return send(recipient, user.smtp_user, user.smtp_pwd, subject, body,
                host=user.smtp_host or None, port=user.smtp_port or None)


def notify_welcome(user):
    """注册欢迎邮件：仅在服务器 SMTP 已配置时发送，失败静默。"""
    if not user:
        return False
    recipient = getattr(user, 'notify_email', None) or user.email
    if not recipient:
        return False
    subject, body = render_email('welcome', username=user.username, recipient=recipient)
    return send_via_server(recipient, subject, body)


def _train_recipient(task):
    user = task.user if task else None
    if not user:
        return None, None
    recipient = getattr(user, 'notify_email', None) or user.email
    return user, recipient


def notify_train_progress(task, server_url, step, total_steps, losses='-', stage='-', eta='-'):
    """训练过程中按 report_interval 步发送进度邮件。"""
    user, recipient = _train_recipient(task)
    if not user or not recipient:
        return False
    try:
        subject, body = render_email('train_progress', speaker=task.speaker,
                                     model_type=task.model_type, step=step,
                                     total_steps=total_steps or '?', stage=stage or '-',
                                     eta=eta or '-', losses=losses or '-', username=user.username)
        cfg = _server_smtp_config()
        if cfg:
            return send(recipient, cfg['user'], cfg['pwd'], subject, body,
                        host=cfg['host'], port=cfg['port'], from_addr=cfg['from'])
        return send(recipient, user.smtp_user, user.smtp_pwd, subject, body,
                    host=user.smtp_host or None, port=user.smtp_port or None)
    except Exception:
        return False


def notify_train_complete(task, server_url):
    """训练完成/失败时发送邮件。"""
    user, recipient = _train_recipient(task)
    if not user or not recipient:
        return False
    status_cn = '成功' if task.status == 'done' else '失败'
    model_link = f'{server_url}/train/result/{task.id}' if task.model_path else '无'
    cfg_link = f'{server_url}/train/result/{task.id}?config=1' if task.config_path else '无'
    diff_link = f'{server_url}/train/result/{task.id}?diff=1' if task.diff_model_path else '无'
    diff_cfg_link = f'{server_url}/train/result/{task.id}?diff=1&config=1' if task.diff_config_path else '无'
    dur = ''
    if task.created_at and task.done_at:
        secs = int((task.done_at - task.created_at).total_seconds())
        dur = f'{secs // 3600}h{(secs % 3600) // 60}m' if secs > 3600 else f'{secs // 60}m{secs % 60}s'
    if task.status == 'done':
        key = 'train_done'
        kw = dict(speaker=task.speaker, model_type=task.model_type, status_cn=status_cn,
                  duration=dur, total_steps=task.total_steps or 0,
                  model_link=model_link, cfg_link=cfg_link, diff_link=diff_link, diff_cfg_link=diff_cfg_link,
                  username=user.username)
    else:
        key = 'train_failed'
        kw = dict(speaker=task.speaker, model_type=task.model_type, status_cn=status_cn,
                  error=task.error_msg or '-', username=user.username)
    try:
        subject, body = render_email(key, **kw)
        cfg = _server_smtp_config()
        if cfg:
            return send(recipient, cfg['user'], cfg['pwd'], subject, body,
                        host=cfg['host'], port=cfg['port'], from_addr=cfg['from'])
        return send(recipient, user.smtp_user, user.smtp_pwd, subject, body,
                    host=user.smtp_host or None, port=user.smtp_port or None)
    except Exception:
        return False


