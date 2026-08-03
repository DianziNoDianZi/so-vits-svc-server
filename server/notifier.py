import smtplib
import os
from email.mime.text import MIMEText


def send(recipient, smtp_user, smtp_pwd, subject, body, host=None, port=None, use_ssl=True):
    if not recipient or not smtp_user or not smtp_pwd:
        return False
    host = host or 'smtp.qq.com'
    port = int(port or 465)
    try:
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = recipient
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=10) as s:
                s.login(smtp_user, smtp_pwd)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=10) as s:
                s.starttls()
                s.login(smtp_user, smtp_pwd)
                s.send_message(msg)
        return True
    except Exception:
        return False


def notify_train_complete(task, server_url):
    user = task.user
    if not user or not user.email_notify or not user.email:
        return False
    status_cn = '成功' if task.status == 'done' else '失败'
    model_link = f'{server_url}/train/result/{task.id}' if task.model_path else '无'
    cfg_link = f'{server_url}/train/result/{task.id}?config=1' if task.config_path else '无'
    diff_link = f'{server_url}/train/result/{task.id}?diff=1' if task.diff_model_path else '无'
    diff_cfg_link = f'{server_url}/train/result/{task.id}?diff=1&config=1' if task.diff_config_path else '无'
    dur = ''
    if task.created_at and task.done_at:
        secs = int((task.done_at - task.created_at).total_seconds())
        if secs > 3600:
            dur = f'{secs // 3600}h{(secs % 3600) // 60}m'
        else:
            dur = f'{secs // 60}m{secs % 60}s'
    body = f"""训练任务 {task.speaker} 已完成

说话人: {task.speaker}
类型: {task.model_type}
状态: {status_cn}
用时: {dur}
步数: {task.total_steps}

SoVITS 模型: {model_link}
SoVITS 配置: {cfg_link}
扩散模型: {diff_link}
扩散配置: {diff_cfg_link}

--- So-VITS-SVC 推理服务 ---
"""
    return send(user.email, user.smtp_user, user.smtp_pwd,
                f'[SoVITS] 训练 {status_cn}: {task.speaker}', body,
                host=user.smtp_host or None, port=user.smtp_port or None)


def notify_train_progress(task, server_url, step, total_steps, losses, stage, eta):
    """训练过程中的阶段性进度报告。"""
    user = task.user if task else None
    if not user or not user.email_notify or not user.email:
        return False
    body = f"""训练进度报告: {task.speaker}

任务: #{task.id} ({task.model_type})
当前: step {step} / {total_steps or '?'}
阶段: {stage or '-'}
ETA: {eta or '-'}
最新 Loss: {losses or '-'}

--- So-VITS-SVC 推理服务 ---
"""
    return send(user.email, user.smtp_user, user.smtp_pwd,
                f'[SoVITS] 训练进度 {task.speaker} (step {step})', body,
                host=user.smtp_host or None, port=user.smtp_port or None)


def notify_inference_complete(task, server_url):
    """推理任务完成/失败时的邮件通知。"""
    user = task.user if task else None
    if not user or not user.infer_notify or not user.email:
        return False
    status_cn = '成功' if task.status == 'done' else '失败'
    model_name = task.config.model.name if task.config and task.config.model else '-'
    result_link = f'{server_url}/tasks/{task.id}/result' if task.result_filename else '无'
    body = f"""推理任务 {status_cn}

任务: #{task.id}
模型: {model_name}
状态: {status_cn}
结果: {result_link}
进度: {task.progress_msg or '-'}

--- So-VITS-SVC 推理服务 ---
"""
    return send(user.email, user.smtp_user, user.smtp_pwd,
                f'[SoVITS] 推理 {status_cn}: {model_name}', body,
                host=user.smtp_host or None, port=user.smtp_port or None)
