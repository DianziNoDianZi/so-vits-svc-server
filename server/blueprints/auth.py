"""认证蓝图：登录/注册/改密/通知设置/退出。"""
import re
from datetime import datetime

from flask import Blueprint, render_template, request, redirect, url_for, flash, abort
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from extensions import db
from db_models import User
from services.quota import current_quota, get_setting
from apputils import (
    _login_blocked, _record_login_failure, _clear_login_failures,
    _register_blocked, _record_register, is_guest,
)

bp = Blueprint('auth', __name__)


@bp.route('/login', methods=['GET', 'POST'], endpoint='login')
def login():
    if get_setting('auth_mode', 'password') == 'ip':
        abort(404)
    if request.method == 'POST':
        remote = request.remote_addr or 'unknown'
        if _login_blocked(remote):
            flash('尝试过于频繁，请 15 分钟后再试', 'danger')
        else:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password_hash, password):
                if login_user(user):
                    _clear_login_failures(remote)
                    if getattr(user, 'must_change_password', False):
                        return redirect(url_for('change_password'))
                    return redirect(url_for('dashboard'))
            _record_login_failure(remote)
            flash('用户名或密码错误', 'danger')
    return render_template('login.html')


@bp.route('/register', methods=['GET', 'POST'], endpoint='register')
def register():
    if get_setting('auth_mode', 'password') == 'ip':
        abort(404)
    if get_setting('allow_registration', __import__('os').environ.get('ALLOW_REGISTRATION', '1')) != '1':
        flash('当前未开放注册', 'danger')
        return redirect(url_for('login'))
    if request.method == 'POST':
        remote = request.remote_addr or 'unknown'
        if _register_blocked(remote):
            flash('注册尝试过于频繁，请稍后再试', 'danger')
            return render_template('register.html')
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        email = request.form.get('email', '').strip()
        notify_email = request.form.get('notify_email', '').strip()
        errors = []
        if len(username) < 2:
            errors.append('用户名至少 2 个字符')
        if len(password) < 6:
            errors.append('密码至少 6 位')
        if password != confirm:
            errors.append('两次密码不一致')
        if not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
            errors.append('邮箱格式不正确')
        if notify_email and not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', notify_email):
            errors.append('结果接收邮箱格式不正确')
        if not errors and User.query.filter_by(username=username).first():
            errors.append('用户名已存在')

        # 邀请码模式：required 必填有效码；optional 选填（填了必须有效）
        invite_mode = get_setting('invite_mode', 'off')
        invite_code = request.form.get('invite_code', '').strip()
        from db_models import InviteCode
        code_obj = None
        if invite_code:
            code_obj = InviteCode.query.filter_by(code=invite_code).first()
            if not code_obj:
                errors.append('邀请码不存在')
            elif code_obj.used_at:
                errors.append('邀请码已被使用')
            elif code_obj.expires_at and code_obj.expires_at < datetime.utcnow():
                errors.append('邀请码已过期')
        elif invite_mode == 'required':
            errors.append('当前需要邀请码才能注册')

        if errors:
            _record_register(remote)
            for e in errors:
                flash(e, 'danger')
            return render_template('register.html', username=username, email=email, notify_email=notify_email,
                                   invite_mode=invite_mode)
        user = User(username=username, password_hash=generate_password_hash(password), role='user',
                    is_active=True, email=email or None, notify_email=notify_email or None, infer_notify=True)
        db.session.add(user)
        db.session.flush()
        if code_obj:
            code_obj.used_by_user_id = user.id
            code_obj.used_at = datetime.utcnow()
        db.session.commit()
        current_quota(user)
        login_user(user)
        try:
            from notifier import notify_welcome
            notify_welcome(user)
        except Exception:
            pass
        flash('注册成功，欢迎使用', 'success')
        return redirect(url_for('dashboard'))
    invite_mode = get_setting('invite_mode', 'off')
    return render_template('register.html', invite_mode=invite_mode)


@bp.route('/admin-login', methods=['GET', 'POST'], endpoint='admin_login')
def admin_login():
    if current_user.is_authenticated and not is_guest(current_user):
        return redirect(url_for('admin_index'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username, role='admin').first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('admin_index'))
        flash('管理员账号或密码错误', 'danger')
    return render_template('login.html', admin_login=True)


@bp.route('/change-password', methods=['GET', 'POST'], endpoint='change_password')
@login_required
def change_password():
    if is_guest(current_user):
        abort(403)
    if request.method == 'POST':
        old = request.form.get('old_password', '')
        new = request.form.get('new_password', '')
        confirm = request.form.get('confirm_password', '')
        new_username = request.form.get('username', '').strip()
        if not check_password_hash(current_user.password_hash, old):
            flash('当前密码错误', 'danger')
            return render_template('change_password.html')
        if new_username and new_username != current_user.username:
            if User.query.filter_by(username=new_username).first():
                flash('用户名已存在', 'danger')
                return render_template('change_password.html')
            current_user.username = new_username
        if new:
            if len(new) < 6:
                flash('新密码至少 6 位', 'danger')
                return render_template('change_password.html')
            if new != confirm:
                flash('两次密码不一致', 'danger')
                return render_template('change_password.html')
            current_user.password_hash = generate_password_hash(new)
        current_user.must_change_password = False
        db.session.commit()
        flash('设置已保存', 'success')
        return redirect(url_for('dashboard'))
    return render_template('change_password.html')


@bp.route('/save-notify', methods=['POST'], endpoint='save_notify')
@login_required
def save_notify():
    if is_guest(current_user):
        abort(403)
    current_user.email = request.form.get('email', '').strip() or None
    current_user.notify_email = request.form.get('notify_email', '').strip() or None
    current_user.infer_notify = request.form.get('infer_notify') == '1'
    db.session.commit()
    flash('通知设置已保存', 'success')
    return redirect(url_for('settings'))


@bp.route('/test-notify', methods=['POST'], endpoint='test_notify')
@login_required
def test_notify():
    if is_guest(current_user):
        abort(403)
    from notifier import send_via_server
    recipient = getattr(current_user, 'notify_email', None) or current_user.email
    if not recipient:
        flash('请先填写接收邮箱', 'danger')
        return redirect(url_for('settings'))
    ok = send_via_server(recipient, '[SoVITS] 测试通知', '这是一封测试邮件，通知配置正常！')
    flash('测试邮件已发送，请检查收件箱' if ok else '发送失败，请检查服务器 SMTP 配置',
          'success' if ok else 'danger')
    return redirect(url_for('settings'))


@bp.route('/settings', methods=['GET', 'POST'], endpoint='settings')
@login_required
def settings():
    if is_guest(current_user):
        abort(403)
    if request.method == 'POST':
        old = request.form.get('old_password', '')
        new = request.form.get('new_password', '')
        confirm = request.form.get('confirm_password', '')
        new_username = request.form.get('username', '').strip()
        if not check_password_hash(current_user.password_hash, old):
            flash('当前密码错误', 'danger')
            return render_template('settings.html')
        current_user.email = request.form.get('email', '').strip() or None
        current_user.notify_email = request.form.get('notify_email', '').strip() or None
        current_user.infer_notify = request.form.get('infer_notify') == '1'
        if new_username and new_username != current_user.username:
            if User.query.filter_by(username=new_username).first():
                flash('用户名已存在', 'danger')
                return render_template('settings.html')
            current_user.username = new_username
        if new:
            if len(new) < 6:
                flash('新密码至少 6 位', 'danger')
                return render_template('settings.html')
            if new != confirm:
                flash('两次密码不一致', 'danger')
                return render_template('settings.html')
            current_user.password_hash = generate_password_hash(new)
            current_user.must_change_password = False
        db.session.commit()
        flash('设置已保存', 'success')
        return redirect(url_for('settings'))
    return render_template('settings.html')


@bp.route('/logout', endpoint='logout')
@login_required
def logout():
    logout_user()
    if get_setting('auth_mode', 'password') == 'ip':
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))
