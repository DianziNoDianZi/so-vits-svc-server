"""模型蓝图：列表/上传/编辑/删除/申请公开/审核/导出ONNX。"""
import os
import sys
from datetime import datetime

from flask import Blueprint, render_template, request, redirect, url_for, flash, abort
from flask_login import login_required, current_user

from authorization import can_manage_model, is_admin
from extensions import db
from db_models import InferenceConfig, Model
from services.quota import current_quota, usable_models_for
from services.validation import check_model_config
from apputils import allowed_file, save_uploaded, arch_label_from_cfg, read_model_cfg

bp = Blueprint('models', __name__)


@bp.route('/models', endpoint='model_list')
@login_required
def model_list():
    models = usable_models_for(current_user)
    model_items = []
    for m in models:
        cfg = read_model_cfg(m.config_path)
        arch = (cfg.get('model') or {}).get('arch', '')
        label, sub = arch_label_from_cfg(cfg)
        onnx_exists = False
        if m.model_path:
            onnx_exists = os.path.exists(os.path.join(
                current_upload(), 'models', m.model_path + '.onnx'))
        model_items.append({
            'm': m, 'arch': arch or 'sovits-v1', 'arch_label': label, 'flow_mode': sub,
            'status': getattr(m, 'status', 'ready'), 'visibility': getattr(m, 'visibility', 'private'),
            'tags': [t.strip() for t in (m.tags or '').split(',') if t.strip()],
            'onnx': onnx_exists,
            'can_manage': can_manage_model(current_user, m),
        })
    return render_template('models_list.html', model_items=model_items)


def current_upload():
    from flask import current_app
    return current_app.config['UPLOAD_FOLDER']


@bp.route('/models/upload', methods=['GET', 'POST'], endpoint='model_upload')
@login_required
def model_upload():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        model_file = request.files.get('model_file')
        config_file = request.files.get('config_file')
        diff_file = request.files.get('diff_file')
        diff_config_file = request.files.get('diff_config_file')
        cluster_file = request.files.get('cluster_file')
        if not name:
            flash('请输入模型名称', 'danger')
            return render_template('model_upload.html')
        if not model_file or not model_file.filename or not config_file or not config_file.filename:
            flash('请选择模型文件 (.pth) 和配置文件 (.json)', 'danger')
            return render_template('model_upload.html')
        if not allowed_file(model_file.filename) or not allowed_file(config_file.filename):
            flash('文件类型不允许：仅支持 .pth/.pt/.json/.yaml/.yml', 'danger')
            return render_template('model_upload.html')
        for f, label in ((diff_file, '扩散模型'), (diff_config_file, '扩散配置'), (cluster_file, '聚类模型')):
            if f and f.filename and not allowed_file(f.filename):
                flash(f'{label}文件类型不允许', 'danger')
                return render_template('model_upload.html')

        quota = current_quota(current_user)
        if not quota.enabled:
            flash('当前账号已被禁用，无法上传模型', 'danger')
            return render_template('model_upload.html')
        private_cnt = Model.query.filter(
            Model.user_id == current_user.id, Model.visibility == 'private',
            Model.status.in_(['pending_review', 'ready'])).count()
        if private_cnt >= quota.max_private_models:
            flash('已达到私有模型数量上限（被拒绝的模型不占名额）', 'danger')
            return render_template('model_upload.html')

        model_path = save_uploaded(model_file, 'models')
        config_path = save_uploaded(config_file, 'configs')
        diff_path = save_uploaded(diff_file, 'models') if diff_file and diff_file.filename else None
        diff_config_path = save_uploaded(diff_config_file, 'configs') if diff_config_file and diff_config_file.filename else None
        cluster_path = save_uploaded(cluster_file, 'models') if cluster_file and cluster_file.filename else None

        is_official = is_admin(current_user) and request.form.get('visibility') == 'official'
        m = Model(user_id=current_user.id, name=name,
                  visibility='official' if is_official else 'private',
                  status='ready' if is_official else 'pending_review',
                  model_path=model_path, config_path=config_path,
                  diff_model_path=diff_path, diff_config_path=diff_config_path, cluster_path=cluster_path)
        db.session.add(m)
        db.session.commit()
        flash('模型已上传，等待管理员审核' if not is_official else '官方模型已发布', 'success')
        return redirect(url_for('model_list'))
    return render_template('model_upload.html')


@bp.route('/models/<int:model_id>/edit', methods=['GET', 'POST'], endpoint='model_edit')
@login_required
def model_edit(model_id):
    m = Model.query.get_or_404(model_id)
    if not can_manage_model(current_user, m):
        abort(403)
    models_dir = os.path.join(current_upload(), 'models')
    configs_dir = os.path.join(current_upload(), 'configs')
    diff_files, cluster_files, diff_configs = [], [], []
    try:
        for f in sorted(os.listdir(models_dir)):
            if f.lower().endswith('.pt'):
                (cluster_files if 'kmeans' in f.lower() or 'cluster' in f.lower() else diff_files).append(f)
    except OSError:
        pass
    try:
        diff_configs = [f for f in sorted(os.listdir(configs_dir)) if f.lower().endswith(('.yaml', '.yml'))]
    except OSError:
        pass
    if request.method == 'POST':
        m.name = request.form.get('name', m.name).strip()
        m.tags = request.form.get('tags', '').strip() or None
        diff_file = request.files.get('diff_file')
        diff_config_file = request.files.get('diff_config_file')
        cluster_file = request.files.get('cluster_file')
        for f, label in ((diff_file, '扩散模型'), (diff_config_file, '扩散配置'), (cluster_file, '聚类模型')):
            if f and f.filename and not allowed_file(f.filename):
                flash(f'{label}文件类型不允许', 'danger')
                return render_template('model_edit.html', model=m, diff_files=diff_files,
                                       diff_configs=diff_configs, cluster_files=cluster_files)
        sel_diff = request.form.get('diff_model_select', '').strip()
        sel_diff_cfg = request.form.get('diff_config_select', '').strip()
        sel_cluster = request.form.get('cluster_select', '').strip()
        if sel_diff == '__clear__':
            m.diff_model_path = None
        elif sel_diff:
            if os.path.exists(os.path.join(models_dir, sel_diff)):
                m.diff_model_path = sel_diff
            else:
                flash('所选扩散模型文件不存在', 'danger')
        elif diff_file and diff_file.filename:
            m.diff_model_path = save_uploaded(diff_file, 'models')
        if sel_diff_cfg == '__clear__':
            m.diff_config_path = None
        elif sel_diff_cfg:
            if os.path.exists(os.path.join(configs_dir, sel_diff_cfg)):
                m.diff_config_path = sel_diff_cfg
            else:
                flash('所选扩散配置文件不存在', 'danger')
        elif diff_config_file and diff_config_file.filename:
            m.diff_config_path = save_uploaded(diff_config_file, 'configs')
        if sel_cluster == '__clear__':
            m.cluster_path = None
        elif sel_cluster:
            if os.path.exists(os.path.join(models_dir, sel_cluster)):
                m.cluster_path = sel_cluster
            else:
                flash('所选聚类模型文件不存在', 'danger')
        elif cluster_file and cluster_file.filename:
            m.cluster_path = save_uploaded(cluster_file, 'models')
        db.session.commit()
        flash('模型已更新', 'success')
        return redirect(url_for('model_list'))
    return render_template('model_edit.html', model=m, diff_files=diff_files,
                           diff_configs=diff_configs, cluster_files=cluster_files)


def delete_model_resources(m):
    # 审核不通过=整包删掉，别留着占地方。这也是用户明确要的：
    # "不通过则会删除"，那就删干净，文件、配置、关联推理配置一起带走。
    InferenceConfig.query.filter_by(model_id=m.id).delete()
    for attr in ['model_path', 'config_path', 'diff_model_path', 'diff_config_path', 'cluster_path']:
        path = getattr(m, attr)
        if path:
            full = os.path.join(current_upload(), 'models' if path.endswith(('.pth', '.pt')) else 'configs', path)
            if os.path.exists(full):
                try:
                    os.remove(full)
                except OSError:
                    pass
    db.session.delete(m)


@bp.route('/models/<int:model_id>/delete', methods=['POST'], endpoint='model_delete')
@login_required
def model_delete(model_id):
    m = Model.query.get_or_404(model_id)
    if not can_manage_model(current_user, m):
        abort(403)
    delete_model_resources(m)
    db.session.commit()
    flash('模型已删除', 'success')
    return redirect(url_for('model_list'))


@bp.route('/models/<int:model_id>/request_public', methods=['POST'], endpoint='model_request_public')
@login_required
def model_request_public(model_id):
    m = Model.query.get_or_404(model_id)
    if m.user_id != current_user.id:
        abort(403)
    if m.visibility != 'private' or m.status != 'ready':
        flash('仅可对已审核通过的私有模型申请公开', 'warning')
        return redirect(url_for('model_list'))
    m.public_requested = True
    db.session.commit()
    flash('已提交公开申请，等待管理员审核', 'success')
    return redirect(url_for('model_list'))


@bp.route('/models/<int:model_id>/review', methods=['POST'], endpoint='model_review')
@login_required
def model_review(model_id):
    if not is_admin(current_user):
        abort(403)
    m = Model.query.get_or_404(model_id)
    action = request.form.get('action', '')
    note = request.form.get('note', '').strip() or None
    if action == 'approve':
        m.status = 'ready'
        if getattr(m, 'public_requested', False):
            m.visibility = 'official'
            m.public_requested = False
        m.review_note = note
        m.reviewed_at = datetime.utcnow()
        ok, issues = check_model_config(m)
        msg = f'模型 #{m.id} 已通过审核' + ('，已设为公开' if m.visibility == 'official' else '')
        if not ok:
            msg += '（警告：' + '；'.join(issues[:2]) + '）'
        flash(msg, 'success')
    elif action == 'reject':
        delete_model_resources(m)
        db.session.commit()
        flash(f'模型 #{m.id} 未通过审核，已删除', 'warning')
        return redirect(url_for('admin_models'))
    elif action == 'disable':
        m.status = 'disabled'
        m.review_note = note or m.review_note
        flash(f'模型 #{m.id} 已下架', 'warning')
    elif action == 'enable':
        m.status = 'ready'
        m.review_note = note or m.review_note
        flash(f'模型 #{m.id} 已上架', 'success')
    elif action == 'official':
        m.status = 'ready'
        m.visibility = 'official'
        m.review_note = note or m.review_note
        m.reviewed_at = datetime.utcnow()
        flash(f'模型 #{m.id} 已发布为官方模型', 'success')
    else:
        abort(400)
    db.session.commit()
    from services.audit import audit_log
    audit_log('model_review', f'模型 #{m.id} {m.name} 操作: {action}，备注: {note or "—"}')
    return redirect(url_for('admin_models'))


@bp.route('/models/<int:model_id>/export_onnx', methods=['POST'], endpoint='model_export_onnx')
@login_required
def model_export_onnx(model_id):
    m = db.session.get(Model, model_id)
    if not m or not can_manage_model(current_user, m):
        abort(404)
    if not m.model_path or not m.config_path:
        flash('模型缺少模型文件或配置，无法导出', 'danger')
        return redirect(url_for('model_list'))
    model_file = os.path.join(current_upload(), 'models', m.model_path)
    config_file = os.path.join(current_upload(), 'configs', m.config_path)
    out_file = model_file + '.onnx'
    if not os.path.exists(model_file) or not os.path.exists(config_file):
        flash('模型或配置文件不存在', 'danger')
        return redirect(url_for('model_list'))
    import subprocess as _sp
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, 'onnx_export_generator.py')
    try:
        r = _sp.run([sys.executable, '-X', 'utf8', script, model_file, config_file, out_file],
                    cwd=root, capture_output=True, text=True, timeout=900)
        if r.returncode == 0 and os.path.exists(out_file):
            flash(f'ONNX 导出成功（{os.path.getsize(out_file) / 1048576:.0f}MB），推理将自动使用', 'success')
        else:
            flash(f'ONNX 导出失败：{(r.stderr or r.stdout or "")[-400:]}', 'danger')
    except Exception as e:
        flash(f'ONNX 导出异常：{e}', 'danger')
    return redirect(url_for('model_list'))
