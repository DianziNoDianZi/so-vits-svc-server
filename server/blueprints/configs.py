"""推理配置蓝图。"""
import json

from flask import Blueprint, render_template, request, redirect, url_for, flash, abort
from flask_login import login_required, current_user

from authorization import can_use_model
from extensions import db
from db_models import DEFAULT_PARAMS, InferenceConfig, Model
from services.quota import usable_models_for

bp = Blueprint('configs', __name__)


def _parse_params(form, base=None):
    params = dict(base) if base else {}
    for key, default in DEFAULT_PARAMS.items():
        val = form.get(key)
        if val is None:
            continue
        if isinstance(default, bool):
            params[key] = val == 'on' or val == '1'
        elif isinstance(default, int):
            try:
                params[key] = int(val)
            except (ValueError, TypeError):
                params[key] = params.get(key, default)
        elif isinstance(default, float):
            try:
                params[key] = float(val)
            except (ValueError, TypeError):
                params[key] = params.get(key, default)
        else:
            params[key] = val
    return params


@bp.route('/configs', endpoint='config_list')
@login_required
def config_list():
    configs = (InferenceConfig.query.filter_by(user_id=current_user.id)
               .order_by(InferenceConfig.created_at.desc()).all())
    return render_template('configs_list.html', configs=configs)


@bp.route('/configs/create', methods=['GET', 'POST'], endpoint='config_create')
@login_required
def config_create():
    models = usable_models_for(current_user)
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        model_id = request.form.get('model_id', type=int)
        if not name or not model_id:
            flash('请填写名称并选择模型', 'danger')
            return render_template('config_create.html', models=models, params=DEFAULT_PARAMS.copy())
        m = db.session.get(Model, model_id)
        if not m or not can_use_model(current_user, m):
            flash('无效或不可用的模型', 'danger')
            return render_template('config_create.html', models=models, params=DEFAULT_PARAMS.copy())
        params = _parse_params(request.form)
        c = InferenceConfig(user_id=current_user.id, model_id=model_id, name=name,
                            params_json=json.dumps(params, ensure_ascii=False))
        db.session.add(c)
        db.session.commit()
        flash('推理配置已创建', 'success')
        return redirect(url_for('config_list'))
    return render_template('config_create.html', models=models, params=DEFAULT_PARAMS.copy())


@bp.route('/configs/<int:config_id>/edit', methods=['GET', 'POST'], endpoint='config_edit')
@login_required
def config_edit(config_id):
    c = InferenceConfig.query.get_or_404(config_id)
    if c.user_id != current_user.id:
        abort(403)
    models = usable_models_for(current_user)
    if request.method == 'POST':
        c.name = request.form.get('name', c.name).strip()
        model_id = request.form.get('model_id', type=int)
        if model_id:
            m = db.session.get(Model, model_id)
            if not m or not can_use_model(current_user, m):
                flash('无效的模型选择', 'danger')
                return render_template('config_create.html', models=models, config=c,
                                       params=json.loads(c.params_json) if c.params_json else DEFAULT_PARAMS.copy())
            c.model_id = model_id
        base = json.loads(c.params_json) if c.params_json else {}
        c.params_json = json.dumps(_parse_params(request.form, base), ensure_ascii=False)
        db.session.commit()
        flash('配置已更新', 'success')
        return redirect(url_for('config_list'))
    params = json.loads(c.params_json) if c.params_json else DEFAULT_PARAMS.copy()
    return render_template('config_create.html', models=models, config=c, params=params)


@bp.route('/configs/<int:config_id>/delete', methods=['POST'], endpoint='config_delete')
@login_required
def config_delete(config_id):
    c = InferenceConfig.query.get_or_404(config_id)
    if c.user_id != current_user.id:
        abort(403)
    db.session.delete(c)
    db.session.commit()
    flash('配置已删除', 'success')
    return redirect(url_for('config_list'))
