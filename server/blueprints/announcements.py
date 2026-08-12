"""用户公告页蓝图。"""
from flask import Blueprint, render_template
from flask_login import login_required

from db_models import Announcement

bp = Blueprint('announcements', __name__)


@bp.route('/announcements', endpoint='announcements')
@login_required
def announcements():
    items = (Announcement.query.filter_by(is_active=True)
             .order_by(Announcement.is_pinned.desc(), Announcement.created_at.desc()).all())
    return render_template('announcements.html', items=items)
