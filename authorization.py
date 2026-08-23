from __future__ import annotations


def is_admin(user) -> bool:
    return bool(user and getattr(user, 'role', 'user') == 'admin')


def is_active_user(user) -> bool:
    if not user:
        return False
    val = getattr(user, 'is_active', None)
    # 旧库历史行 is_active 可能为 NULL，按“未禁用”处理
    return True if val is None else bool(val)


def can_manage_model(user, model) -> bool:
    if not user or not model:
        return False
    if is_admin(user):
        return True
    return getattr(model, 'user_id', None) == getattr(user, 'id', None)


def can_use_model(user, model) -> bool:
    if not user or not model:
        return False
    if is_admin(user):
        return getattr(model, 'status', 'ready') not in ('disabled', 'rejected')
    if getattr(user, 'role', 'user') == 'guest':
        return (getattr(model, 'status', 'ready') == 'ready'
                and getattr(model, 'visibility', 'private') == 'official')
    if getattr(model, 'status', 'ready') != 'ready':
        return False
    visibility = getattr(model, 'visibility', 'private')
    if visibility == 'official':
        return True
    return getattr(model, 'user_id', None) == getattr(user, 'id', None)


def can_use_config(user, config, model) -> bool:
    if not user or not config or not model:
        return False
    if getattr(config, 'user_id', None) != getattr(user, 'id', None):
        return False
    return can_use_model(user, model)
