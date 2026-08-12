"""模型配置基础校验。"""
from apputils import read_model_cfg

_REQUIRED_MODEL_FIELDS = ('ssl_dim', 'n_speakers', 'vocab_size', 'n_layers')
_VALID_ARCHS = ('sovits-v1', 'rvc', 'rvc-flow')


def check_model_config(model):
    """解析模型 config.json 做基础兼容性检查，返回 (ok, issues[])。"""
    cfg = read_model_cfg(model.config_path)
    if not cfg:
        return False, ['配置文件缺失或无法解析']
    m = cfg.get('model') or {}
    issues = []
    for f in _REQUIRED_MODEL_FIELDS:
        if f not in m:
            issues.append(f'model 缺字段 {f}')
    arch = m.get('arch', 'sovits-v1')
    if arch not in _VALID_ARCHS:
        issues.append(f'未知架构 {arch}')
    if not cfg.get('train'):
        issues.append('缺少 train 配置')
    if not cfg.get('data'):
        issues.append('缺少 data 配置')
    return (len(issues) == 0, issues)
