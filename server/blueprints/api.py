"""对外 REST API（Bearer token 鉴权）。实现见后续端点。"""
from flask import Blueprint

bp = Blueprint('api', __name__, url_prefix='/api/v1')
