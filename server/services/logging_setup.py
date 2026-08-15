"""结构化日志：RotatingFileHandler 按大小轮转，写到 server/logs/。"""
import logging
import os
from logging.handlers import RotatingFileHandler

from flask import has_request_context, request

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5MB per file
LOG_BACKUP_COUNT = 5


class _RequestFormatter(logging.Formatter):
    def format(self, record):
        if has_request_context():
            record.request_info = f' [{request.remote_addr} {request.method} {request.path}]'
        else:
            record.request_info = ''
        return super().format(record)


def setup_logging(app):
    os.makedirs(LOG_DIR, exist_ok=True)
    handler = RotatingFileHandler(
        os.path.join(LOG_DIR, 'app.log'),
        maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding='utf-8',
    )
    handler.setFormatter(_RequestFormatter(
        '%(asctime)s %(levelname)s %(name)s%(request_info)s %(message)s'
    ))
    handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    app.logger.setLevel(logging.INFO)
    app.logger.info('服务启动，结构化日志已开启 (logs/app.log)')
