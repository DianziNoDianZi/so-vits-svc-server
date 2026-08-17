"""实时变声 WebSocket 服务：gevent 双端口方案，跑在 :5001。

刻意不 import 顶层 app 模块——那会把 torch（经 scheduler→inference_daemon）
拉进这个进程，3.8G 内存上两个 torch 进程会 OOM。这里只构建最小 Flask app：
Config + db + db_models + api_auth + quota（均不 import torch）。

架构：
  客户端 ──WebSocket :5001──▶ ws_server(gevent, 鉴权, 无模型)
                                  │  AF_UNIX socket
                                  ▼
                         inference_daemon(常驻, 持有 Svc 模型, 流式推理)
"""
import os
import sys
import time
import logging

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SERVER_DIR)
sys.path.insert(0, os.path.dirname(SERVER_DIR))

from config import Config
from extensions import db

SOCK_PATH = os.path.join(SERVER_DIR, 'logs', 'ssvc_stream.sock')
IDLE_TIMEOUT = 60          # 无音频断连（秒）
MAX_SESSION_SECONDS = 1800  # 单会话时长上限（秒）


def build_mini_app():
    from flask import Flask
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    with app.app_context():
        from services.quota import get_setting
        pass
    return app


def _auth_user(app, ws):
    """从 ws 握手 query 或 header 取 api_key，返回 user 或 None。"""
    query = ws.environ.get('QUERY_STRING', '')
    import urllib.parse
    params = urllib.parse.parse_qs(query)
    api_key = (params.get('api_key') or [None])[0]
    if not api_key:
        api_key = ws.environ.get('HTTP_X_API_KEY', '')
    if not api_key:
        return None
    from services.api_auth import hash_token
    from db_models import ApiToken
    with app.app_context():
        t = ApiToken.query.filter_by(token_hash=hash_token(api_key)).first()
        if not t or t.revoked_at:
            return None
        return t.user
    return None


def _build_daemon_payload(user, model, config, params):
    """组装发给 daemon 的模型 payload（与 scheduler.task_worker 字段一致）。"""
    upload = os.path.join(Config.UPLOAD_FOLDER)
    return {
        'model_path': os.path.join(upload, 'models', model.model_path),
        'config_path': os.path.join(upload, 'configs', model.config_path),
        'diff_path': os.path.join(upload, 'models', model.diff_model_path) if model.diff_model_path else 'none',
        'diff_config_path': os.path.join(upload, 'configs', model.diff_config_path) if model.diff_config_path else 'none',
        'cluster_path': os.path.join(upload, 'models', model.cluster_path) if model.cluster_path else 'none',
        'k_step': int(params.get('k_step') or 0),
        'params': params,
        'device': user.device_pref or 'auto',
        'max_cpu_cores': 0,
    }


def handle_ws(app, ws):
    started = time.time()
    with app.app_context():
        user = _auth_user(app, ws)
    if not user:
        try:
            ws.send('{"op":"error","error":"unauthorized"}')
        except Exception:
            pass
        ws.close()
        return
    from services.quota import current_quota
    with app.app_context():
        quota = current_quota(user)
    if not quota.enabled:
        try:
            ws.send('{"op":"error","error":"disabled"}')
        except Exception:
            pass
        ws.close()
        return

    # 等待客户端发 init 帧（含 config_id + 流式参数）
    try:
        init_msg = ws.receive()
    except Exception:
        ws.close()
        return
    if not init_msg:
        ws.close()
        return
    import json as _json
    try:
        init = _json.loads(init_msg)
    except Exception:
        try:
            ws.send('{"op":"error","error":"bad init json"}')
        except Exception:
            pass
        ws.close()
        return
    if init.get('op') != 'init':
        try:
            ws.send('{"op":"error","error":"expected init"}')
        except Exception:
            pass
        ws.close()
        return

    # 校验 config/model 归属
    from db_models import InferenceConfig, Model
    from authorization import can_use_model
    config_id = init.get('config_id')
    with app.app_context():
        cfg = db.session.get(InferenceConfig, config_id) if config_id else None
        if not cfg or cfg.user_id != user.id:
            try:
                ws.send('{"op":"error","error":"bad config"}')
            except Exception:
                pass
            ws.close()
            return
        model = db.session.get(Model, cfg.model_id)
        if not model or not can_use_model(user, model):
            try:
                ws.send('{"op":"error","error":"model unavailable"}')
            except Exception:
                pass
            ws.close()
            return
        import json as _j2
        try:
            params = _j2.loads(cfg.params_json or '{}')
        except Exception:
            params = {}
        # 流式参数合并
        for k in ('chunk_seconds', 'tran', 'auto_predict_f0', 'noice_scale', 'f0_predictor', 'k_step', 'cluster_ratio'):
            if k in init:
                params[k] = init[k]
        payload = _build_daemon_payload(user, model, cfg, params)
        # speaker 可选：不传则由 daemon 取 config 里第一个说话人
        if init.get('speaker'):
            payload['speaker'] = init['speaker']

    # 连 daemon AF_UNIX socket
    from multiprocessing.connection import Client
    try:
        conn = Client(SOCK_PATH, family='AF_UNIX')
    except Exception as e:
        try:
            ws.send('{"op":"error","error":"daemon unavailable: %s"}' % str(e)[:80])
        except Exception:
            pass
        ws.close()
        return
    try:
        conn.send({'op': 'init', 'payload': payload})
        resp = conn.recv()
        if not resp.get('ok'):
            try:
                ws.send('{"op":"error","error":%s}' % _json.dumps(resp.get('error', 'init failed')))
            except Exception:
                pass
            ws.close()
            return
        ws.send('{"op":"ready"}')
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        ws.close()
        return

    # 双向转发循环
    last_activity = time.time()
    try:
        while True:
            if time.time() - last_activity > IDLE_TIMEOUT:
                try:
                    ws.send('{"op":"close","reason":"idle"}')
                except Exception:
                    pass
                break
            if time.time() - started > MAX_SESSION_SECONDS:
                try:
                    ws.send('{"op":"close","reason":"timeout"}')
                except Exception:
                    pass
                break
            msg = ws.receive(timeout=1.0)
            if msg is None:
                continue
            if isinstance(msg, bytes):
                # 二进制 = float32 PCM 44.1k 上行
                try:
                    conn.send({'op': 'infer', 'audio': msg})
                    resp = conn.recv()
                    if resp.get('pending'):
                        continue
                    if resp.get('audio'):
                        ws.send(resp['audio'], binary=True)
                    last_activity = time.time()
                except Exception:
                    break
            else:
                # 文本 JSON 控制
                try:
                    ctrl = _json.loads(msg)
                except Exception:
                    continue
                op = ctrl.get('op')
                if op in ('pause', 'resume'):
                    # 暂停/恢复：当前简化实现只支持 stop
                    pass
                elif op == 'ping':
                    try:
                        ws.send('{"op":"pong"}')
                    except Exception:
                        pass
                elif op == 'stop' or op == 'close':
                    break
                elif op == 'set':
                    # 动态改参：重建 payload 的流式参数并让 daemon 应用（简化：忽略）
                    pass
    except Exception:
        pass
    finally:
        try:
            conn.send({'op': 'close'})
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        try:
            ws.close()
        except Exception:
            pass


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    from gevent.pywsgi import WSGIServer
    from geventwebsocket.handler import WebSocketHandler
    app = build_mini_app()

    def ws_app(environ, start_response):
        if environ.get('PATH_INFO') != '/api/v1/ws/stream':
            start_response('404 Not Found', [('Content-Type', 'text/plain')])
            return [b'not found']
        ws = environ.get('wsgi.websocket')
        if not ws:
            start_response('400 Bad Request', [('Content-Type', 'text/plain')])
            return [b'websocket required']
        handle_ws(app, ws)
        return []

    port = int(os.environ.get('PORT_WS', '5001'))
    server = WSGIServer(('0.0.0.0', port), ws_app, handler_class=WebSocketHandler)
    logging.info(f'ws_server listening on :{port} (sock={SOCK_PATH})')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
