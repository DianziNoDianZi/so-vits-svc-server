#!/usr/bin/env python3
"""命令行上传模型到 So-VITS-SVC 服务器（自动处理登录与 CSRF，无需第三方依赖）。

示例:
  python upload_model.py --url http://127.0.0.1:5000 --user admin --password xxx \
      --model G_4000.pth --config config_G_4000.json --name Kei
"""
import argparse
import http.cookiejar
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
import uuid


def get_csrf(html):
    m = re.search(r'name="_csrf_token"\s+value="([^"]+)"', html or '')
    return m.group(1) if m else None


def build_multipart_body(fields, files):
    """把表单字段和文件写进 multipart 临时文件，返回 (fileobj, content_type, length)。"""
    boundary = '----SoVITSUpload' + uuid.uuid4().hex
    body = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)  # 超过 64MB 自动落盘
    length = 0

    def write(data):
        nonlocal length
        if isinstance(data, str):
            data = data.encode('utf-8')
        body.write(data)
        length += len(data)

    for k, v in fields.items():
        write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n')
    for k, (filename, path) in files.items():
        write(f'--{boundary}\r\n')
        write(f'Content-Disposition: form-data; name="{k}"; filename="{os.path.basename(filename)}"\r\n')
        write('Content-Type: application/octet-stream\r\n\r\n')
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                body.write(chunk)
                length += len(chunk)
        write('\r\n')
    write(f'--{boundary}--\r\n')

    body.seek(0)
    body.length = length  # urllib 通过该属性设置 Content-Length
    return body, f'multipart/form-data; boundary={boundary}'


def main():
    ap = argparse.ArgumentParser(description='Upload a trained model to So-VITS-SVC server')
    ap.add_argument('--url', default='http://127.0.0.1:5000', help='server base URL')
    ap.add_argument('--user', default='admin', help='login username')
    ap.add_argument('--password', required=True, help='login password')
    ap.add_argument('--model', required=True, help='G_*.pth model file path')
    ap.add_argument('--config', required=True, help='config.json path')
    ap.add_argument('--diff-model', default='', help='optional diffusion model .pt path')
    ap.add_argument('--diff-config', default='', help='optional diffusion config .yaml path')
    ap.add_argument('--cluster', default='', help='optional cluster model .pt path')
    ap.add_argument('--name', default='', help='model display name (default: model file name)')
    ap.add_argument('--timeout', type=int, default=600, help='upload timeout seconds')
    args = ap.parse_args()

    for label, path in (('--model', args.model), ('--config', args.config)):
        if not os.path.isfile(path):
            print(f'文件不存在: {label} {path}', file=sys.stderr)
            sys.exit(1)

    base = args.url.rstrip('/')
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    # 1) 登录页拿 CSRF token
    with opener.open(base + '/login', timeout=args.timeout) as r:
        csrf = get_csrf(r.read().decode('utf-8', 'replace'))
    if not csrf:
        print('无法获取登录页 CSRF token，请确认服务器地址正确', file=sys.stderr)
        sys.exit(1)

    # 2) 登录
    data = urllib.parse.urlencode({
        'username': args.user, 'password': args.password, '_csrf_token': csrf,
    }).encode('utf-8')
    with opener.open(base + '/login', data=data, timeout=args.timeout) as r:
        final_url = r.geturl()
    if '/login' in final_url:
        print('登录失败：用户名或密码错误', file=sys.stderr)
        sys.exit(1)

    # 3) 上传页再拿一次 CSRF（同一 session，token 不变）
    with opener.open(base + '/models/upload', timeout=args.timeout) as r:
        csrf = get_csrf(r.read().decode('utf-8', 'replace'))
    if not csrf:
        print('无法获取上传页 CSRF token', file=sys.stderr)
        sys.exit(1)

    name = args.name or os.path.splitext(os.path.basename(args.model))[0]
    fields = {'name': name, '_csrf_token': csrf}
    files = {
        'model_file': (args.model, args.model),
        'config_file': (args.config, args.config),
    }
    if args.diff_model:
        files['diff_file'] = (args.diff_model, args.diff_model)
    if args.diff_config:
        files['diff_config_file'] = (args.diff_config, args.diff_config)
    if args.cluster:
        files['cluster_file'] = (args.cluster, args.cluster)

    print(f'上传中: {name} ({args.model}) ...')
    body, ctype = build_multipart_body(fields, files)
    req = urllib.request.Request(base + '/models/upload', data=body, method='POST', headers={
        'Content-Type': ctype,
        'Content-Length': str(body.length),
    })
    with opener.open(req, timeout=args.timeout) as r:
        final_url = r.geturl()

    if '/models' in final_url:
        print('上传成功！')
        print(f'模型: {name}')
        print(f'查看: {base}/models')
    else:
        print('上传失败（服务器未跳转到模型列表），请检查输出', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
