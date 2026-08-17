# 实时变声 WebSocket 接口文档

客户端（电脑/手机）采集麦克风音频 → WebSocket 上传 → 服务器用已训模型实时变声 → 回传变声后音频 → 客户端播放。

- **服务器**：独立 WS 服务，端口 `5001`（HTTP 服务仍为 `5000`）
- **延迟定位**：准实时（约 1~2s，CPU 推理吞吐上限，非协议保证）

## 连接

```
ws://<host>:5001/api/v1/ws/stream?api_key=<KEY>
wss://<host>:5001/api/v1/ws/stream?api_key=<KEY>   # 有 TLS 时
```

- **鉴权**：`api_key` 用你在设置页"开发者 API"生成的 Key（查询参数）。
  浏览器 WebSocket 不能自定义 header，query 是唯一可行方式；原生客户端可改用 `X-API-Key` 请求头。
- 鉴权失败：收到 `{"op":"error","error":"unauthorized"}` 后连接关闭。
- 账号被禁用：`{"op":"error","error":"disabled"}`。

## 消息类型

连接建立后，**客户端必须先发 `init` 帧**，服务器回 `ready` 后才开始收发音频。

### 控制帧（文本 JSON）

**1. init（客户端 → 服务器）**

```json
{
  "op": "init",
  "config_id": 1,
  "speaker": "可选，说话人；缺省用配置第一个",
  "chunk_seconds": 0.363,
  "tran": 0,
  "auto_predict_f0": false,
  "noice_scale": 0.4,
  "f0_predictor": "pm",
  "k_step": 0,
  "cluster_ratio": 0
}
```

| 字段 | 说明 | 默认 |
|------|------|------|
| `config_id` | 必填，你的推理配置 ID（`GET /api/v1/configs` 可查） | — |
| `chunk_seconds` | 切片长度，0.1~2.0s。越小延迟越低但块间衔接/音质变差；越大延迟越高越稳 | 0.363 |
| `tran` | 变调（半音） | 0 |
| `auto_predict_f0` | 自动预测 F0 | false |
| `noice_scale` | 生成噪声 | 0.4 |
| `f0_predictor` | `pm`/`rmvpe`/`harvest`/`dio`/`crepe` | `pm` |
| `k_step` | 浅扩散步数（挂了扩散模型时） | 0 |
| `cluster_ratio` | 特征检索比例 | 0 |

服务器成功返回 `{"op":"ready"}`。

**2. 心跳**：`{"op":"ping"}` → `{"op":"pong"}`（另有 WebSocket 层原生 ping/pong）。

**3. 结束**：`{"op":"stop"}` 或 `{"op":"close"}`。

**4. 暂停/恢复**：`{"op":"pause"}` / `{"op":"resume"}`（预留，当前仅 stop）。

### 音频帧（二进制）

- **上行**（客户端 → 服务器）：**44.1kHz 单声道 float32** 的 PCM 原始字节（可发任意大小，建议每 20~40ms 一块，即 882~1764 字节）。
- **下行**（服务器 → 客户端）：**44.1kHz 单声道 float32** 变声后 PCM 原始字节。服务器攒满一个窗口（`chunk_seconds` ≈ 默认 363ms）后回发，每帧约 16000 字节。

服务器未攒满窗口时**不下发**（客户端持续推流即可，无需请求-响应配对）。

## 会话限制

- 全局同一时刻仅 1 个流式会话，第二个连接收到 `{"op":"error","error":"busy"}`。
- 无音频超过 **60s** 断开：`{"op":"close","reason":"idle"}`。
- 单会话最长 **30 分钟**：`{"op":"close","reason":"timeout"}`。

## 客户端示例（Python websockets）

```python
import asyncio, json, struct, websockets

async def main():
    async with websockets.connect(
        "ws://127.0.0.1:5001/api/v1/ws/stream?api_key=YOUR_KEY"
    ) as ws:
        await ws.send(json.dumps({"op": "init", "config_id": 1}))
        print(await ws.recv())  # {"op":"ready"}

        # 从麦克风读 44.1k float32 块 → ws.send(bytes)；
        # 循环接收变声帧 → 写扬声器
        async for frame in ws:
            if isinstance(frame, bytes):
                play(frame)  # 44.1k float32 PCM
            else:
                print(frame)  # 控制帧

asyncio.run(main())
```

## 浏览器示例（JS）

```js
const ws = new WebSocket(`ws://${host}:5001/api/v1/ws/stream?api_key=${key}`);
ws.onopen = () => ws.send(JSON.stringify({op:'init', config_id: 1}));
ws.onmessage = (e) => {
  if (e.data instanceof Blob) { /* 变声 PCM float32 → AudioWorklet/ScriptProcessor 播放 */ }
  else { /* JSON 控制帧 */ }
};
// 麦克风：navigator.mediaDevices.getUserMedia → AudioContext({sampleRate:44100}) → 每 20-40ms 取 Float32Array → ws.send(buffer)
```

## 注意事项

- 上行/下行均为 **44.1kHz**；服务器内部自行做 44.1k→16k 特征提取，客户端无需重采样。
- `chunk_seconds` 越小，服务器推理次数越多，CPU 小机可能跟不上 → 延迟反而升高。CPU 建议 0.3~0.5s；若块间有杂音/爆音，调大它。
- 变声延迟 = 窗口时间 + 推理耗时 + 网络。准实时场景客户端应缓冲 1~2s 平滑播放。
