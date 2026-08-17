"""实时变声流式推理：攒满一个窗口 → 拼上窗口前段 → Svc.infer → numpy 交叉淡化。

复用 infer_tool.Svc（已常驻在 daemon），输入 44.1k 单声道 numpy 块，
输出等长的 44.1k 变声窗口。chunk_seconds 可配（0.1~2.0s，默认 0.363s）。
"""
import io
import numpy as np
import soundfile as sf


def _crossfade(a, b, fade_len):
    """对 a 尾部 fade_len 与 b 头部做线性交叉淡化，返回等长 b。"""
    fade_len = max(1, int(fade_len))
    if len(a) < fade_len or len(b) < fade_len:
        return b
    ramp = np.linspace(0, 1, fade_len)
    b_head = b[:fade_len] * ramp
    a_tail = a[-fade_len:] * (1 - ramp)
    out = b.copy()
    out[:fade_len] = b_head + a_tail
    return out


class StreamingVC:
    """单会话流式变声状态机。每个 ws 会话一个实例，维护 last_chunk / last_o。"""

    def __init__(self, sample_rate=44100, chunk_seconds=0.363, crossfade_ratio=0.24):
        self.sample_rate = int(sample_rate)
        self.chunk_len = max(int(chunk_seconds * self.sample_rate), 1000)
        self.pre_len = max(int(self.chunk_len * crossfade_ratio), 320)
        self._buf = np.zeros(0, dtype=np.float32)
        self.last_chunk = None
        self.last_o = None

    def feed(self, audio):
        """接收一块 44.1k 单声道 numpy float32，攒满一个窗口后返回变声结果。

        返回 (output_audio: np.ndarray float32 44.1k, 或 None 未满)。
        """
        self._buf = np.concatenate([self._buf, np.asarray(audio, dtype=np.float32)])
        if len(self._buf) < self.chunk_len:
            return None
        window = self._buf[:self.chunk_len]
        self._buf = self._buf[self.chunk_len:]
        return self._process_window(window)

    def _process_window(self, window):
        raise NotImplementedError('由 daemon 内绑定 Svc 的子类实现')


class DaemonStreamingVC(StreamingVC):
    """绑定 Svc 模型实例的流式变声。"""

    def __init__(self, svc, speaker, tran=0, auto_predict_f0=False, noice_scale=0.4,
                 f0_predictor='pm', k_step=0, cluster_infer_ratio=0, chunk_seconds=0.363,
                 sample_rate=None, **kwargs):
        sr = int(svc.target_sample) if sample_rate is None else int(sample_rate)
        super().__init__(sample_rate=sr, chunk_seconds=chunk_seconds)
        self.svc = svc
        self.speaker = speaker
        self.tran = tran
        self.auto_predict_f0 = auto_predict_f0
        self.noice_scale = noice_scale
        self.f0_predictor = f0_predictor
        self.k_step = k_step
        self.cluster_infer_ratio = cluster_infer_ratio

    def _wav_bytes(self, audio):
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format='wav', subtype='PCM_16')
        buf.seek(0)
        return buf

    def _infer(self, wav_buf):
        out_audio, _out_sr, _out_frame = self.svc.infer(
            self.speaker, self.tran, wav_buf,
            cluster_infer_ratio=self.cluster_infer_ratio,
            auto_predict_f0=self.auto_predict_f0,
            noice_scale=self.noice_scale,
            f0_predictor=self.f0_predictor,
            k_step=self.k_step,
        )
        return np.asarray(out_audio.cpu().numpy(), dtype=np.float32)

    def _process_window(self, window):
        if self.last_chunk is None:
            # 首个窗口：直接推理，输出尾部作为窗口
            out = self._infer(self._wav_bytes(window))
            self.last_chunk = out[-self.pre_len:]
            self.last_o = out
            return out[-self.chunk_len:]
        else:
            # 拼前窗口尾部，整体重推，交叉淡化衔接
            combined = np.concatenate([self.last_chunk, window])
            out = self._infer(self._wav_bytes(combined))
            merged = _crossfade(self.last_o, out, self.pre_len)
            self.last_chunk = out[-self.pre_len:]
            self.last_o = out
            return merged[self.chunk_len:2 * self.chunk_len]
