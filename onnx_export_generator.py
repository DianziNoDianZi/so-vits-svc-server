"""主生成器 ONNX 导出（v1 / rvc / rvc-flow）。

用法：
  python onnx_export_generator.py <model.pth> <config.json> <out.onnx>

把采样噪声作为输入导出（z_p = m_p + noise * exp(logs_p)），推理时由外部生成 noise，
避免 torch.randn 被固化。编码器 / F0 / 扩散暂保持 PyTorch（后续单独 ONNX 化）。
"""
import argparse
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
import modules.commons as commons
from models import SynthesizerTrn, SynthesizerTrnRvc, SynthesizerTrnRvcFlow


class GeneratorOnnx(nn.Module):
    """导出用包装：采样噪声作为输入（z_p = m_p + noise * exp(logs_p)），
    推理时由外部生成 noise，避免 randn 被固化。"""

    def __init__(self, net_g):
        super().__init__()
        self.net_g = net_g
        self.arch = getattr(net_g, 'arch_name', 'v1')
        self.flow_mode = getattr(net_g, 'flow_mode', '')
        self.inter = net_g.inter_channels
        # 方案3统一流：导出时按固定 hybrid_steps 展开 FM 欧拉积分
        self.use_unified = getattr(net_g, 'use_unified_flow', False)
        self.hybrid_steps = getattr(net_g, 'hybrid_steps', 4)

    def _emb_g(self, g):
        if g.dim() == 1:
            g = g.unsqueeze(0)
        return self.net_g.emb_g(g).transpose(1, 2)

    def forward(self, c, f0, uv, g, noise):
        # 注：rvc / rvc-flow-A1 无采样，noise 不参与图；vol_embedding=False 时 vol 为 0
        net = self.net_g
        c_lengths = torch.ones(c.size(0), dtype=torch.long).to(c.device) * c.size(-1)
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        gg = self._emb_g(g)
        vv = 0
        x = net.pre(c) * x_mask + net.emb_uv(uv.long()).transpose(1, 2) + vv
        if self.arch == 'rvc':
            return net.dec(x, g=gg, f0=f0)
        if self.arch == 'rvc-flow':
            if self.flow_mode == 'a2':
                stats = net.prior_proj(x) * x_mask
                m_p, logs_p = torch.split(stats, self.inter, dim=1)
                z_p = m_p + noise * torch.exp(logs_p) * 0.35
                if self.use_unified:
                    # 方案3 hybrid：NF 逆变换(mode='nf')给出起点，FM 欧拉积分精修
                    # mode='nf' 返回 (x, logdet)，取 x
                    x_t = net.flow(z_p, x_mask, g=gg, mode='nf', reverse=True)[0]
                    dt = 1.0 / self.hybrid_steps
                    for i in range(self.hybrid_steps):
                        # t 用 [1,1] 常量，靠广播适配任意 batch（ONNX 友好）
                        t_val = float(i) / self.hybrid_steps
                        t = torch.full((1, 1), t_val, device=x_t.device)
                        v = net.flow(x_t, x_mask, g=gg, t=t, mode='fm')
                        x_t = x_t + v * dt
                    z = x_t
                else:
                    z = net.flow(z_p, x_mask, g=gg, reverse=True)
            else:
                z = net.flow(x, x_mask, g=gg)
            return net.dec(z, g=gg, f0=f0)
        # v1：TextEncoder 手工展开（采样噪声外部传入）
        x = x + net.enc_p.f0_emb(utils.f0_to_coarse(f0)).transpose(1, 2)
        x = net.enc_p.enc_(x * x_mask, x_mask)
        stats = net.enc_p.proj(x) * x_mask
        m_p, logs_p = torch.split(stats, net.enc_p.out_channels, dim=1)
        z_p = m_p + noise * torch.exp(logs_p) * 0.35
        z = net.flow(z_p, x_mask, g=gg, reverse=True)
        return net.dec(z * x_mask, g=gg, f0=f0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='G_*.pth path')
    parser.add_argument('config', help='config.json path')
    parser.add_argument('out', help='output .onnx path')
    parser.add_argument('--num_frames', type=int, default=200, help='example frame count')
    args = parser.parse_args()

    device = torch.device('cpu')
    hps = utils.get_hparams_from_file(args.config, True)
    arch = hps.model.get('arch') or 'sovits-v1'
    if arch == 'rvc':
        cls = SynthesizerTrnRvc
    elif arch == 'rvc-flow':
        cls = SynthesizerTrnRvcFlow
    else:
        cls = SynthesizerTrn
    net_g = cls(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model)
    utils.load_checkpoint(args.model, net_g, None, use_ema=True)
    net_g.eval().to(device)
    # 解码器切换到 ONNX 模式（谐波源/上采样走可 trace 路径）
    if hasattr(net_g, 'dec') and hasattr(net_g.dec, 'OnnxExport'):
        net_g.dec.OnnxExport()
    for p in net_g.parameters():
        p.requires_grad = False

    infer = GeneratorOnnx(net_g)
    nf = args.num_frames
    ssl_dim = net_g.ssl_dim
    inter = net_g.inter_channels
    c = torch.randn(1, ssl_dim, nf)
    f0 = torch.rand(1, nf) * 200 + 80
    uv = torch.ones(1, nf)
    g = torch.zeros(1, 1, dtype=torch.long)
    # 统一传 noise：rvc / rvc-flow-A1 不用时会被 torch.onnx 自动消除（不产生无用输入）
    sample_inputs = (c, f0, uv, g, torch.randn(1, inter, nf))
    names = ['c', 'f0', 'uv', 'g', 'noise']
    daxes = {
        'c': {0: 'batch', 2: 'frames'},
        'f0': {0: 'batch', 1: 'frames'},
        'uv': {0: 'batch', 1: 'frames'},
        'noise': {0: 'batch', 2: 'frames'},
        'audio': {0: 'batch', 2: 'samples'},
    }

    torch.onnx.export(
        infer, sample_inputs, args.out,
        input_names=names,
        output_names=['audio'],
        dynamic_axes=daxes,
        opset_version=17,
    )
    print(f'exported {arch} -> {args.out}')


if __name__ == '__main__':
    main()
