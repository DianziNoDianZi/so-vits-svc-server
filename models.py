import torch
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

import modules.attentions as attentions
import modules.commons as commons
import modules.modules as modules
import utils
from modules.commons import get_padding
from utils import f0_to_coarse


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0,
                 share_parameter=False
                 ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=gin_channels) if share_parameter else None

        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                              gin_channels=gin_channels, mean_only=True, wn_sharing_parameter=self.wn))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

class TransformerCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 n_flows=4,
                 gin_channels=0,
                 share_parameter=False
                 ):
            
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = attentions.FFT(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, isflow = True, gin_channels = self.gin_channels) if share_parameter else None

        for i in range(n_flows):
            self.flows.append(
                modules.TransformerCouplingLayer(channels, hidden_channels, kernel_size, n_layers, n_heads, p_dropout, filter_channels, mean_only=True, wn_sharing_parameter=self.wn, gin_channels = self.gin_channels))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class GeneralizedCouplingLayer(nn.Module):
    """
    统一 NF + FM 的耦合层（方案3）
    共享 FFT 骨干，双输出头：
      - NF 路径：通道拆分 coupling（x0 不变，x1 += shift(x0)），严格可逆
      - FM 路径：速度场看整个 x + 时间嵌入，不需可逆
    """
    def __init__(self, channels, hidden_channels, kernel_size,
                 dilation_rate, n_layers, gin_channels=0,
                 mean_only=True):
        super().__init__()
        assert channels % 2 == 0, "channels should be divisible by 2"
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.gin_channels = gin_channels
        self.mean_only = mean_only
        self.half_channels = channels // 2

        # 共享 FFT 骨干（复用现有 attentions.FFT，isflow=True 以支持 g 条件）
        # FFT 输入维度 = hidden_channels；NF 路径先 pre 投影 half→hidden，FM 路径用 proj_fm 投影 channels→hidden
        self.fft = attentions.FFT(
            hidden_channels, hidden_channels * 4,
            n_heads=2, n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=0, isflow=True,
            gin_channels=gin_channels
        )

        # NF 路径：half_channels -> hidden（pre）和 hidden -> half_channels（post，mean-only shift）
        self.pre_nf = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.post_nf = nn.Conv1d(hidden_channels, self.half_channels, 1)
        self.post_nf.weight.data.zero_()
        self.post_nf.bias.data.zero_()

        # FM 路径：channels -> hidden（pre）和 hidden -> channels（速度场输出）
        # head_fm 零初始化：FM 从恒等映射（v=0）开始，渐进学习速度场
        # 这样 Hybrid 初期 = NF，不会因 FM 随机输出破坏语音
        self.pre_fm = nn.Conv1d(channels, hidden_channels, 1)
        self.head_fm = nn.Conv1d(hidden_channels, channels, 1)
        self.head_fm.weight.data.zero_()
        self.head_fm.bias.data.zero_()

        # 时间嵌入投影（FM 专用）
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

    def forward(self, x, x_mask, g=None, t=None, mode='both', reverse=False):
        """
        Args:
            x: [B, C, L] 输入
            x_mask: [B, 1, L] 掩码
            g: [B, gin_channels, 1] 条件（FFT 内部会 cond_layer 投影）
            t: [B, 1] 时间（FM 专用，NF 模式下为 None）
            mode: 'nf' | 'fm' | 'both'
            reverse: bool，NF 模式下是否逆变换

        Returns:
            mode='nf':   (x_out, logdet)
            mode='fm':   (v_field,)
            mode='both': (x_out, logdet, v_field)
        """
        results = {}

        # NF 路径：通道拆分 coupling，shift 只依赖 x0（不变半），严格可逆
        if mode in ('nf', 'both'):
            x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
            h = self.pre_nf(x0) * x_mask
            h = self.fft(h, x_mask, g=g)
            shift = self.post_nf(h) * x_mask  # mean-only
            if not reverse:
                x1 = x1 + shift
            else:
                x1 = x1 - shift
            x_out = torch.cat([x0, x1], 1)
            logdet = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device)
            results['nf'] = (x_out, logdet)

        # FM 路径：看整个 x + 时间嵌入，输出速度场
        if mode in ('fm', 'both'):
            t_bias = 0
            if t is not None:
                t_emb = self.time_mlp(t)  # [B, hidden]
                t_bias = t_emb.unsqueeze(-1)  # [B, hidden, 1]
            h = self.pre_fm(x) * x_mask
            h = self.fft(h + t_bias * x_mask, x_mask, g=g)
            v_fm = self.head_fm(h) * x_mask
            results['fm'] = v_fm

        # 返回
        if mode == 'nf':
            return results['nf']
        elif mode == 'fm':
            return results['fm']
        else:  # both
            return results['nf'][0], results['nf'][1], results['fm']


class GeneralizedFlow(nn.Module):
    """
    统一 NF + FM 的流容器（方案3）
    包装多个 GeneralizedCouplingLayer + Flip
    """
    def __init__(self, channels, hidden_channels, kernel_size,
                 dilation_rate, n_layers, n_flows=4,
                 gin_channels=0, share_parameter=False):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.gin_channels = gin_channels
        self.n_flows = n_flows

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                GeneralizedCouplingLayer(
                    channels, hidden_channels, kernel_size,
                    dilation_rate, n_layers, gin_channels
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, t=None, mode='both', reverse=False):
        """
        Args:
            x: [B, C, L]
            x_mask: [B, 1, L]
            g: [B, gin_channels, 1]
            t: [B, 1] or None
            mode: 'nf' | 'fm' | 'both'
            reverse: bool

        Returns:
            mode='nf':   (x, logdet)
            mode='fm':   v_field
            mode='both': (x, logdet, v_field)
        """
        logdet_total = 0
        v_accum = 0

        if not reverse:
            for flow in self.flows:
                if isinstance(flow, modules.Flip):
                    # Flip forward 返回 (x, logdet)
                    x, _ = flow(x, x_mask, g=g, reverse=False)
                    continue

                if mode in ('nf', 'both'):
                    x, ld = flow(x, x_mask, g=g, t=t, mode='nf', reverse=False)
                    logdet_total = logdet_total + ld
                if mode in ('fm', 'both'):
                    _, _, v = flow(x, x_mask, g=g, t=t, mode='both', reverse=False)
                    v_accum = v_accum + v
        else:
            for flow in reversed(self.flows):
                if isinstance(flow, modules.Flip):
                    # Flip reverse 只返回 x
                    x = flow(x, x_mask, g=g, reverse=True)
                    continue

                if mode in ('nf', 'both'):
                    x, ld = flow(x, x_mask, g=g, t=t, mode='nf', reverse=True)
                    logdet_total = logdet_total + ld

        if mode == 'nf':
            return x, logdet_total
        elif mode == 'fm':
            return v_accum
        else:
            return x, logdet_total, v_accum


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        # print(x.shape,x_lengths.shape)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class TextEncoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 n_layers,
                 gin_channels=0,
                 filter_channels=None,
                 n_heads=None,
                 p_dropout=None):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.f0_emb = nn.Embedding(256, hidden_channels)

        self.enc_ = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

    def forward(self, x, x_mask, f0=None, noice_scale=1):
        x = x + self.f0_emb(f0).transpose(1, 2)
        x = self.enc_(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noice_scale) * x_mask

        return z, m, logs, x_mask


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames - partial_frames, partial_hop):
            mel_range = torch.arange(i, i + partial_frames)
            mel_slices.append(mel_range)

        return mel_slices

    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:, -partial_frames:]

        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:, s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)

            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            # embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)

        return embed

class F0Decoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 spk_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = attentions.FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        x = torch.detach(x)
        if (spk_emb is not None):
            x = x + self.cond(spk_emb)
        x += self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 gin_channels,
                 ssl_dim,
                 n_speakers,
                 sampling_rate=44100,
                 vol_embedding=False,
                 vocoder_name = "nsf-hifigan",
                 use_depthwise_conv = False,
                 use_automatic_f0_prediction = True,
                 flow_share_parameter = False,
                 n_flow_layer = 4,
                 n_layers_trans_flow = 3,
                 use_transformer_flow = False,
                 **kwargs):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.vol_embedding = vol_embedding
        self.arch_name = 'v1'
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.use_depthwise_conv = use_depthwise_conv
        self.use_automatic_f0_prediction = use_automatic_f0_prediction
        self.n_layers_trans_flow = n_layers_trans_flow
        if vol_embedding:
           self.emb_vol = nn.Linear(1, hidden_channels)

        self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )
        hps = {
            "sampling_rate": sampling_rate,
            "inter_channels": inter_channels,
            "resblock": resblock,
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "gin_channels": gin_channels,
            "use_depthwise_conv":use_depthwise_conv
        }
        
        modules.set_Conv1dModel(self.use_depthwise_conv)

        if vocoder_name == "nsf-hifigan":
            from vdecoder.hifigan.models import Generator
            self.dec = Generator(h=hps)
        elif vocoder_name == "nsf-snake-hifigan":
            from vdecoder.hifiganwithsnake.models import Generator
            self.dec = Generator(h=hps)
        else:
            print("[?] Unkown vocoder: use default(nsf-hifigan)")
            from vdecoder.hifigan.models import Generator
            self.dec = Generator(h=hps)

        self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(inter_channels, hidden_channels, filter_channels, n_heads, n_layers_trans_flow, 5, p_dropout, n_flow_layer,  gin_channels=gin_channels, share_parameter= flow_share_parameter)
        else:
            self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, n_flow_layer, gin_channels=gin_channels, share_parameter= flow_share_parameter)
        if self.use_automatic_f0_prediction:
            self.f0_decoder = F0Decoder(
                1,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                spk_channels=gin_channels
            )
        self.emb_uv = nn.Embedding(2, hidden_channels)
        self.character_mix = False

    def EnableCharacterMix(self, n_speakers_map, device):
        self.speaker_map = torch.zeros((n_speakers_map, 1, 1, self.gin_channels)).to(device)
        for i in range(n_speakers_map):
            self.speaker_map[i] = self.emb_g(torch.LongTensor([[i]]).to(device))
        self.speaker_map = self.speaker_map.unsqueeze(0).to(device)
        self.character_mix = True

    def forward(self, c, f0, uv, spec, g=None, c_lengths=None, spec_lengths=None, vol = None):
        g = self.emb_g(g).transpose(1,2)

        # vol proj
        vol = self.emb_vol(vol[:,:,None]).transpose(1,2) if vol is not None and self.vol_embedding else 0

        # ssl prenet
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1,2) + vol
        
        # f0 predict
        if self.use_automatic_f0_prediction:
            lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
            norm_lf0 = utils.normalize_f0(lf0, x_mask, uv)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
        else:
            lf0 = 0
            norm_lf0 = 0
            pred_lf0 = 0
        # encoder
        z_ptemp, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=f0_to_coarse(f0))
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)

        # flow
        z_p = self.flow(z, spec_mask, g=g)
        z_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(z, f0, spec_lengths, self.segment_size)

        # nsf decoder
        o = self.dec(z_slice, g=g, f0=pitch_slice)

        # 末尾补 loss_flow_match=0，保持与 train.py 的 8 元组解包一致（统一流才非 0）
        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0, 0

    @torch.no_grad()
    def infer(self, c, f0, uv, g=None, noice_scale=0.35, seed=52468, predict_f0=False, vol = None):

        if c.device == torch.device("cuda"):
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        if self.character_mix and len(g) > 1:   # [N, S]  *  [S, B, 1, H]
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
            g = g * self.speaker_map  # [N, S, B, 1, H]
            g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
        else:
            if g.dim() == 1:
                g = g.unsqueeze(0)
            g = self.emb_g(g).transpose(1, 2)
        
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        # vol proj
        
        vol = self.emb_vol(vol[:,:,None]).transpose(1,2) if vol is not None and self.vol_embedding else 0

        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

        
        if self.use_automatic_f0_prediction and predict_f0:
            lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
            norm_lf0 = utils.normalize_f0(lf0, x_mask, uv, random_scale=False)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
            f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
        
        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=f0_to_coarse(f0), noice_scale=noice_scale)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=f0)
        return o,f0


class SynthesizerTrnRvc(nn.Module):
    """
    RVC 式轻量生成器：去掉 TextEncoder / Flow，ContentVec 特征直连解码器。
    与 SynthesizerTrn 保持相同 forward/infer 签名，train.py 调用侧零改动。

    架构: pre(ssl_dim -> inter_channels) + uv 嵌入 + 说话人嵌入
          -> dec(nsf-hifigan, f0 由谐波源注入)
    无 KL / 无 F0 预测（use_automatic_f0_prediction 固定 False）。
    """

    def __init__(self,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 gin_channels,
                 ssl_dim,
                 n_speakers,
                 sampling_rate=44100,
                 vol_embedding=False,
                 vocoder_name="nsf-hifigan",
                 use_depthwise_conv=False,
                 use_automatic_f0_prediction=True,
                 flow_share_parameter=False,
                 n_flow_layer=4,
                 n_layers_trans_flow=3,
                 use_transformer_flow=False,
                 **kwargs):
        super().__init__()
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.vol_embedding = vol_embedding
        self.use_automatic_f0_prediction = False  # RVC 式不预测 F0，直接用输入 F0
        self.character_mix = False
        self.arch_name = 'rvc'

        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.emb_uv = nn.Embedding(2, hidden_channels)
        if vol_embedding:
            self.emb_vol = nn.Linear(1, hidden_channels)

        # 特征直连：ssl_dim -> inter_channels（与解码器 conv_pre 输入通道对齐）
        self.pre = nn.Conv1d(ssl_dim, inter_channels, kernel_size=5, padding=2)

        hps = {
            "sampling_rate": sampling_rate,
            "inter_channels": inter_channels,
            "resblock": resblock,
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "gin_channels": gin_channels,
            "use_depthwise_conv": use_depthwise_conv,
        }
        modules.set_Conv1dModel(use_depthwise_conv)

        if vocoder_name == "nsf-hifigan":
            from vdecoder.hifigan.models import Generator
            self.dec = Generator(h=hps)
        elif vocoder_name == "nsf-snake-hifigan":
            from vdecoder.hifiganwithsnake.models import Generator
            self.dec = Generator(h=hps)
        else:
            print("[?] Unknown vocoder: use default(nsf-hifigan)")
            from vdecoder.hifigan.models import Generator
            self.dec = Generator(h=hps)

    def EnableCharacterMix(self, n_speakers_map, device):
        self.speaker_map = torch.zeros((n_speakers_map, 1, 1, self.gin_channels)).to(device)
        for i in range(n_speakers_map):
            self.speaker_map[i] = self.emb_g(torch.LongTensor([[i]]).to(device))
        self.speaker_map = self.speaker_map.unsqueeze(0).to(device)
        self.character_mix = True

    def forward(self, c, f0, uv, spec, g=None, c_lengths=None, spec_lengths=None, vol=None):
        g = self.emb_g(g).transpose(1, 2)
        vol = self.emb_vol(vol[:, :, None]).transpose(1, 2) if vol is not None and self.vol_embedding else 0

        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

        # 特征与 f0 同步切片（frame 级别，segment_size 为帧数）
        x_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            x, f0, c_lengths, self.segment_size)
        o = self.dec(x_slice, g=g, f0=pitch_slice)

        # 无 flow：占位返回，train.py 中 loss_kl 按 z_p is None 处理为 0
        # 末尾补 loss_flow_match=0，保持 8 元组解包一致
        return o, ids_slice, x_mask, (None, None, None, None, None, None), 0, 0, 0, 0

    @torch.no_grad()
    def infer(self, c, f0, uv, g=None, noice_scale=0.35, seed=52468, predict_f0=False, vol=None):
        if c.device == torch.device("cuda"):
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        if self.character_mix and len(g) > 1:
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))
            g = g * self.speaker_map
            g = torch.sum(g, dim=1)
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0)
        else:
            if g.dim() == 1:
                g = g.unsqueeze(0)
            g = self.emb_g(g).transpose(1, 2)

        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        vol = self.emb_vol(vol[:, :, None]).transpose(1, 2) if vol is not None and self.vol_embedding else 0
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

        o = self.dec(x, g=g, f0=f0)
        return o, f0


class _LightPosteriorEncoder(nn.Module):
    """极轻量后验编码器（rvc-flow A2 用）：1 层 WN，hidden 默认 96，比 v1 的 16 层 Encoder 轻一个量级。"""

    def __init__(self, in_channels, out_channels, hidden_channels=96, kernel_size=5,
                 dilation_rate=1, n_layers=1, gin_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                              gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class SynthesizerTrnRvcFlow(nn.Module):
    """
    RVC 轻量直连 + 轻量 TransformerFlow。

    flow_mode='a1'（特征先验流）：pre(c)+uv → flow 正向 → dec，KL 约束 flow 输出到标准正态先验。
      训练与推理都走 flow 正向（特征信息完整保留，无先验采样），KL 仅作正则。

    flow_mode='a2'（后验流，推荐）：极小 enc_q(spec) 提供后验 → flow 对齐先验；
      先验由 pre 输出经单层卷积投影得到（推理时从先验采样再逆流，内容由特征驱动）。
      A2 有 spec 后验信息，训练比 A1 稳，仍远轻于 v1（enc_q 仅 1 层 WN）。
    """

    def __init__(self,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 gin_channels,
                 ssl_dim,
                 n_speakers,
                 sampling_rate=44100,
                 vol_embedding=False,
                 vocoder_name="nsf-hifigan",
                 use_depthwise_conv=False,
                 use_automatic_f0_prediction=True,
                 flow_share_parameter=False,
                 n_flow_layer=2,
                 n_layers_trans_flow=2,
                 use_transformer_flow=True,
                 flow_mode='a2',
                 enc_q_hidden=96,
                 use_unified_flow=False,
                 hybrid_steps=4,
                 **kwargs):
        super().__init__()
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.vol_embedding = vol_embedding
        self.use_automatic_f0_prediction = False
        self.character_mix = False
        self.arch_name = 'rvc-flow'
        self.flow_mode = flow_mode
        # 方案3：统一 NF+FM 流
        self.use_unified_flow = use_unified_flow
        self.hybrid_steps = hybrid_steps

        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.emb_uv = nn.Embedding(2, hidden_channels)
        if vol_embedding:
            self.emb_vol = nn.Linear(1, hidden_channels)

        # 特征直连：ssl_dim -> inter_channels（与解码器 conv_pre 输入通道对齐）
        self.pre = nn.Conv1d(ssl_dim, inter_channels, kernel_size=5, padding=2)

        hps = {
            "sampling_rate": sampling_rate,
            "inter_channels": inter_channels,
            "resblock": resblock,
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "gin_channels": gin_channels,
            "use_depthwise_conv": use_depthwise_conv,
        }
        modules.set_Conv1dModel(use_depthwise_conv)

        if vocoder_name == "nsf-hifigan":
            from vdecoder.hifigan.models import Generator
            self.dec = Generator(h=hps)
        elif vocoder_name == "nsf-snake-hifigan":
            from vdecoder.hifiganwithsnake.models import Generator
            self.dec = Generator(h=hps)
        else:
            print("[?] Unknown vocoder: use default(nsf-hifigan)")
            from vdecoder.hifigan.models import Generator
            self.dec = Generator(h=hps)

        # 轻量 TransformerFlow（affine coupling + FFT）
        # A1 模式无 enc_q，无法为 FM 提供目标，强制走 TransformerCouplingBlock
        if use_unified_flow and flow_mode == 'a2':
            # 方案3：用 GeneralizedFlow 替换 TransformerCouplingBlock
            # gin_channels 与现有 flow 一致（speaker embedding 通道）
            self.flow = GeneralizedFlow(
                inter_channels, hidden_channels,
                kernel_size=5, dilation_rate=1,
                n_layers=n_layers_trans_flow, n_flows=n_flow_layer,
                gin_channels=gin_channels
            )
        else:
            if use_unified_flow and flow_mode != 'a2':
                print(f'[!] {flow_mode} 模式不支持 unified_flow（A1 无 enc_q 提供 FM 目标），自动使用 TransformerCouplingBlock')
                self.use_unified_flow = False
            self.flow = TransformerCouplingBlock(
                inter_channels, hidden_channels, filter_channels, n_heads,
                n_layers_trans_flow, 5, p_dropout, n_flow_layer,
                gin_channels=gin_channels, share_parameter=flow_share_parameter)

        if flow_mode == 'a2':
            self.enc_q = _LightPosteriorEncoder(
                spec_channels, inter_channels, hidden_channels=enc_q_hidden,
                gin_channels=gin_channels)
            self.prior_proj = nn.Conv1d(inter_channels, inter_channels * 2, 1)
        else:
            # A1（特征先验流）：无 enc_q，固定先验 N(0,1)
            # KL = -0.5 + 0.5*||z_p||^2 约束 flow(x) 输出方差≈1，防止漂移
            # 不用学习先验：flow 和 prior_proj 都看 x 会串谋使 KL→-∞
            self.enc_q = None

    def EnableCharacterMix(self, n_speakers_map, device):
        self.speaker_map = torch.zeros((n_speakers_map, 1, 1, self.gin_channels)).to(device)
        for i in range(n_speakers_map):
            self.speaker_map[i] = self.emb_g(torch.LongTensor([[i]]).to(device))
        self.speaker_map = self.speaker_map.unsqueeze(0).to(device)
        self.character_mix = True

    def forward(self, c, f0, uv, spec, g=None, c_lengths=None, spec_lengths=None, vol=None):
        g = self.emb_g(g).transpose(1, 2)
        vol = self.emb_vol(vol[:, :, None]).transpose(1, 2) if vol is not None and self.vol_embedding else 0
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

        # 方案3 FM loss（默认 0，use_unified_flow+training 时才计算）
        loss_flow_match = 0

        if self.flow_mode == 'a2':
            # 后验：enc_q(spec)；flow 对齐先验；dec 用后验样本（v1 式）
            z_q, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
            if self.use_unified_flow:
                # GeneralizedFlow：NF 模式返回 (x, logdet)，取 x
                z_p, _ = self.flow(z_q, spec_mask, g=g, mode='nf', reverse=False)
            else:
                z_p = self.flow(z_q, spec_mask, g=g)
            stats = self.prior_proj(x) * x_mask
            m_p, logs_p = torch.split(stats, self.inter_channels, dim=1)
            x_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(
                z_q, f0, spec_lengths, self.segment_size)
            o = self.dec(x_slice, g=g, f0=pitch_slice)

            # 方案3：FM 路径（只在 use_unified_flow + training 时计算）
            if self.use_unified_flow and self.training:
                # FM 学习从 NF 输出(x_0)到真实 z_q(x_1)的精修速度场
                # x_1 = 真实后验样本
                x_1 = z_q.detach()
                # x_0 = NF 逆变换输出（与推理时的 Hybrid 起点一致）
                # 先验采样 + NF 逆变换，模拟推理时的起点
                z_p_fm = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * 0.35
                x_0, _ = self.flow(z_p_fm, spec_mask, g=g, mode='nf', reverse=True)
                x_0 = x_0.detach()  # 切断 FM 到 NF 的梯度
                t = torch.rand(x_1.shape[0], 1, 1, device=x_1.device)
                x_t = (1 - t) * x_0 + t * x_1
                u_t = x_1 - x_0
                # GeneralizedFlow 的 FM 模式预测速度场
                v_pred = self.flow(x_t, spec_mask, g=g, t=t.squeeze(-1), mode='fm')
                # 命名为 loss_flow_match，避免与 train.py 已有的 loss_fm=feature_loss 混淆
                loss_flow_match = F.mse_loss(v_pred * spec_mask, u_t * spec_mask)

            return o, ids_slice, spec_mask, (z_q, z_p, m_p, logs_p, m_q, logs_q), 0, 0, 0, loss_flow_match
        else:
            # A1（特征先验流）：flow 正向编码 c → z_p，dec(z_p)
            # 训练与推理路径一致（都走 flow 正向），无 enc_q，无先验采样
            # 固定先验 N(0,1)：m_p=0, logs_p=0，KL = -0.5 + 0.5*||z_p||^2
            # 约束 flow 输出方差≈1，防止 z_p 漂移；不串谋（先验不依赖 x）
            z_p = self.flow(x, x_mask, g=g)
            m_p = torch.zeros_like(z_p)
            logs_p = torch.zeros_like(z_p)
            # A1 无 enc_q：logs_q=0（固定后验方差），m_q=None
            logs_q = torch.zeros_like(z_p)
            x_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(
                z_p, f0, c_lengths, self.segment_size)
            o = self.dec(x_slice, g=g, f0=pitch_slice)
            # 返回元组：(z, z_p, m_p, logs_p, m_q, logs_q)
            # A1 中 z = z_p（flow 输出即为 decoder 输入）
            return o, ids_slice, x_mask, (z_p, z_p, m_p, logs_p, None, logs_q), 0, 0, 0, loss_flow_match

    @torch.no_grad()
    def infer(self, c, f0, uv, g=None, noice_scale=0.35, seed=52468, predict_f0=False, vol=None):
        if c.device == torch.device("cuda"):
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        if self.character_mix and len(g) > 1:
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))
            g = g * self.speaker_map
            g = torch.sum(g, dim=1)
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0)
        else:
            if g.dim() == 1:
                g = g.unsqueeze(0)
            g = self.emb_g(g).transpose(1, 2)

        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        vol = self.emb_vol(vol[:, :, None]).transpose(1, 2) if vol is not None and self.vol_embedding else 0
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

        if self.flow_mode == 'a2':
            # 先验采样 -> 逆流还原 -> 解码
            stats = self.prior_proj(x) * x_mask
            m_p, logs_p = torch.split(stats, self.inter_channels, dim=1)
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noice_scale
            if self.use_unified_flow:
                # GeneralizedFlow：NF 逆变换返回 (x, logdet)，取 x
                z, _ = self.flow(z_p, x_mask, g=g, mode='nf', reverse=True)
            else:
                z = self.flow(z_p, x_mask, g=g, reverse=True)
            o = self.dec(z, g=g, f0=f0)
        else:
            # A1：flow 正向（与训练一致，特征信息完整保留）
            z_p = self.flow(x, x_mask, g=g)
            o = self.dec(z_p, g=g, f0=f0)
        return o, f0

    @torch.no_grad()
    def infer_hybrid(self, c, f0, uv, g=None, noice_scale=0.35, seed=52468,
                     hybrid_steps=None, vol=None):
        """
        Hybrid 推理（方案3）：NF 快速定位 + FM 少步精修
        仅在 use_unified_flow=True 时有意义；若未启用则退化为纯 NF。

        Args:
            c: content features [B, ssl_dim, L]
            f0: f0 [B, L]
            uv: uv flag [B, L]
            g: speaker id（emb_g 输入）
            noice_scale: 先验采样噪声缩放
            hybrid_steps: FM 精修步数（None → 从 config 读 self.hybrid_steps）
            vol: 音量

        Returns:
            o: 音频 [B, 1, T]
        """
        if self.flow_mode != 'a2':
            raise RuntimeError(
                f"infer_hybrid 仅支持 A2 模式（后验流），当前 flow_mode={self.flow_mode}。"
                "A1 模式（特征先验流）无 enc_q 提供 FM 目标，不支持 Hybrid 推理，"
                "请使用 infer() 进行纯 NF 推理。")

        if c.device == torch.device("cuda"):
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

        if hybrid_steps is None:
            hybrid_steps = getattr(self, 'hybrid_steps', 4)

        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        if self.character_mix and len(g) > 1:
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))
            g = g * self.speaker_map
            g = torch.sum(g, dim=1)
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0)
        else:
            if g.dim() == 1:
                g = g.unsqueeze(0)
            g = self.emb_g(g).transpose(1, 2)

        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        vol = self.emb_vol(vol[:, :, None]).transpose(1, 2) if vol is not None and self.vol_embedding else 0
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

        # Step 1: 先验采样 + NF 逆变换（快速给起点）
        stats = self.prior_proj(x) * x_mask
        m_p, logs_p = torch.split(stats, self.inter_channels, dim=1)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noice_scale

        if self.use_unified_flow:
            # GeneralizedFlow：NF 逆变换返回 (x, logdet)
            x_t, _ = self.flow(z_p, x_mask, g=g, mode='nf', reverse=True)
            # Step 2: FM 精修（hybrid_steps 步欧拉积分，从 t=0→1）
            dt = 1.0 / max(hybrid_steps, 1)
            for i in range(max(hybrid_steps, 1)):
                t_val = float(i) / max(hybrid_steps, 1)
                t = torch.full((x_t.size(0), 1), t_val, device=x_t.device)
                v = self.flow(x_t, x_mask, g=g, t=t, mode='fm')
                x_t = x_t + v * dt
        else:
            # 未启用方案3：退化为纯 NF 逆变换
            x_t = self.flow(z_p, x_mask, g=g, reverse=True)

        # Step 3: Decoder → 音频
        o = self.dec(x_t * x_mask, g=g, f0=f0)
        return o, f0
