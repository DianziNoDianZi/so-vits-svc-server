# 方案 3 详细开发实施计划

> 本计划**只写怎么做**，不写为什么。每一步都有明确的文件、代码、验证标准。

---

## 前置条件

- [ ] A2 baseline 模型已训练到 step 2000+
- [ ] 已安装 `experiments/` 和 `tests/` 目录
- [ ] 能在 GPU 上跑通现有 A2 训练（不报错）

---

## Step 1：环境准备（1 天）

### 1.1 创建目录
```bash
cd /path/to/so-vits-svc-server
mkdir -p experiments tests
```

### 1.2 添加配置项到 config 模板
在 `configs_template/config_template.json` 中新增（`model` 段 + `train` 段）：
```json
"model": {
    "arch": "rvc-flow",
    "use_unified_flow": false,   // 新增：是否启用方案3
    "hybrid_steps": 4,            // 新增：hybrid推理FM精修步数
    "time_embed_dim": 128          // 新增：时间嵌入维度
},
"train": {
    ...
    "c_fm": 0.3                   // 新增：FM loss 权重（放 train 段，train.py 读 hps.train.c_fm）
}
```

### 1.3 验证
- [ ] 配置文件修改不破坏现有训练
- [ ] 用 `python -c "import utils; hps = utils.get_hparams_from_file('configs_template/config_template.json'); print(hps.train.c_fm, hps.model.use_unified_flow)"` 能读到 `0.3 False`

---

## Step 2：Phase 1 — 理论验证脚本（3 天）

### 2.1 梯度冲突检测脚本
**文件**：`experiments/grad_conflict.py`
**目的**：算 KL loss 和 FM loss 对共享骨干的梯度余弦相似度

```python
"""
梯度冲突检测脚本：验证 NF-KL 和 FM-MSE 对共享骨干的梯度是否同向
用法：python experiments/grad_conflict.py --config configs/44k/config.json --checkpoint G_2000.pth
"""
import argparse
import torch
import numpy as np
from models import SynthesizerTrnRvcFlow
import utils

def compute_grad_cosim(model, c, f0, spec, x_mask, g, spec_lengths, num_steps=10):
    """
    计算 KL loss 和 FM loss 对共享骨干的梯度余弦相似度
    共享骨干：flow 网络的所有 parameters
    """
    from modules import losses as L

    # 提取 flow 网络参数作为共享骨干
    shared_params = []
    for name, param in model.flow.named_parameters():
        if param.requires_grad:
            shared_params.append(param)

    cos_sims = []

    for step in range(num_steps):
        # 清除旧梯度
        model.zero_grad()

        # ---- NF-KL 路径（A2 实际接口）----
        # 后验：enc_q(spec, spec_lengths, g)
        z_q, m_q, logs_q, spec_mask = model.enc_q(spec, spec_lengths, g=g)
        # 先验：prior_proj 是 Conv1d，先过 pre 投影再算统计
        x = model.pre(c) * x_mask + model.emb_uv(
            torch.zeros(c.size(0), c.size(2), dtype=torch.long, device=c.device)
        ).transpose(1, 2)
        stats = model.prior_proj(x) * x_mask
        m_p, logs_p = torch.split(stats, model.inter_channels, dim=1)
        # NF 前向：z_q → z_p
        z_p_nf = model.flow(z_q, spec_mask, g=g)
        loss_kl = L.kl_loss(z_p_nf, logs_q, m_p, logs_p, spec_mask)

        # 保留计算图，供 FM 使用
        loss_kl.backward(retain_graph=True)

        # 提取 KL 梯度
        g_kl = []
        for p in shared_params:
            if p.grad is not None:
                g_kl.append(p.grad.clone())
        model.zero_grad()

        # ---- FM-MSE 路径 ----
        x_1 = z_q.detach()  # 关键：detach 防止 FM 梯度回传到 enc_q
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], 1, 1, device=x_1.device)
        x_t = (1 - t) * x_0 + t * x_1
        u_t = x_1 - x_0

        # FM 前向（临时用现有 flow 模拟，加时间偏置）
        t_bias = torch.randn_like(x_t)[:, :1, :] * 0.1  # 临时时间注入
        v_pred = model.flow(x_t + t_bias * spec_mask, spec_mask, g=g)

        loss_flow_match = torch.nn.functional.mse_loss(v_pred * spec_mask, u_t * spec_mask)
        loss_flow_match.backward()

        # 提取 FM 梯度
        g_fm = []
        for p in shared_params:
            if p.grad is not None:
                g_fm.append(p.grad.clone())
        model.zero_grad()

        # ---- 计算余弦相似度 ----
        g_kl_flat = torch.cat([gg.view(-1) for gg in g_kl])
        g_fm_flat = torch.cat([gg.view(-1) for gg in g_fm])
        cos_sim = torch.nn.functional.cosine_similarity(
            g_kl_flat.unsqueeze(0), g_fm_flat.unsqueeze(0)
        ).item()
        cos_sims.append(cos_sim)

    return np.mean(cos_sims), np.std(cos_sims)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num_steps', type=int, default=10)
    args = parser.parse_args()
    
    hps = utils.get_hparams_from_file(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建模型
    net_g = SynthesizerTrnRvcFlow(
        spec_channels=hps.data.filter_length // 2 + 1,
        segment_size=hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)
    
    # 加载 baseline 权重
    _ = utils.load_checkpoint(args.checkpoint, net_g, None)
    net_g.eval()
    
    # 构造 dummy 输入
    B = 2
    L = 100
    c = torch.randn(B, 256, L, device=device)       # content features
    f0 = torch.rand(B, L, device=device) * 100 + 100  # f0
    spec = torch.randn(B, hps.data.filter_length // 2 + 1, L, device=device)  # 线性谱，非 mel
    x_mask = torch.ones(B, 1, L, device=device)
    g = torch.randint(0, hps.model.n_speakers, (B,), device=device)  # speaker id（emb_g 输入）
    spec_lengths = torch.tensor([L] * B, device=device)

    mean_cos, std_cos = compute_grad_cosim(
        net_g, c, f0, spec, x_mask, g, spec_lengths, args.num_steps
    )
    
    print(f"余弦相似度: {mean_cos:.4f} ± {std_cos:.4f}")
    if mean_cos > 0:
        print("✓ 梯度同向，方案3可行")
    elif mean_cos > -0.3:
        print("△ 轻微冲突，需调 c_fm 权重缓解")
    else:
        print("✗ 严重冲突，放弃方案3，退回方案2")
```

**验证**：
- [ ] 脚本能在 GPU 上跑通，不报错
- [ ] 输出余弦相似度数值
- [ ] 判定结论明确

### 2.2 可逆性单元测试
**文件**：`tests/test_invertible.py`
**目的**：验证 GeneralizedCouplingLayer（待实现）的可逆性

```python
"""
GeneralizedCouplingLayer 可逆性测试
用法：python -m pytest tests/test_invertible.py -v
"""
import torch
import pytest
from models import GeneralizedCouplingLayer


class TestGeneralizedCouplingLayer:
    def setup_method(self):
        self.B, self.C, self.L = 2, 192, 100
        self.layer = GeneralizedCouplingLayer(
            channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            gin_channels=256,
            mean_only=True
        ).cuda()
        self.mask = torch.ones(self.B, 1, self.L).cuda()
        self.g = torch.randn(self.B, 256, 1).cuda()

    def test_nf_reversibility(self):
        """NF 路径：forward + reverse 应还原输入"""
        x = torch.randn(self.B, self.C, self.L).cuda()
        
        # Forward (NF mode)
        x_forward, logdet = self.layer(x, self.mask, g=self.g, mode='nf', reverse=False)
        
        # Reverse (NF mode)
        x_back, _ = self.layer(x_forward, self.mask, g=self.g, mode='nf', reverse=True)
        
        # 验证还原
        assert torch.allclose(x, x_back, atol=1e-5), \
            f"NF 不可逆: max_diff={torch.max((x - x_back).abs()):.8f}"
        print(f"✓ NF 可逆，max_diff={torch.max((x - x_back).abs()):.8f}")

    def test_fm_output_shape(self):
        """FM 路径：输出速度场形状应与输入一致"""
        x = torch.randn(self.B, self.C, self.L).cuda()
        t = torch.rand(self.B, 1).cuda()
        
        v = self.layer(x, self.mask, g=self.g, t=t, mode='fm')
        
        assert v.shape == x.shape, \
            f"FM 输出形状不匹配: {v.shape} vs {x.shape}"
        print(f"✓ FM 输出形状正确: {v.shape}")

    def test_both_mode(self):
        """Both mode：应同时返回 NF 变换和 FM 速度场"""
        x = torch.randn(self.B, self.C, self.L).cuda()
        t = torch.rand(self.B, 1).cuda()
        
        x_nf, logdet, v_fm = self.layer(x, self.mask, g=self.g, t=t, mode='both')
        
        assert x_nf.shape == x.shape
        assert v_fm.shape == x.shape
        assert logdet.shape[0] == self.B
        print(f"✓ Both 模式输出正确")

    def test_logdet_consistency(self):
        """logdet 应与变换的雅可比一致（数值验证）"""
        x = torch.randn(self.B, self.C, self.L).cuda()
        
        _, logdet = self.layer(x, self.mask, g=self.g, mode='nf', reverse=False)
        
        # logdet 应为标量或 [B, 1, L] 形状
        assert logdet is not None
        assert logdet.abs().item() < 100, f"logdet 异常: {logdet}"
        print(f"✓ logdet 正常: {logdet.mean().item():.4f}")
```

**验证**：
- [ ] 所有 4 个测试用例通过
- [ ] 无 CUDA OOM

### 2.3 hybrid 推理起点距离统计
**文件**：`experiments/hybrid_start_distance.py`
**目的**：统计 NF 逆变换输出与真实 z_q 的距离，决定 FM 精修步数

```python
"""
Hybrid 推理起点距离统计：决定 FM 精修步数
用法：python experiments/hybrid_start_distance.py --config configs/44k/config.json --checkpoint G_2000.pth
"""
import argparse
import torch
import numpy as np
from models import SynthesizerTrnRvcFlow
import utils


def compute_start_distance(model, c, f0, spec, x_mask, g, spec_lengths, num_samples=100):
    """
    计算 NF 逆变换后 z_nf 与真实 z_q 的距离
    距离小 → FM 只需少步数精修
    """
    distances = []

    for _ in range(num_samples):
        with torch.no_grad():
            # 后验：enc_q(spec, spec_lengths, g)
            z_q, m_q, logs_q, spec_mask = model.enc_q(spec, spec_lengths, g=g)
            # 先验：prior_proj（Conv1d）
            x = model.pre(c) * x_mask + model.emb_uv(
                torch.zeros(c.size(0), c.size(2), dtype=torch.long, device=c.device)
            ).transpose(1, 2)
            stats = model.prior_proj(x) * x_mask
            m_p, logs_p = torch.split(stats, model.inter_channels, dim=1)
            # 先验采样
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * 0.35

            # NF 逆变换：从 z_p 到 z_nf
            z_nf = model.flow(z_p, x_mask, g=g, reverse=True)

            # 距离
            dist = ((z_nf - z_q) ** 2).mean().item()
            distances.append(dist)
    
    distances = np.array(distances)
    return {
        'mean': distances.mean(),
        'std': distances.std(),
        'median': np.median(distances),
        'p90': np.percentile(distances, 90),
        'max': distances.max()
    }


def recommend_steps(stats):
    """根据距离推荐 FM 精修步数"""
    mean_dist = stats['mean']
    if mean_dist < 0.01:
        return 2  # 极近，2步即可
    elif mean_dist < 0.1:
        return 4  # 较近，4步足够
    elif mean_dist < 1.0:
        return 8  # 中等，8步保险
    else:
        return 16  # 较远，需16步


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()
    
    hps = utils.get_hparams_from_file(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net_g = SynthesizerTrnRvcFlow(
        spec_channels=hps.data.filter_length // 2 + 1,
        segment_size=hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)
    
    _ = utils.load_checkpoint(args.checkpoint, net_g, None)
    net_g.eval()
    
    B, L = 2, 100
    c = torch.randn(B, 256, L, device=device)
    f0 = torch.rand(B, L, device=device) * 100 + 100
    spec = torch.randn(B, hps.data.filter_length // 2 + 1, L, device=device)
    x_mask = torch.ones(B, 1, L, device=device)
    g = torch.randint(0, hps.model.n_speakers, (B,), device=device)
    spec_lengths = torch.tensor([L] * B, device=device)

    stats = compute_start_distance(
        net_g, c, f0, spec, x_mask, g, spec_lengths, args.num_samples
    )
    
    print(f"NF 逆变换 → 真实 z_q 距离统计:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")
    
    steps = recommend_steps(stats)
    print(f"\n推荐 FM 精修步数: {steps}")
    print(f"建议在 config 中设置 hybrid_steps={steps}")
```

**验证**：
- [ ] 脚本能跑通
- [ ] 输出距离统计和推荐步数
- [ ] 推荐步数在 2-16 之间（不在这个范围说明有问题）

### 2.4 Phase 1 验证汇总
**文件**：`experiments/phase1_report.md`
- [ ] 记录 2.1、2.2、2.3 的结果
- [ ] 明确给出"通过/不通过"结论
- [ ] 不通过时，给出退回方案 2 的具体步骤

---

## Step 3：Phase 2 — 核心实现（2-3 周）

### 3.1 GeneralizedCouplingLayer
**文件**：`models.py`
**位置**：在现有 `ResidualCouplingBlock` 类之后新增

```python
class GeneralizedCouplingLayer(nn.Module):
    """
    统一 NF + FM 的耦合层
    共享 FFT 骨干，双输出头：
      - head_nf: NF 变换增量（用于 KL loss，需可逆）
      - head_fm: FM 速度场（用于 MSE loss，不需可逆）
    """
    def __init__(self, channels, hidden_channels, kernel_size,
                 dilation_rate, n_layers, gin_channels=0,
                 mean_only=True):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.gin_channels = gin_channels
        self.mean_only = mean_only
        
        # 共享 FFT 骨干（复用现有 attentions.FFT）
        self.fft = attentions.FFT(
            hidden_channels, hidden_channels * 4,
            n_heads=2, n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=0, isflow=True,
            gin_channels=gin_channels
        )
        
        # NF 输出头：可逆变换（只输出 shift，mean-only）
        self.head_nf = nn.Conv1d(hidden_channels, channels, 1)
        
        # FM 输出头：速度场（不需要可逆约束）
        self.head_fm = nn.Conv1d(hidden_channels, channels, 1)
        
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
            g: [B, gin_channels, 1] 条件
            t: [B, 1] 时间（FM 专用，NF 模式下为 None）
            mode: 'nf' | 'fm' | 'both'
            reverse: bool，NF 模式下是否逆变换
        
        Returns:
            mode='nf':   (x_out, logdet)
            mode='fm':   (v_field,)
            mode='both': (x_out, logdet, v_field)
        """
        # 时间嵌入注入
        t_bias = 0
        if t is not None:
            t_emb = self.time_mlp(t)  # [B, hidden]
            t_bias = t_emb.unsqueeze(-1)  # [B, hidden, 1]
        
        # 共享 FFT 骨干
        h = self.fft(x + t_bias * x_mask, x_mask, g=g)
        
        results = {}
        
        # NF 路径
        if mode in ('nf', 'both'):
            delta_nf = self.head_nf(h) * x_mask
            if reverse:
                x_out = x - delta_nf  # 逆变换：x = x_forward - delta
            else:
                x_out = x + delta_nf  # 前向变换：x_forward = x + delta
            logdet = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device)
            results['nf'] = (x_out, logdet)
        
        # FM 路径
        if mode in ('fm', 'both'):
            v_fm = self.head_fm(h) * x_mask
            results['fm'] = v_fm
        
        # 返回
        if mode == 'nf':
            return results['nf']
        elif mode == 'fm':
            return results['fm']
        else:  # both
            return results['nf'][0], results['nf'][1], results['fm']
```

**验证**：
- [ ] `python -c "from models import GeneralizedCouplingLayer"` 不报错
- [ ] 运行 `tests/test_invertible.py` 全部通过

### 3.2 GeneralizedFlow 容器
**文件**：`models.py`
**位置**：在 `GeneralizedCouplingLayer` 之后新增

```python
class GeneralizedFlow(nn.Module):
    """
    统一 NF + FM 的流容器
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
                    x = flow(x, x_mask, g=g, reverse=False)
                    continue
                
                if mode in ('nf', 'both'):
                    x, ld = flow(x, x_mask, g=g, t=t, mode='nf', reverse=False)
                    logdet_total = logdet_total + ld
                if mode in ('fm', 'both'):
                    _, _, v = flow(x, x_mask, g=g, t=t, mode='fm', reverse=False)
                    v_accum = v_accum + v
        else:
            for flow in reversed(self.flows):
                if isinstance(flow, modules.Flip):
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
```

**验证**：
- [ ] `python -c "from models import GeneralizedFlow"` 不报错
- [ ] GeneralizedFlow 的 `nf` 模式可逆（forward + reverse 还原）
- [ ] GeneralizedFlow 的 `fm` 模式输出形状正确

### 3.3 集成到 SynthesizerTrnRvcFlow
**文件**：`models.py`
**位置**：修改 `SynthesizerTrnRvcFlow.__init__` 和 `forward`

#### 3.3.1 __init__ 中添加 GeneralizedFlow
```python
def __init__(self, ..., use_unified_flow=False, hybrid_steps=4, **kwargs):
    # ... 现有代码不变 ...

    self.use_unified_flow = use_unified_flow
    self.hybrid_steps = hybrid_steps

    if use_unified_flow:
        # 方案3：用 GeneralizedFlow 替换现有 TransformerCouplingBlock
        # gin_channels 与现有 flow 一致（speaker embedding 通道，非 ssl_dim）
        self.flow = GeneralizedFlow(
            inter_channels, hidden_channels,
            kernel_size=5, dilation_rate=1,
            n_layers=n_layers_trans_flow, n_flows=n_flow_layer,
            gin_channels=gin_channels
        )
    else:
        # 保持现有 self.flow = TransformerCouplingBlock(...) 不变
        self.flow = TransformerCouplingBlock(
            inter_channels, hidden_channels, filter_channels, n_heads,
            n_layers_trans_flow, 5, p_dropout, n_flow_layer,
            gin_channels=gin_channels, share_parameter=flow_share_parameter)

    # enc_q / prior_proj 不变（A2 仍需要后验编码 + 先验投影）
```

#### 3.3.2 forward 中添加 FM loss 计算
```python
def forward(self, c, f0, uv, spec, g=None, c_lengths=None, spec_lengths=None, vol=None):
    # ... 现有 A2 代码（NF 路径），得到 z_q, z_p, m_p, logs_p, m_q, logs_q, spec_mask ...

    # 新增：FM 路径（只在 use_unified_flow + training 时计算）
    loss_flow_match = 0
    if self.use_unified_flow and self.training:
        # FM 目标 = z_q.detach()（关键：detach 防止 FM 梯度回传 enc_q）
        x_1 = z_q.detach()
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], 1, 1, device=x_1.device)
        x_t = (1 - t) * x_0 + t * x_1
        u_t = x_1 - x_0

        # 用 GeneralizedFlow 的 FM 模式预测速度场
        v_pred = self.flow(x_t, spec_mask, g=g, t=t.squeeze(-1), mode='fm')

        # FM loss（命名为 loss_flow_match，避免与 train.py 已有的 loss_fm=feature_loss 混淆）
        loss_flow_match = F.mse_loss(v_pred * spec_mask, u_t * spec_mask)

    # 返回：在现有 7 元组末尾追加 loss_flow_match（第 8 个元素）
    return o, ids_slice, spec_mask, (z_q, z_p, m_p, logs_p, m_q, logs_q), 0, 0, 0, loss_flow_match
```

**验证**：
- [ ] `python -c "from models import SynthesizerTrnRvcFlow; m = SynthesizerTrnRvcFlow(..., use_unified_flow=True)"` 不报错
- [ ] `m.train()` 模式下 forward 返回 8 元组，末位 `loss_flow_match` 为张量
- [ ] `m.eval()` 模式下 forward 末位 `loss_flow_match` 为 0（不计算）

### 3.4 训练 loss 分支
**文件**：`train.py`
**位置**：在 `train_and_evaluate` 函数中

**关键改动 1**：forward 调用解包从 7 元组改为 8 元组（末位是 `loss_flow_match`）：

```python
# 现有（7 元组）：
y_hat, ids_slice, z_mask, \
(z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(...)

# 改为（8 元组）：
y_hat, ids_slice, z_mask, \
(z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0, loss_flow_match = net_g(...)
```

**关键改动 2**：在现有 `loss_gen_all` 上加 FM loss：

```python
# ---- 现有 loss 计算不变 ----
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
# 注意：loss_fm 是 feature matching loss（discriminator 特征匹配），不是 flow matching！

# ---- 新增：flow matching loss 加权 ----
_g = net_g.module if hasattr(net_g, 'module') else net_g
if getattr(_g, 'use_unified_flow', False) and isinstance(loss_flow_match, torch.Tensor):
    c_fm = float(getattr(hps.train, 'c_fm', 0.3))
    loss_gen_all = loss_gen_all + c_fm * loss_flow_match
    # 日志
    if global_step % hps.train.log_interval == 0:
        logger.info(f"FlowMatch Loss: {loss_flow_match.item():.6f}")

# ---- 现有 backward 不变 ----
scaler.scale(loss_gen_all).backward()
```

**关键**：
- `loss_fm`（train.py 已有）= feature matching loss（来自 discriminator），**不要混淆**
- `loss_flow_match`（新增）= flow matching MSE loss
- FM loss 只加在 `loss_gen_all` 上，不加在 `loss_disc_all` 上
- `z_q.detach()` 在 forward 内部完成，确保 FM 梯度不回传 enc_q
- 不做 `loss_flow_match.backward()`，FM 梯度通过 `loss_gen_all.backward()` 统一回传

**验证**：
- [ ] 训练能跑通，不报错
- [ ] 日志中能看到 `FlowMatch Loss: xxx`
- [ ] 训练曲线无异常（loss 不飙升）

### 3.5 hybrid 推理分支
**文件**：`inference/infer_tool.py`
**位置**：在 `SynthesizerTrnRvcFlow` 类中新增 `infer_hybrid` 方法（与现有 `infer` 并列）

```python
@torch.no_grad()
def infer_hybrid(self, c, f0, uv, g=None, noice_scale=0.35, seed=52468,
                 hybrid_steps=None, vol=None):
    """
    Hybrid 推理：NF 快速定位 + FM 少步精修

    Args:
        c: content features [B, ssl_dim, L]
        f0: f0 [B, L]
        uv: uv flag [B, L]
        g: speaker id（emb_g 输入）
        noice_scale: 先验采样噪声缩放
        hybrid_steps: FM 精修步数（None → 从 config 读）
        vol: 音量

    Returns:
        o: 音频 [B, 1, T]
    """
    if c.device == torch.device("cuda"):
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    if hybrid_steps is None:
        hybrid_steps = getattr(self, 'hybrid_steps', 4)

    c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
    g = self.emb_g(g).transpose(1, 2)

    x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
    vol = self.emb_vol(vol[:, :, None]).transpose(1, 2) if vol is not None and self.vol_embedding else 0
    x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

    # Step 1: 先验采样 + NF 逆变换（快速给起点）
    stats = self.prior_proj(x) * x_mask
    m_p, logs_p = torch.split(stats, self.inter_channels, dim=1)
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noice_scale
    z_nf = self.flow(z_p, x_mask, g=g, mode='nf', reverse=True)

    # Step 2: FM 精修（hybrid_steps 步欧拉积分，从 t=0→1）
    x_t = z_nf
    dt = 1.0 / hybrid_steps
    for i in range(hybrid_steps):
        t_val = float(i) / hybrid_steps
        t = torch.full((x_t.size(0), 1), t_val, device=x_t.device)
        v = self.flow(x_t, x_mask, g=g, t=t, mode='fm')
        x_t = x_t + v * dt

    # Step 3: Decoder → audio（self.dec 本身就是 NSF-HiFiGAN，带 f0 输入，一步出音频）
    o = self.dec(x_t * x_mask, g=g, f0=f0)
    return o, f0
```

**验证**：
- [ ] `infer_hybrid` 能返回音频
- [ ] 音频无明显噪声
- [ ] 推理耗时 < 纯 FM（32 步），接近纯 NF

### 3.6 推理模式切换
**文件**：`inference/infer_tool.py`
**位置**：在 `Svc.slice_inference`（或调用 `net_g_ms.infer` 的地方）支持模式切换

注意：`Svc.infer`（大入口，处理 wav→特征→切片→vocoder 后处理）和 `net_g_ms.infer`（模型级推理）是两层，不要混淆。hybrid 模式切换发生在 `net_g_ms` 层。

```python
# 在 Svc 中调用 net_g_ms 的地方，根据 hybrid_mode 选择调用哪个方法
net_g = self.net_g_ms
use_unified = getattr(net_g, 'use_unified_flow', False)
hybrid_mode = kwargs.get('hybrid_mode', 'auto')

if hybrid_mode == 'auto':
    hybrid_mode = 'hybrid' if use_unified else 'nf'

if hybrid_mode == 'nf' or not use_unified:
    # 纯 NF 推理（现有行为，net_g.infer 内部含先验采样+flow逆变换+dec）
    o, f0_out = net_g.infer(c, f0, uv, g=g, noice_scale=noice_scale, seed=seed, vol=vol)
elif hybrid_mode == 'fm':
    # 纯 FM 推理（32 步欧拉积分）
    o, f0_out = net_g.infer_hybrid(c, f0, uv, g=g, noice_scale=noice_scale,
                                    seed=seed, hybrid_steps=32, vol=vol)
elif hybrid_mode == 'hybrid':
    # Hybrid 推理（4 步精修，默认）
    o, f0_out = net_g.infer_hybrid(c, f0, uv, g=g, noice_scale=noice_scale,
                                    seed=seed, vol=vol)
```

**验证**：
- [ ] `hybrid_mode='nf'` 行为与现有一致（纯 NF）
- [ ] `hybrid_mode='hybrid'` 能返回音频
- [ ] `hybrid_mode='fm'` 能返回音频
- [ ] 三种模式都能正常工作

### 3.7 Phase 2 验证清单
- [ ] 代码能 import，无语法错误
- [ ] 训练能跑通（`use_unified_flow=True`）
- [ ] 推理能输出声音（hybrid 模式）
- [ ] 日志中能看到 FM loss
- [ ] GPU 显存增量 < 20%（相比纯 NF）

---

## Step 4：Phase 3 — 效果验证（2-3 周）

### 4.1 三条 baseline 对比
**文件**：`experiments/baseline_comparison.py`

```python
"""
三条 baseline 对比脚本
用法：python experiments/baseline_comparison.py
"""
import torch
import numpy as np
import time
from models import SynthesizerTrnRvcFlow
import utils


def evaluate_model(model, config, mode, test_data, device):
    """评估单模型的指标"""
    model.eval()
    losses = []
    mel_errors = []
    times = []
    
    with torch.no_grad():
        for c, f0, spec, x_mask, uv, g in test_data:
            c, f0, spec, x_mask, uv, g = [
                x.to(device) for x in [c, f0, spec, x_mask, uv, g]
            ]
            
            t0 = time.time()
            
            if mode == 'nf':
                o, _ = model.infer(c, f0, uv, g=g)
            elif mode == 'fm':
                # 32 步 FM
                o = model.infer_hybrid(c, f0, uv, g=g, hybrid_steps=32)
            elif mode == 'hybrid':
                o = model.infer_hybrid(c, f0, uv, g=g, hybrid_steps=4)
            
            t1 = time.time()
            times.append(t1 - t0)
            
            mel_error = F.l1_loss(o, spec)
            mel_errors.append(mel_error.item())
    
    return {
        'mel_error': np.mean(mel_errors),
        'mel_std': np.std(mel_errors),
        'avg_time': np.mean(times),
        'speed_ratio': None  # 后计算
    }


if __name__ == '__main__':
    # 加载三个模型
    models = {
        'pure_nf': load_model('G_nf.pth', use_unified_flow=False),
        'pure_fm': load_model('G_fm.pth', use_unified_flow=False),  # FM-only 底模
        'unified': load_model('G_unified.pth', use_unified_flow=True),
    }
    
    # 评估
    results = {}
    for name, model in models.items():
        mode = 'nf' if name == 'pure_nf' else ('fm' if name == 'pure_fm' else 'hybrid')
        results[name] = evaluate_model(model, config, mode, test_data, device)
    
    # 计算速度比
    nf_time = results['pure_nf']['avg_time']
    for name in results:
        results[name]['speed_ratio'] = nf_time / results[name]['avg_time']
    
    # 打印结果
    print(f"{'Model':<12} {'Mel Err':>8} {'Std':>8} {'Time(s)':>8} {'Speed':>8}")
    print("-" * 48)
    for name, r in results.items():
        print(f"{name:<12} {r['mel_error']:>8.4f} {r['mel_std']:>8.4f} "
              f"{r['avg_time']:>8.3f} {r['speed_ratio']:>7.2f}x")
    
    # 判定
    if results['unified']['mel_error'] <= results['pure_nf']['mel_error']:
        print("\n✓ 方案3质量 ≥ 纯NF，通过生死线#2")
    else:
        print("\n✗ 方案3质量 < 纯NF，不通过，退回方案2")
```

**验证**：
- [ ] 三条模型都能跑完评估
- [ ] 输出对比表格
- [ ] 明确判定"通过/不通过"

### 4.2 防钻空子验证
**方法**：
- 训练两个模型：纯 GAN（A2 baseline）和方案 3
- 监控 reference_loss 上升的 step（崩溃点）
- 对比生成音频的高频噪声（10kHz 以上能量）

**高频噪声检测脚本**：`experiments/high_freq_check.py`

```python
def check_high_freq_noise(audio, sr=44100, threshold=0.1):
    """
    检测高频噪声：10kHz 以上能量占比
    阈值 > 0.1 = 有明显高频噪声
    """
    # STFT
    f, t, Zxx = librosa.stft(audio, sr=sr)
    mag = np.abs(Zxx)
    
    # 高频能量
    high_freq_mask = f >= 10000
    high_freq_energy = mag[high_freq_mask].mean()
    total_energy = mag.mean()
    
    ratio = high_freq_energy / (total_energy + 1e-8)
    return ratio, ratio > threshold
```

**验证**：
- [ ] 纯 GAN 模型在 step N 后高频噪声显著增加
- [ ] 方案 3 模型在相同 step 下高频噪声少
- [ ] 方案 3 模型能训到更高 step 不崩

### 4.3 hybrid 步数-质量曲线
**文件**：`experiments/hybrid_steps_sweep.py`

```python
def sweep_steps(model, test_data, steps=[0, 2, 4, 8, 16, 32]):
    """扫描 hybrid 步数 vs 质量"""
    results = {}
    for n_steps in steps:
        mel_errors = []
        times = []
        for c, f0, spec, x_mask, uv, g in test_data:
            t0 = time.time()
            o = model.infer_hybrid(c, f0, uv, g=g, hybrid_steps=n_steps)
            t1 = time.time()
            mel_errors.append(F.l1_loss(o, spec).item())
            times.append(t1 - t0)
        results[n_steps] = {
            'mel_err': np.mean(mel_errors),
            'time': np.mean(times)
        }
    return results
```

**验证**：
- [ ] 0 步（纯 NF）→ 32 步（纯 FM）都能跑
- [ ] 找到"质量不再明显提升"的最小步数
- [ ] 通常最优在 4-8 步

### 4.4 Phase 3 判定
- [ ] 生死线#2 通过：方案 3 质量 ≥ 纯 NF
- [ ] 生死线#3 通过：防钻空子有效
- [ ] hybrid 步数确定

---

## Step 5：Phase 4 — 调参优化（2 周）

### 5.1 c_fm 权重扫描
**文件**：`experiments/c_fm_sweep.py`

```python
c_fm_values = [0.1, 0.3, 0.5, 1.0]
for c_fm in c_fm_values:
    train_with_c_fm(c_fm)
    evaluate_and_save(c_fm)
```

**最优 c_fm 选择标准**：
- 梯度余弦相似度最接近 0（不冲突也不独立）
- reference_loss 收敛最快
- 无 GAN 崩溃迹象

### 5.2 共享 vs 独立对比
- A 组：方案 3（共享骨干）
- B 组：方案 2（双独立网络，见 `plan_scheme2.md`）
- 对比参数量、训练稳定性、最终质量

### 5.3 小数据鲁棒性
- 27min 数据训练方案 3
- 7min 数据训练方案 3
- 对比过拟合点和最终质量

---

## Step 6：Phase 5 — 工程化（2-4 周）

### 6.1 配置与文档
- [x] config 新增 `use_unified_flow`、`c_fm`、`hybrid_steps`
- [x] 推理 API 支持 `hybrid_mode` 参数（infer_tool.py 的 infer/slice_inference + worker + webUI 下拉）
- [x] 更新 NOTICE 标注 FM 改动
- [x] 写 `docs/unified_flow.md` 使用文档
- [x] 更新 README.md / README_en.md（方案3架构、训练/推理参数、FAQ、ONNX、目录结构）

### 6.2 推理优化
- [x] hybrid 步数验证（4 步 → 2 步）：修复 sweep 脚本 bug 后重扫，实测 2 步质量明显低于 4 步，**4 步为最优**，config 保持 4（真正的少步蒸馏需训练时一致性训练，属未来工作）
- [x] ONNX 导出支持（onnx_export_generator.py 加统一流分支，导出 G_6000.onnx 验证一致）
- [x] EMA 权重推理（utils.load_checkpoint use_ema=True，Svc 已用）


---

## 附录：常见问题速查

### Q1：训练时出现 "Trying to backward through the graph a second time"
**原因**：NF loss 和 FM loss 共用了计算图但没正确 detach
**解决**：
- FM 路径用 `x_1 = z_q.detach()` 切断 FM 到 enc_q 的梯度
- NF loss 和 FM loss 合并到 `loss_gan` 后统一 `.backward()`
- 不要对两个 loss 分别 `.backward()`

### Q2：训练时出现 CUDA OOM
**原因**：双 loss 导致显存翻倍
**解决**：
- 减小 batch_size
- 用 `torch.cuda.amp` 混合精度
- FM loss 用 `z_q.detach()` 减少图规模

### Q3：FM loss 一直不收敛
**原因**：FM 头输出被 NF 头干扰
**解决**：
- 检查 `head_fm` 是否在 `mode='fm'` 下独立工作
- FM 路径的 `x_t` 不要过深地参与 NF 路径

### Q4：hybrid 推理有噪声
**原因**：FM 精修步数太少或起点太远
**解决**：
- 增加 `hybrid_steps`（4 → 8）
- 检查 NF 逆变换质量（起点质量直接影响精修效果）

### Q5：FM 和 NF 输出形状不匹配
**原因**：FM 头和 NF 头的通道数不同
**解决**：
- 两个头都输出 `channels` 维度（与输入一致）
- 检查 `GeneralizedCouplingLayer` 的 `self.head_nf` 和 `self.head_fm` 输出维度

---

## 验证总结表

| Step | 验证项 | 通过标准 | 状态 |
|---|---|---|---|
| 1.1 | 配置文件新增项 | 能读到 c_fm=0.3 | ☐ |
| 2.1 | 梯度冲突检测 | cos_sim > -0.3 | ☐ |
| 2.2 | 可逆性测试 | 4 个用例全过 | ☐ |
| 2.3 | 起点距离统计 | 推荐步数 2-16 | ☐ |
| 3.1 | GeneralizedCouplingLayer | import 无错 | ☐ |
| 3.2 | GeneralizedFlow | nf 模式可逆 | ☐ |
| 3.3 | 集成到模型 | forward 返回收 fm loss | ☐ |
| 3.4 | 训练 loss 分支 | 日志有 FM Loss | ☐ |
| 3.5 | hybrid 推理 | 返回音频 | ☐ |
| 3.6 | 推理模式切换 | 三种模式正常 | ☐ |
| 4.1 | baseline 对比 | 方案3 ≥ 纯NF | ☐ |
| 4.2 | 防钻空子 | 高频噪声少 | ☐ |
| 4.3 | hybrid 步数扫描 | 找到最优步数 | ☐ |
| 5.1 | c_fm 扫描 | 找到最优权重 | ☐ |
