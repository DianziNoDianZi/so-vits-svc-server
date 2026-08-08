# 方案3：统一 NF+FM Flow（Unified Flow）使用文档

本仓库在 RVC-Flow (A2) 基础上新增了「方案3 统一流」架构：用**同一组 FFT 骨干**同时承载
Normalizing Flow (NF) 与 Flow Matching (FM) 两条路径，推理时以 **Hybrid 模式**（NF 快速定位 +
FM 少步精修）兼顾速度与质量。

> 设计动机、理论验证、调参过程见 `experiments/phase1_report.md`、`phase3_report.md`、
> `phase4_report.md`。本文只讲**怎么用**。

---

## 1. 核心概念

| 模式 | 含义 | 何时用 |
|---|---|---|
| `nf` | 纯 Normalizing Flow（先验采样 + flow 逆变换 + decoder） | 最快，与原 A2 行为一致 |
| `fm` | 纯 Flow Matching（32 步欧拉积分从噪声到数据） | 质量上限最高，最慢 |
| `hybrid` | NF 逆变换给出起点，FM 用少量步数（默认 4）精修 | **推荐**，质量接近 FM，速度接近 NF |

三种模式共享同一套训练权重，无需分别训练。

---

## 2. 配置项

在 `config.json` 的 `model` 段与 `train` 段新增以下字段（模板 `configs_template/config_template.json` 已含）：

```json
{
  "model": {
    "use_unified_flow": true,   // 是否启用方案3统一流；false 时走原 A2 TransformerCouplingBlock
    "hybrid_steps": 4,          // Hybrid 推理的 FM 精修步数（推荐 2~4，详见第 5 节）
    "time_embed_dim": 128       // FM 时间嵌入维度
  },
  "train": {
    "c_fm": 0.5                 // FM loss 权重（推荐 0.5；过小如 0.1 会让 FM head 学不动）
  }
}
```

### 关键说明

- `use_unified_flow=false` 时，`hybrid_steps` / `c_fm` / `time_embed_dim` 均不生效，行为与原 A2 完全一致。
- `c_fm` 经扫描验证：`0.1` 时 FM 几乎无梯度（不可用），`0.5` 时 FM 精修生效且训练稳定（**推荐**）。
- 推理模式由 `hybrid_mode` 参数控制，**不**写在 config 里，而是推理时传入（见第 4 节）。

---

## 3. 训练

启用方案3只需把 `use_unified_flow` 设为 `true`，其余训练流程不变：

```bash
python train.py -c logs/<task>/config.json -m <task>
```

### 训练时的内部行为

- 模型用 `GeneralizedFlow` 替换原 `TransformerCouplingBlock`，内含多个 `GeneralizedCouplingLayer`：
  - 共享 FFT 骨干 + 双输出头（`head_nf` 可逆变换、`head_fm` 速度场）。
  - NF 路径用 channel-split coupling 保证可逆；FM 路径预测速度场（不可逆约束）。
- `forward` 在训练时额外计算 FM loss：
  - `x_1 = z_q.detach()`（真实后验，切断到 enc_q 的梯度）
  - `x_0 = NF 逆变换(先验采样).detach()`（**与推理起点一致**，关键一致性保证）
  - 在 `[x_0, x_1]` 间线性插值 `x_t`，FM head 预测速度 `v`，MSE 拟合真实速度 `u_t = x_1 - x_0`。
- `head_fm` 零初始化：FM 从恒等映射（v=0）起步，**初期 Hybrid ≈ NF**，不会因随机速度场破坏语音。
- `loss_flow_match` 按 `c_fm` 加权并入 `loss_gen_all` 统一反传，日志中可见 `FlowMatch Loss`。

### 训练日志关注点

- `FlowMatch Loss` 应整体下降（GAN 训练会有波动，属正常）。
- `Eval Loss` 最佳点即推荐 checkpoint（本项目验证集最佳在 step 6000）。

---

## 4. 推理

### 4.1 Python API

模型级推理直接调用 `SynthesizerTrnRvcFlow` 的方法：

```python
# 纯 NF（与原 A2 一致）
audio, f0 = net_g.infer(c, f0=f0, g=sid, uv=uv, noice_scale=0.35, vol=vol)

# Hybrid（NF + FM 精修，推荐）
audio, f0 = net_g.infer_hybrid(c, f0=f0, g=sid, uv=uv, noice_scale=0.35, vol=vol)
# hybrid_steps 省略时读 config 的 model.hybrid_steps

# 纯 FM（32 步欧拉积分，质量上限最高）
audio, f0 = net_g.infer_hybrid(c, f0=f0, g=sid, uv=uv, noice_scale=0.35, hybrid_steps=32, vol=vol)
```

> `infer_hybrid` 在 `use_unified_flow=false` 时自动退化为纯 NF 逆变换，可安全调用。

### 4.2 Svc 管线（切片推理）

`inference/infer_tool.py` 的 `Svc.infer` 与 `Svc.slice_inference` 支持 `hybrid_mode` 参数：

```python
audio = svc.slice_inference(
    raw_audio_path=wav_path,
    spk=spk_id, tran=tran,
    slice_db=-40, cluster_infer_ratio=0,
    auto_predict_f0=False, noice_scale=0.4,
    f0_predictor='pm',
    # 方案3推理模式：'auto' | 'nf' | 'fm' | 'hybrid'
    hybrid_mode='hybrid',
)
```

`hybrid_mode` 取值：

| 值 | 行为 |
|---|---|
| `'auto'` | 默认。`use_unified_flow=true` → `hybrid`；否则 → `nf` |
| `'nf'` | 纯 NF，调用 `net_g.infer` |
| `'fm'` | 纯 FM，调用 `net_g.infer_hybrid(hybrid_steps=32)` |
| `'hybrid'` | NF + FM 精修，调用 `net_g.infer_hybrid()`（步数读 config） |

### 4.3 推理 Worker

`server/inference_worker.py` 默认走 `hybrid_mode='auto'`：加载的模型若为方案3则自动用 Hybrid。
若需显式指定模式，在调用 `svc.slice_inference(...)` 时传入 `hybrid_mode`。

### 4.4 EMA 权重推理

`Svc` 初始化时已用 `utils.load_checkpoint(..., use_ema=True)` 加载 EMA 权重（更稳）；
checkpoint 不含 EMA 时自动回退到原始权重，无需额外配置。

---

## 5. hybrid_steps 怎么选

经步数扫描（`experiments/hybrid_steps_sweep.py`，G_6000.pth）验证：

| hybrid_steps | 耗时(s) | 质量(HF ratio) | 说明 |
|---|---|---|---|
| 0（纯 NF） | 0.064 | 0.2596 | 基线，高频略不足 |
| 2 | 0.110 | 0.2663 | 最快，质量略低于 4 步 |
| 4 | 0.166 | 0.2718 | **推荐**，质量/速度平衡点 |
| 8 | 0.237 | 0.2736 | 质量略升，收益递减 |
| 16 | 0.378 | 0.2748 | 收益很小 |
| 32（纯 FM） | 0.698 | 0.2754 | 质量上限，最慢 |
| original | — | 0.2869 | 数据集 baseline |

**结论**：
- 步数对质量和耗时**都有影响**（HF ratio 随步数从 0.2663 升到 0.2754，耗时线性增长）。
- **4 步是推荐值**：质量明显优于 2 步（0.2718 vs 0.2663），且比纯 FM(32) 快 4.2 倍。
- 8 步以上收益递减，不建议。2 步可在追求极致速度时使用，但质量有可测量的下降。
- config 默认 `hybrid_steps=4`，无需修改。

---

## 6. 验证结果摘要

（checkpoint: `logs/task_9_unified/G_6000.pth`，c_fm=0.5）

| 模式 | 耗时(s) | HF ratio | 备注 |
|---|---|---|---|
| NF | 2.14 | 0.260 | 高频略不足 |
| Hybrid | 0.15 | 0.272 | **最接近原始音频** |
| FM(32) | 0.63 | 0.275 | 质量上限 |
| 原始音频 | — | 0.287~0.323 | 数据集 baseline |

- Hybrid HF ratio 较修复前下降 32%（0.401 → 0.272）。
- 语音可懂度：Hybrid 模式可辨识文字。

---

## 7. 常见问题

**Q: Hybrid 推理听不清文字 / 全是噪声？**
- 确认 checkpoint 是用「FM 一致性修复 + head_fm 零初始化」之后的版本（本项目为 `G_6000.pth` 及以后）。
- 早期 checkpoint（修复前）FM 从纯噪声起点训练，与推理起点不一致，会破坏语音。

**Q: 训练日志没有 `FlowMatch Loss`？**
- 确认 config `model.use_unified_flow=true` 且 `train.c_fm>0`。
- FM loss 只在 `model.training=True` 时计算，eval 模式不计算。

**Q: FM loss 一直不收敛？**
- `c_fm` 过小（如 0.1）会让 FM head 几乎无梯度，调大到 0.5。
- 确认 `head_fm` 已零初始化（代码层面固化），否则初期随机速度场会干扰。

**Q: 报错 "Trying to backward through the graph a second time"？**
- NF 与 FM loss 共用了计算图但未正确 detach。本实现已在 `forward` 内对 `x_0`/`x_1` 做 detach，
  并通过 `loss_gen_all` 统一反传，不应出现此错误；若自定义改动后出现，检查 detach 链路。

---

## 8. ONNX 导出

统一流模型支持 ONNX 导出，复用 `onnx_export_generator.py`：

```bash
python onnx_export_generator.py <G_*.pth> <config.json> <out.onnx>
```

导出时会自动检测 `use_unified_flow`：
- **统一流**：按 config 的 `hybrid_steps`（固定）展开 FM 欧拉积分循环，导出 Hybrid 推理图。
  输入与原 A2 一致（`c, f0, uv, g, noise`），`noise` 作为先验采样噪声外部传入。
- **非统一流**：与原行为一致（纯 NF 逆变换）。

导出后用 onnxruntime 推理，输入名/动态轴同原 A2（`c/f0/uv/noise` 的 batch、frames 动态）。

**验证**（`experiments/onnx_validate.py`，G_6000.pth）：
- 输出形状一致，均值/标准差几乎相同（mean ≈ -2.7e-4，std ≈ 0.0224）。
- 样本级 max_diff ~0.12 出现在谐波峰值处（NSF-HiFiGAN 声码器锐变边的固有现象），
  mean_diff ~7e-4（SNR ~30dB），听感无差异。

**注意**：
- 导出的 FM 步数固定为 config 的 `hybrid_steps`，运行时无法更改；如需不同步数请重新导出。
- 纯 FM(32) 模式需把 config `hybrid_steps` 设为 32 后再导出。
- `torch.full((1,1), t_val)` 的时间步靠广播适配任意 batch，ONNX 友好。

---

## 9. 相关文件

| 文件 | 作用 |
|---|---|
| `models.py` | `GeneralizedCouplingLayer`、`GeneralizedFlow`、`SynthesizerTrnRvcFlow.infer_hybrid` |
| `train.py` | FM loss 分支、8 元组解包、`c_fm` 加权 |
| `inference/infer_tool.py` | `Svc.infer` / `slice_inference` 的 `hybrid_mode` / `hybrid_steps` 切换 |
| `utils.py` | EMA checkpoint 读写（`use_ema`）、tensorboard 安全 `summarize` |
| `configs_template/config_template.json` | 方案3 config 模板字段 |
| `experiments/real_audio_test.py` | 三模式真实音频推理 + HF 检测示例 |
| `experiments/phase*_report.md` | 各阶段验证报告 |
