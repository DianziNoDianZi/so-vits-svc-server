"""
Modified from upstream so-vits-svc (AGPL-3.0, svc-develop-team).
Changes: CPU training support, max_steps/auto_stop, validation-loss logging.
See NOTICE. 2026-08.
"""
import logging
import multiprocessing
import os
import time
import warnings

warnings.filterwarnings('ignore', '.*pynvml.*')
warnings.filterwarnings('ignore', '.*GradScaler.*', category=FutureWarning)
warnings.filterwarnings('ignore', '.*autocast.*', category=FutureWarning)
warnings.filterwarnings('ignore', '.*weight_norm.*', category=FutureWarning)

import torch

_load = torch.load
torch.load = lambda *a, **kw: _load(*a, **{**kw, 'weights_only': False})
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import modules.commons as commons
import utils
from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
    SynthesizerTrnRvc,
    SynthesizerTrnRvcFlow,
)
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


class _MaxStepsReached(Exception):
    """达到 max_steps 或 auto_stop 条件时提前结束训练。"""
    pass


_best_ref = {'loss': None, 'step': 0}


def _apply_d_lr_scale(hps, optim_g, optim_d, step):
    """RVC 式架构可选：训练初期压低判别器 lr，防止判别器过早碾压生成器。
    通过 hps.train.d_lr_scale 开启（默认 1.0 不生效），前 D_LR_SCALE_STEPS 步生效。"""
    D_LR_SCALE_STEPS = 1000
    scale = float(hps.train.get('d_lr_scale') or 1.0)
    if scale <= 0 or scale >= 1.0:
        return
    # 记录基准 lr（每次从调度器/初始值同步一次，避免 warmup 干扰）
    if not hasattr(optim_d, '_base_lr'):
        optim_d._base_lr = [pg['lr'] for pg in optim_d.param_groups]
        optim_g._base_lr = [pg['lr'] for pg in optim_g.param_groups]
    for pg, base in zip(optim_d.param_groups, optim_d._base_lr):
        pg['lr'] = base * scale if step < D_LR_SCALE_STEPS else base
    for pg, base in zip(optim_g.param_groups, optim_g._base_lr):
        pg['lr'] = base


def main():
    """Single Node Multi GPUs or CPU Training"""
    hps = utils.get_hparams()

    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        run(0, 1, hps)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = hps.train.port
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    
    single = n_gpus <= 1
    if not single:
        dist.init_process_group(backend='gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    collate_fn = TextAudioCollate()
    all_in_mem = hps.train.all_in_mem   # If you have enough memory, turn on this option to avoid disk IO and speed up training.
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps, all_in_mem=all_in_mem)
    # 默认单进程加载数据；内存充足的机器可在 config 中设置 num_workers 加速
    num_workers = getattr(hps.train, 'num_workers', 0) or 0
    if all_in_mem:
        num_workers = 0
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
                              batch_size=hps.train.batch_size, collate_fn=collate_fn)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps, all_in_mem=all_in_mem,vol_aug = False)
        eval_loader = DataLoader(eval_dataset, num_workers=min(num_workers, 1), shuffle=False,
                                 batch_size=1, pin_memory=False,
                                 drop_last=False, collate_fn=collate_fn)

    arch = hps.model.get('arch') or 'sovits-v1'
    if arch == 'rvc':
        _g_cls = SynthesizerTrnRvc
    elif arch == 'rvc-flow':
        _g_cls = SynthesizerTrnRvcFlow
    else:
        _g_cls = SynthesizerTrn
    net_g = _g_cls(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    if not single:
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])

    skip_optimizer = False
    try:
        ckpt_g = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        ckpt_d = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        is_base = 'G_0.pth' in ckpt_g or 'G_0.pth' in ckpt_d
        if is_base and hps.model.ssl_dim != 768:
            skip_optimizer = True
        _, _, _, epoch_str = utils.load_checkpoint(ckpt_g, net_g, optim_g, skip_optimizer)
        _, _, _, epoch_str = utils.load_checkpoint(ckpt_d, net_d, optim_d, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        name=utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        global_step=int(name[name.rfind("_")+1:name.rfind(".")])+1
    except Exception:
        print("load old checkpoint failed...skip_optimizer=True")
        skip_optimizer = True
        try:
            _, _, _, epoch_str = utils.load_checkpoint(ckpt_g, net_g, optim_g, skip_optimizer)
            _, _, _, epoch_str = utils.load_checkpoint(ckpt_d, net_d, optim_d, skip_optimizer)
            epoch_str = max(epoch_str, 1)
            global_step = 0
        except Exception as e2:
            print(f"load failed again: {e2}")
            epoch_str = 1
            global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0
        global_step = 0

    warmup_epoch = hps.train.warmup_epochs
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    # CPU 不支持 fp16 混合精度，自动关闭
    scaler = GradScaler(enabled=hps.train.fp16_run and torch.cuda.is_available())

    for epoch in range(epoch_str, hps.train.epochs + 1):
        # set up warm-up learning rate
        if epoch <= warmup_epoch:
            for param_group in optim_g.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
            for param_group in optim_d.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
        # training
        try:
            if rank == 0:
                train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                                   [train_loader, eval_loader], logger, [writer, writer_eval])
            else:
                train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                                   [train_loader, None], None, None)
        except _MaxStepsReached:
            if rank == 0:
                logger.info(f'====> 达到 max_steps/auto_stop 条件，提前停止于 step {global_step}')
            break
        # update learning rate
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers
    
    half_type = torch.bfloat16 if hps.train.half_type=="bf16" else torch.float16
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    _apply_d_lr_scale(hps, optim_g, optim_d, global_step)
    for batch_idx, items in enumerate(train_loader):
        c, f0, spec, y, spk, lengths, uv,volume = items
        g = spk.to(device, non_blocking=True)
        spec, y = spec.to(device, non_blocking=True), y.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        f0 = f0.to(device, non_blocking=True)
        uv = uv.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)
        
        with autocast(enabled=hps.train.fp16_run, dtype=half_type):
            y_hat, ids_slice, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(c, f0, uv, spec, g=g, c_lengths=lengths,
                                                                                spec_lengths=lengths,vol = volume)

            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

            with autocast(enabled=False, dtype=half_type):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)
        

        with autocast(enabled=hps.train.fp16_run, dtype=half_type):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False, dtype=half_type):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                # RVC 式架构无 flow（z_p is None），KL 项为 0（用 0 维张量，避免 int 参与 .item() 崩溃）
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl if z_p is not None else torch.zeros((), device=y_hat.device)
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                _g = net_g.module if hasattr(net_g, 'module') else net_g
                loss_lf0 = F.mse_loss(pred_lf0, lf0) if _g.use_automatic_f0_prediction else 0
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
        _apply_d_lr_scale(hps, optim_g, optim_d, global_step + 1)

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                reference_loss=0
                for i in losses:
                    reference_loss += i
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}, reference_loss: {reference_loss}")

                # auto_stop：loss 在 auto_stop 步内无改善则提前停止
                auto_stop = getattr(hps.train, 'auto_stop', 0) or 0
                if auto_stop > 0:
                    ref = float(reference_loss.detach())
                    if _best_ref['loss'] is None or ref < _best_ref['loss'] - 1e-9:
                        _best_ref['loss'] = ref
                        _best_ref['step'] = global_step
                    elif global_step - _best_ref['step'] >= auto_stop:
                        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                              os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                              os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                        keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                        if keep_ckpts > 0:
                            utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                        raise _MaxStepsReached()

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl,
                                    "loss/g/lf0": loss_lf0})

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())
                }

                if _g.use_automatic_f0_prediction:
                    image_dict.update({
                        "all/lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                              pred_lf0[0, 0, :].detach().cpu().numpy()),
                        "all/norm_lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                                   norm_lf0[0, 0, :].detach().cpu().numpy())
                    })

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )

            if global_step > 0 and global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval, logger)
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

        global_step += 1
        max_steps = getattr(hps.train, 'max_steps', 0) or 0
        if max_steps > 0 and global_step >= max_steps:
            if rank == 0:
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
            raise _MaxStepsReached()

    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, '.2f')
        logger.info(f'====> Epoch: {epoch}, cost {durtaion} s')
        start_time = now


def evaluate(hps, generator, eval_loader, writer_eval, logger=None):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    eval_mel_sum = 0.0
    eval_count = 0
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, _, uv,volume = items
            g = spk[:1].to(device)
            spec, y = spec[:1].to(device), y[:1].to(device)
            c = c[:1].to(device)
            f0 = f0[:1].to(device)
            uv = uv[:1].to(device)
            if volume is not None:
                volume = volume[:1].to(device)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            g_module = generator.module if hasattr(generator, 'module') else generator
            y_hat,_ = g_module.infer(c, f0, uv, g=g,vol = volume)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            # 验证集重建误差（用于过拟合监测）
            t = min(mel.size(-1), y_hat_mel.size(-1))
            eval_mel_sum += float(F.l1_loss(mel[..., :t], y_hat_mel[..., :t]))
            eval_count += 1

            audio_dict.update({
                f"gen/audio_{batch_idx}": y_hat[0],
                f"gt/audio_{batch_idx}": y[0]
            })
        image_dict.update({
            "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
        })
    if eval_count > 0 and logger is not None:
        logger.info(f"Eval Losses: [{eval_mel_sum / eval_count:.4f}], step: {global_step}")
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    main()
