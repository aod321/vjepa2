import os

# -- limit visible devices for distributed runs on SLURM
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import gc
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from app.vjepa_reward_done_head.droid import init_data
from app.vjepa_reward_done_head.utils import (
    init_opt,
    init_video_model,
    load_checkpoint,
    load_pretrained,
)
from app.vjepa_droid_decoder.transforms import make_transforms
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
logger = get_logger(__name__, force=True)
_GLOBAL_SEED = 0


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def _compute_pos_weight(labels: torch.Tensor, cfg_value: float, max_pos_weight: float):
    if cfg_value is not None:
        return torch.tensor(cfg_value, device=labels.device, dtype=labels.dtype)
    # dynamic estimate from batch
    pos_rate = labels.float().mean().clamp(min=1e-6)
    weight = (1.0 - pos_rate) / pos_rate
    if max_pos_weight is not None and max_pos_weight > 0:
        weight = weight.clamp(max=max_pos_weight)
    return weight


def _has_trainable_params(module: torch.nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters())


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  CONFIG
    # ----------------------------------------------------------------------- #
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    cfgs_model = args.get("model")
    cfgs_data = args.get("data")
    cfgs_data_aug = args.get("data_aug", {})
    cfgs_loss = args.get("loss")
    cfgs_opt = args.get("optimization")

    os.makedirs(folder, exist_ok=True)

    # -- META
    r_file = cfgs_meta.get("resume_checkpoint", None)
    p_file = cfgs_meta.get("pretrain_checkpoint", None)
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    load_encoder = cfgs_meta.get("load_encoder", True)
    freeze_encoder = cfgs_meta.get("freeze_encoder", True)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    which_dtype = cfgs_meta.get("dtype", "float32")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MODEL
    model_name = cfgs_model.get("model_name", "vit_giant_xformers")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    wide_silu = cfgs_model.get("wide_silu", False)
    patch_size = cfgs_model.get("patch_size", 16)
    tubelet_size = cfgs_model.get("tubelet_size", 1)
    pooling = cfgs_model.get("pooling", "mean")
    head_hidden_dim = cfgs_model.get("head_hidden_dim", 512)
    head_layers = cfgs_model.get("head_layers", 1)
    head_dropout = cfgs_model.get("head_dropout", 0.0)

    # -- DATA
    datasets = cfgs_data.get("datasets", [])
    dataset_path = datasets[0]
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    batch_size = cfgs_data.get("batch_size")
    fps = cfgs_data.get("fps", 10)
    crop_size = cfgs_data.get("crop_size", 256)
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)
    sample_stride = cfgs_data.get("frame_stride", 1)
    tubelet_size_data = cfgs_data.get("tubelet_size", tubelet_size)
    derive_dones = cfgs_data.get("derive_dones", False)
    sample_tail_prob = cfgs_data.get("sample_tail_prob", 0.0)

    # -- LOSS
    mode = cfgs_loss.get("mode", "both").lower()
    reward_weight = float(cfgs_loss.get("reward_weight", 1.0))
    done_weight = float(cfgs_loss.get("done_weight", 1.0))
    pos_weight_reward_cfg = cfgs_loss.get("pos_weight_reward", None)
    pos_weight_done_cfg = cfgs_loss.get("pos_weight_done", None)
    max_pos_weight = cfgs_loss.get("max_pos_weight", 100.0)

    # -- OPT
    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    anneal = cfgs_opt.get("anneal")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    enc_lr_scale = cfgs_opt.get("enc_lr_scale", 1.0)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)

    _set_seed(seed)
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init distributed
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # -- logging paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    resume_path = os.path.join(folder, r_file) if r_file is not None else latest_path
    if not os.path.exists(resume_path):
        resume_path = None

    # -- logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "reward_loss"),
        ("%.5f", "done_loss"),
        ("%.5f", "reward_pos_rate"),
        ("%.5f", "done_pos_rate"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        mode="+a",
    )

    # -- model
    encoder, head, tokens_per_frame = init_video_model(
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
        pooling=pooling,
        head_hidden_dim=head_hidden_dim,
        head_layers=head_layers,
        head_dropout=head_dropout,
    )
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    if _has_trainable_params(encoder):
        if dist.is_available() and dist.is_initialized():
            encoder = DistributedDataParallel(encoder, static_graph=True)
        else:
            logger.info("Distributed not initialized; running encoder without DDP.")
    else:
        logger.info("Encoder has no trainable params; skipping DDP wrap.")

    if dist.is_available() and dist.is_initialized():
        head = DistributedDataParallel(head, static_graph=False)
    else:
        logger.info("Distributed not initialized; running head without DDP.")

    encoder = load_pretrained(
        r_path=p_file,
        encoder=encoder,
        context_encoder_key=context_encoder_key,
        load_encoder=load_encoder,
    )

    start_epoch = 0
    # -- transforms and data
    transform = make_transforms(
        random_horizontal_flip=cfgs_data_aug.get("horizontal_flip", False),
        random_resize_aspect_ratio=cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3]),
        random_resize_scale=cfgs_data_aug.get("random_resize_scale", [0.3, 1.0]),
        reprob=cfgs_data_aug.get("reprob", 0.0),
        auto_augment=cfgs_data_aug.get("auto_augment", False),
        motion_shift=cfgs_data_aug.get("motion_shift", False),
        crop_size=crop_size,
    )
    video_collator = torch.utils.data.default_collate
    loader, sampler = init_data(
        data_path=dataset_path,
        batch_size=batch_size,
        frames_per_clip=max_num_frames,
        fps=fps,
        sample_stride=sample_stride,
        tubelet_size=tubelet_size_data,
        transform=transform,
        collator=video_collator,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        rank=rank,
        sample_tail_prob=sample_tail_prob,
        derive_dones=derive_dones,
    )
    _dlen = len(loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- optimizer
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        head=head,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        enc_lr_scale=enc_lr_scale,
        iterations_per_epoch=ipe,
        anneal=anneal,
        warmup=warmup,
        num_epochs=num_epochs,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )

    if resume_path is not None and os.path.exists(resume_path):
        encoder, head, optimizer, scaler, start_epoch = load_checkpoint(
            r_path=resume_path,
            encoder=encoder,
            head=head,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Failed to save checkpoint: {e}")

    sampler.set_epoch(start_epoch)
    data_iter = iter(loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(data_iter)
            except Exception:
                sampler.set_epoch(start_epoch)
                data_iter = iter(loader)
                _ = next(data_iter)

    gc_disable = cfgs_meta.get("sync_gc", False)
    if gc_disable:
        gc.disable()
        gc.collect()

    # -- training loop
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}")
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        done_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        reward_pos_meter = AverageMeter()
        done_pos_meter = AverageMeter()

        for itr in range(ipe):
            itr_start = time.time()
            retries = 0
            loaded = False
            while not loaded:
                try:
                    sample = next(data_iter)
                    loaded = True
                except StopIteration:
                    sampler.set_epoch(epoch)
                    data_iter = iter(loader)
                except Exception as e:
                    retries += 1
                    if retries > 5:
                        raise e
                    logger.warning(f"Data loading error ({retries}): {e}")
                    time.sleep(2)

            clips = sample[0].to(device, non_blocking=True)  # [B, C, T, H, W]
            rewards = sample[1].to(device, dtype=torch.float, non_blocking=True)  # [B, T]
            dones = sample[2].to(device, dtype=torch.float, non_blocking=True)  # [B, T]
            pos_reward = rewards.mean().item()
            pos_done = dones.mean().item()

            if gc_disable and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                loss_r = torch.tensor(0.0, device=device, dtype=dtype)
                loss_d = torch.tensor(0.0, device=device, dtype=dtype)

                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    tokens = encoder(clips)
                    temporal_tokens = tokens.shape[1] // tokens_per_frame
                    if temporal_tokens != rewards.shape[1]:
                        raise RuntimeError(
                            f"Temporal tokens ({temporal_tokens}) do not match label length ({rewards.shape[1]}). "
                            f"Ensure tubelet_size/frameskip keep frame-to-label alignment."
                        )
                    logits = head(tokens)
                    if mode in {"reward", "both"}:
                        pos_weight_r = _compute_pos_weight(rewards, pos_weight_reward_cfg, max_pos_weight)
                        bce_r = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits["reward"], rewards, pos_weight=pos_weight_r
                        )
                        loss_r = reward_weight * bce_r
                    if mode in {"done", "both"}:
                        pos_weight_d = _compute_pos_weight(dones, pos_weight_done_cfg, max_pos_weight)
                        bce_d = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits["done"], dones, pos_weight=pos_weight_d
                        )
                        loss_d = done_weight * bce_d
                    loss = loss_r + loss_d

                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                return (
                    float(loss),
                    float(loss_r),
                    float(loss_d),
                    _new_lr,
                    _new_wd,
                )

            (loss, l_reward, l_done, _new_lr, _new_wd), gpu_time = gpu_timer(train_step)
            iter_time = (time.time() - itr_start) * 1000.0
            loss_meter.update(loss)
            reward_meter.update(l_reward)
            done_meter.update(l_done)
            iter_time_meter.update(iter_time)
            gpu_time_meter.update(gpu_time)
            reward_pos_meter.update(pos_reward)
            done_pos_meter.update(pos_done)

            csv_logger.log(
                epoch + 1,
                itr,
                loss,
                l_reward,
                l_done,
                pos_reward,
                pos_done,
                iter_time,
                gpu_time,
            )

            if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                logger.info(
                    f"[{epoch+1}, {itr:05d}] loss {loss_meter.avg:.4f} "
                    f"(reward {reward_meter.avg:.4f}, done {done_meter.avg:.4f}) "
                    f"[pos rates: r {reward_pos_meter.avg:.4f}, d {done_pos_meter.avg:.4f}] "
                    f"[lr {_new_lr:.3e}] [wd {_new_wd:.3e}] "
                    f"[iter {iter_time_meter.avg:.1f} ms] [gpu {gpu_time_meter.avg:.1f} ms]"
                )

            if (itr + 1) % max(CHECKPOINT_FREQ, 1) == 0:
                save_checkpoint(epoch, latest_path)
            if (save_every_freq > 0) and ((itr + 1) % save_every_freq == 0):
                save_checkpoint(epoch, os.path.join(folder, f"ckpt_e{epoch}_itr{itr}.pt"))

        # end epoch
        save_checkpoint(epoch, latest_path)


if __name__ == "__main__":
    raise SystemExit("Please launch via the main training harness.")
