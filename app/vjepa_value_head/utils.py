import logging
import sys

import torch

import src.models.vision_transformer as video_vit
from src.models.value_head import ValueHead
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, WSDSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def _unwrap_distributed_module(module):
    if module is None:
        return None
    return getattr(module, "module", module)


def _clean_pretrained_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key.replace("backbone.", "")
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        cleaned[new_key] = value
    return cleaned


def load_pretrained(
    r_path,
    encoder=None,
    context_encoder_key="encoder",
    load_encoder=True,
):
    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    if load_encoder:
        pretrained_dict = _clean_pretrained_state_dict(checkpoint[context_encoder_key])
        encoder_to_load = _unwrap_distributed_module(encoder)
        # Drop keys whose shapes do not match current module (e.g., different tubelet_size)
        current_state = encoder_to_load.state_dict()
        filtered = {}
        dropped = []
        for k, v in pretrained_dict.items():
            if k in current_state and current_state[k].shape == v.shape:
                filtered[k] = v
            else:
                dropped.append(k)
        if dropped:
            logger.warning(
                f"Dropping {len(dropped)} keys due to shape mismatch: "
                f"{dropped[:5]}{' ...' if len(dropped) > 5 else ''}"
            )
        msg = encoder_to_load.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded pretrained encoder with msg: {msg}")

    del checkpoint
    return encoder


def load_checkpoint(
    r_path,
    encoder,
    head=None,
    opt=None,
    scaler=None,
    replace_kw=("backbone.",),
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]

    # Encoder
    pretrained_dict = checkpoint["encoder"]
    for kw in replace_kw:
        pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"Loaded encoder from epoch {epoch} with msg: {msg}")

    # Head
    if head is not None and "head" in checkpoint:
        msg = head.load_state_dict(checkpoint["head"], strict=False)
        logger.info(f"Loaded head from epoch {epoch} with msg: {msg}")

    # Optimizer / scaler
    if opt is not None and "opt" in checkpoint:
        opt.load_state_dict(checkpoint["opt"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"Loaded optimizer/scaler from epoch {epoch}")
    del checkpoint
    return encoder, head, opt, scaler, epoch


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=1,
    model_name="vit_giant_xformers",
    crop_size=256,
    uniform_power=False,
    use_sdpa=False,
    use_silu=False,
    wide_silu=False,
    use_activation_checkpointing=False,
    use_rope=False,
    pooling="mean",
    head_hidden_dim=512,
    head_layers=1,
    head_dropout=0.0,
    head_num_actions=None,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )
    tokens_per_frame = (crop_size // patch_size) ** 2
    head = ValueHead(
        encoder_dim=encoder.embed_dim,
        tokens_per_frame=tokens_per_frame,
        hidden_dim=head_hidden_dim,
        num_layers=head_layers,
        dropout=head_dropout,
        pooling=pooling,
        policy_num_actions=head_num_actions,
    )
    encoder.to(device)
    head.to(device)
    logger.info(encoder)
    logger.info(head)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder parameters: {count_parameters(encoder)}")
    logger.info(f"Head parameters: {count_parameters(head)}")
    return encoder, head, tokens_per_frame


def init_opt(
    encoder,
    head,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
):
    def filtered_params(module, predicate):
        if module is None:
            return []
        return [p for name, p in module.named_parameters() if predicate(name, p) and p.requires_grad]

    param_groups = []

    enc_main = filtered_params(encoder, lambda n, p: ("bias" not in n) and (len(p.shape) != 1))
    if enc_main:
        param_groups.append({"params": enc_main, "lr_scale": enc_lr_scale})

    enc_bias = filtered_params(encoder, lambda n, p: ("bias" in n) or (len(p.shape) == 1))
    if enc_bias:
        param_groups.append(
            {"params": enc_bias, "WD_exclude": zero_init_bias_wd, "weight_decay": 0, "lr_scale": enc_lr_scale}
        )

    head_main = filtered_params(head, lambda n, p: ("bias" not in n) and (len(p.shape) != 1))
    if head_main:
        param_groups.append({"params": head_main})
    head_bias = filtered_params(head, lambda n, p: ("bias" in n) or (len(p.shape) == 1))
    if head_bias:
        param_groups.append({"params": head_bias, "WD_exclude": zero_init_bias_wd, "weight_decay": 0})

    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer.")

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
