# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch

import src.models.vision_transformer as video_vit
from src.models.vjepa2_decoder import VJEPA2FrameDecoder
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, WSDSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_pretrained(
    r_path,
    encoder=None,
    context_encoder_key="encoder",
    load_encoder=True,
):
    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    if load_encoder:
        # -- loading encoder
        pretrained_dict = checkpoint[context_encoder_key]
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        if "module." not in encoder.state_dict().keys() and "module." in next(iter(pretrained_dict)):
            pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained encoder with msg: {msg}")

    del checkpoint
    return encoder


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt=None,
    scaler=None,
    replace_kw=["backbone."],
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint["epoch"]

    # -- loading encoder
    pretrained_dict = checkpoint["encoder"]
    for kw in replace_kw:
        pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    # -- loading predictor
    pretrained_dict = checkpoint["predictor"]
    for kw in replace_kw:
        pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
    msg = predictor.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    # -- loading target_encoder
    if target_encoder is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint["target_encoder"]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    if opt is not None:
        opt.load_state_dict(checkpoint["opt"])

    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"loaded optimizers from epoch {epoch}")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    pred_is_frame_causal=True,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    action_embed_dim=7,
    use_extrinsics=False,
    old_pred=False,
    effective_action_dims=None,
    decoder_embed_dim=None,
    decoder_depth=None,
    decoder_num_heads=None,
    decoder_mlp_ratio=4.0,
    decoder_dropout=0.0,
    decoder_attn_dropout=0.0,
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
    decoder_embed_dim = decoder_embed_dim or encoder.embed_dim
    decoder_depth = decoder_depth or encoder.get_num_layers()
    decoder_num_heads = decoder_num_heads or encoder.num_heads

    decoder = VJEPA2FrameDecoder(
        encoder_dim=encoder.embed_dim,
        image_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        channels=3,
        decoder_embed_dim=decoder_embed_dim,
        depth=decoder_depth,
        num_heads=decoder_num_heads,
        mlp_ratio=decoder_mlp_ratio,
        dropout=decoder_dropout,
        attention_dropout=decoder_attn_dropout,
    )
    encoder.to(device)
    decoder.to(device)
    logger.info(encoder)
    logger.info(decoder)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Decoder number of parameters: {count_parameters(decoder)}")

    return encoder, decoder


def init_opt(
    encoder,
    decoder,
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

    dec_main = filtered_params(decoder, lambda n, p: ("bias" not in n) and (len(p.shape) != 1))
    if dec_main:
        param_groups.append({"params": dec_main})

    enc_bias = filtered_params(encoder, lambda n, p: ("bias" in n) or (len(p.shape) == 1))
    if enc_bias:
        param_groups.append(
            {
                "params": enc_bias,
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
                "lr_scale": enc_lr_scale,
            }
        )

    dec_bias = filtered_params(decoder, lambda n, p: ("bias" in n) or (len(p.shape) == 1))
    if dec_bias:
        param_groups.append(
            {
                "params": dec_bias,
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
            }
        )

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
