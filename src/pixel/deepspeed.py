from copy import deepcopy
import logging

import torch
from transformers.deepspeed import deepspeed_optim_sched

logger = logging.getLogger(__name__)


def deepspeed_init(trainer, num_training_steps, resume_from_checkpoint=None, inference=False):
    """
    Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

    If `resume_from_checkpoint` was passed then an attempt to resume from a previously saved checkpoint will be made.

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu
        resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load
        inference: launch in inference mode (no optimizer and no lr scheduler)

    Returns: model, optimizer, lr_scheduler

    We may use `deepspeed_init` more than once during the life of Trainer, when we do - it's a temp hack based on:
    https://github.com/microsoft/DeepSpeed/issues/1394#issuecomment-937405374 until Deepspeed fixes a bug where it
    can't resume from a checkpoint after it did some stepping https://github.com/microsoft/DeepSpeed/issues/1612

    """
    import deepspeed
    from deepspeed.utils import logger as ds_logger

    model = trainer.model
    args = trainer.args

    if hasattr(trainer, "hf_deepspeed_config_orig"):
        hf_deepspeed_config = deepcopy(trainer.hf_deepspeed_config_orig)
    else:
        hf_deepspeed_config = args.hf_deepspeed_config
        trainer.hf_deepspeed_config_orig = deepcopy(args.hf_deepspeed_config)

    # resume config update - some bits like `model` and `num_training_steps` only become available during train
    hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)
    config = hf_deepspeed_config.config

    # set the Deepspeed log level consistent with the Trainer
    ds_logger.setLevel(args.get_process_log_level())

    if inference:
        # only Z3 makes sense for the inference
        if not hf_deepspeed_config.is_zero3():
            raise ValueError("ZeRO inference only makes sense with ZeRO Stage 3 - please adjust your config")

        # in case the training config is re-used for inference
        hf_deepspeed_config.del_config_sub_tree("optimizer")
        hf_deepspeed_config.del_config_sub_tree("lr_scheduler")
        optimizer, lr_scheduler = None, None
        model_parameters = None
    else:
        trainer.optimizer = None  # important for when deepspeed_init is used as re-init
        optimizer, lr_scheduler = deepspeed_optim_sched(trainer, hf_deepspeed_config, args, num_training_steps)
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    # keep for quick debug:
    # from pprint import pprint; pprint(config)

    kwargs = {
        "model": model,
        "model_parameters": model_parameters,
        "config_params": config,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }

    deepspeed_engine, optimizer, _, lr_scheduler = deepspeed.initialize(**kwargs)

    if resume_from_checkpoint is not None:
        # it's possible that the user is trying to resume from model_path, which doesn't necessarily
        # contain a deepspeed checkpoint. e.g. examples just check if the dir exists and assume it's
        # a resume from a checkpoint and not just a local pretrained weight. So we check here if the
        # path contains what looks like a deepspeed checkpoint
        import glob

        deepspeed_checkpoint_dirs = sorted(glob.glob(f"{resume_from_checkpoint}/global_step*"))

        if len(deepspeed_checkpoint_dirs) > 0:
            logger.info(f"Attempting to resume from {resume_from_checkpoint}")
            # this magically updates self.optimizer and self.lr_scheduler
            load_path, _ = deepspeed_engine.load_checkpoint(
                resume_from_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
            )
            if trainer.args.fp16:
                logger.info("Casting deepspeed engine to fp16")
                deepspeed_engine = deepspeed_engine.to(torch.float16)
            elif trainer.args.bf16:
                logger.info("Casting deepspeed engine to bf16")
                deepspeed_engine = deepspeed_engine.to(torch.bfloat16)

            if load_path is None:
                raise ValueError(f"[deepspeed] failed to resume from checkpoint {resume_from_checkpoint}")
        else:
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

    return deepspeed_engine, optimizer, lr_scheduler