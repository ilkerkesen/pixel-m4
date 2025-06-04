import copy
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, Optional

import torch
import wandb
from einops import rearrange
from transformers import TrainingArguments, logging

from .misc import format_img, format_mask, mark_answer


logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)


class Modality(Enum):
    IMAGE = auto()
    TEXT = auto()


@dataclass
class PIXELTrainingArguments(TrainingArguments):
    """
    Custom training arguments that include parameters for early stopping and prediction logging
    """

    early_stopping: Optional[bool] = field(default=True, metadata={"help": "Whether to train with early stopping."})
    early_stopping_patience: Optional[int] = field(
        default=5,
        metadata={
            "help": "Number of evaluation steps without increase in validation performance "
            "until training is terminated if training with early stopping."
        },
    )
    log_predictions: Optional[bool] = field(
        default=False, metadata={"help": "Whether to log predictions to file and wandb."}
    )
    min_learning_rate: Optional[float] = field(
        default=1e-5, metadata={"help": "Minimum learning rate. Only used for pretraining"}
    )


def debug_log_inputs(inputs: Dict[str, torch.Tensor]):
    """
    Logs inputs as square images to wandb
    Only works when training with Modality.IMAGE
    """

    wandb.init(reinit=False)

    input_dict = copy.deepcopy(inputs)

    if len(inputs["pixel_values"].shape) == 5:
        input_dict["pixel_values"] = rearrange(input_dict["pixel_values"], "b n c h w -> (b n) c h w")
        input_dict["attention_mask"] = rearrange(input_dict["attention_mask"], "b n s -> (b n) s")
        if "patch_mask" in input_dict:
            input_dict["patch_mask"] = rearrange(input_dict["patch_mask"], "b n s -> (b n) s")

    images = [wandb.Image(format_img(im)) for im in input_dict["pixel_values"]]
    attention_masks = [wandb.Image(format_mask(am)) for am in input_dict["attention_mask"]]
    seq_length = len(input_dict["attention_mask"][0])
    wandb.log(
        {
            "images": images,
            "attention_masks": attention_masks,
        }
    )

    if "patch_mask" in input_dict:
        patch_masks = [wandb.Image(format_mask(pm)) for pm in input_dict["patch_mask"]]
        wandb.log({"patch_masks": patch_masks})

    if "start_positions" in input_dict and "end_positions" in input_dict:
        marked_answers = [
            wandb.Image(format_mask(mark_answer(s, e, seq_length)))
            for s, e in zip(input_dict["start_positions"], input_dict["end_positions"])
        ]
        wandb.log({"answer_spans": marked_answers})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    train_dataset_names: str = field(metadata={"help": "Name of train dataset in HuggingFace dataset hub"})
    train_splits: str = field(metadata={"help": "Name of the training dataset split."})
    validation_dataset_name: str = field(metadata={"help": "Name of validation dataset in HuggingFace dataset hub"})
    validation_split: str = field(metadata={"help": "Name of the validation dataset split."})
    data_dir: str = field(default=None, metadata={"help": "Local directory that contains the preprocessed datasets"})
    dataset_caches: Optional[str] = field(default=None, metadata={"help": "Directory where the dataset is cached"})
    train_dataset_configs: str = field(default=None, metadata={"help": "Train dataset config/subset"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    streaming: Optional[bool] = field(default=False, metadata={"help": "Whether to stream the training dataset. Only works if supported by dataset."})
    do_normalize: Optional[bool] = field(
        default=False, metadata={"help": "Whether to normalize to model's feature extractor's mean and std."}
    )

    def __post_init__(self):
        self.train_dataset_names = self.train_dataset_names.split(",")
        self.train_splits = self.train_splits.split(",")
        if self.train_dataset_configs:
            self.train_dataset_configs = self.train_dataset_configs.split(",")
        else:
            self.train_dataset_configs = [None] * len(self.train_dataset_names)
        if self.dataset_caches:
            self.dataset_caches = self.dataset_caches.split(",")
        else:
            self.dataset_caches = [None] * len(self.train_dataset_names)
        assert (
            len(self.train_dataset_names)
            == len(self.train_splits)
            == len(self.train_dataset_configs)
            == len(self.dataset_caches)
        )

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/feature extractor we are going to pre-train.
    """

    text_renderer_name_or_path: str = field(
        metadata={
            "help": "Path / Huggingface identifier of the text renderer that was used to prerender the "
            "training/validation data."
        }
    )
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: str = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    mask_ratio: float = field(
        default=0.25, metadata={"help": "The ratio of the number of masked tokens in the input sequence."}
    )
    norm_pix_loss: bool = field(
        default=True, metadata={"help": "Whether or not to train with normalized pixel values as target."}
    )
    span_masking: bool = field(
        default=False, metadata={"help": "Whether to use span masking instead of random masking."}
    )
    masking_max_span_length: Optional[int] = field(
        default=None, metadata={"help": "Maximum span length that can be masked when using span masking."}
    )
    masking_spacing: Optional[int] = field(
        default=None,
        metadata={
            "help": "Spacing between masked spans. Defaults to the length of the span."
            "Use this argument to set it to a fixed number of patches."
            "Recommended setting: For masking ratio <= 0.4 leave the default"
            "For ratios between 0.4 and 0.7 set it to 1. For higher, set it to 0"
        },
    )
    masking_cumulative_span_weights: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated list of cumulative probabilities of sampling a span of length n"
            "when using span masking. Must be a list of size model_args.masking_max_span_length."
        },
    )
    dropout_prob: float = field(
        default=0.1, metadata={"help": "Dropout probability for attention blocks"}
    )

    def __post_init__(self):
        if self.masking_cumulative_span_weights is not None:
            self.masking_cumulative_span_weights = [float(w) for w in self.masking_cumulative_span_weights.split(",")]

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class CustomTrainingArguments(TrainingArguments):
    base_learning_rate: float = field(
        default=1.5e-4, metadata={"help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."}
    )
    min_learning_rate: float = field(
        default=1e-5, metadata={"help": "Minimum learning rate. Only used for pretraining"}
    )