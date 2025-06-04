#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# NOTE bgg / BGG is the new abbreviation for Bigrams within-words
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import datasets
import wandb
from PIL import Image
import torch
import transformers
from datasets import interleave_datasets, load_from_disk, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from time import time

from pixel import (
    PIXELConfig,
    PIXELEmbeddings,
    PIXELForPreTraining,
    PIXELTrainerForPretraining,
    PIXELTrainer,
    SpanMaskingGenerator,
    # PangoCairoTextRenderer,
    get_attention_mask,
    get_transforms,
    get_2d_sincos_pos_embed
)
from transformers import HfArgumentParser, TrainingArguments, ViTFeatureExtractor
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from pixel.data.rendering.pangocairo_renderer_bigrams_iso_char import PangoCairoTextRenderer 
# from pixel.data.rendering.pangocairo_renderer_bigrams_iso_char_wspace import PangoCairoTextRenderer 
""" Pre-training a PIXEL model as an MAE (masked autoencoder)"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install ./datasets")

#MBERT_LANGS = ["af", "an", "ar", "ast", "az", "azb", "ba", "bar", "be", "bg", "bn", "bpy", "br", "bs", "ca", "ce", "ceb", "cs", "cv", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fr", "fy", "ga", "gl", "gu", "he", "hi", "hr", "ht", "hu", "hy", "id", "io", "is", "it", "ja", "jv", "ka", "kk", "kn", "ko", "ky", "la", "lb", "lmo", "lt", "lv", "mg", "min", "mk", "ml", "mn", "mr", "ms", "my", "nds", "ne", "new", "nl", "nn", "no", "oc", "pa", "pl", "pms", "pnb", "pt", "ro", "ru", "scn", "sco", "sh", "sk", "sl", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "vo", "war", "yo", "zh", "zh_classical"]
#MBERT_LANGS = ["en", "vi", "ru", "zh", "zh_classical", "ja", "ko", "ar", "ur", "hi", "he", "ta", "bn", "th", "te", "el", "hy", "my", "ka", "ml", "kn", "gu", "pa"]
# MBERT_LANGS = ["af"]

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: str = field(
        default=None,
        metadata={
            "help": "Directory on disk containing the preprocessed datasets"
        }
    )
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
    do_normalize: Optional[bool] = field(
        default=False, metadata={"help": "Whether to normalize to model's feature extractor's mean and std."}
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


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    inputs = {"pixel_values": pixel_values, "attention_mask": attention_mask}
    if "patch_mask" in examples[0]:
        patch_mask = torch.stack([example["patch_mask"] for example in examples])
        inputs.update({"patch_mask": patch_mask})
    return inputs


def main(config_dict: Dict[str, Any] = None):

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if not config_dict:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, training_args = parser.parse_dict(config_dict)

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Using Torch {torch.__version__}, CUDA {torch.version.cuda}")
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # train_datasets = []
    # eval_datasets = []
    # for lang in MBERT_LANGS:
    #     dataset = load_from_disk(os.path.join(data_args.data_dir, lang)).train_test_split(test_size=0.0001, seed=training_args.seed)
    #     train_datasets.append(dataset["train"])
    #     eval_datasets.append(dataset["test"])

    # # Exponential smoothing to oversample low-resource languages and undersample high-resource languages
    # coeff = 0.7
    # probs = np.array([1.0 * len(d) for d in train_datasets])
    # probs /= probs.sum()
    # probs = np.array([p ** coeff for p in probs])
    # probs /= probs.sum()

    # logger.info("***** Interleaving training datasets *****")
    # for lang, prob in zip(MBERT_LANGS, probs):
    #     logger.info(f"\tlang = {lang}, sampling probability = {prob:.4f}")

    # train_dataset = interleave_datasets(train_datasets, probabilities=probs, seed=training_args.seed, stopping_strategy="all_exhausted")
    # validation_dataset = interleave_datasets(eval_datasets, probabilities=probs, seed=training_args.seed, stopping_strategy="all_exhausted")

    # logger.info(f"Size of combined training dataset = {len(train_dataset)}")
    # logger.info(f"Size of combined validation dataset = {len(validation_dataset)}")
    # for dataset_name in ["preprocessed_bookcorpus_sep_whitespace_filtered", "preprocessed_wikipedia_sep_whitespace_filtered"]:
    #     logger.info(f"Loading {dataset_name} from {data_args.data_dir}")

    #     dataset = load_from_disk(os.path.join(data_args.data_dir, dataset_name)).train_test_split(
    #         test_size=0.0001, seed=training_args.seed
    #     )
    #     train_datasets.append(dataset["train"])
    #     eval_datasets.append(dataset["test"])

    # logger.info("***** Concatenating datasets *****")

    # train_dataset = concatenate_datasets(train_datasets).shuffle(seed=training_args.seed)
    # validation_dataset = concatenate_datasets(eval_datasets).shuffle(seed=training_args.seed)
    

    # logger.info(f"Size of combined training dataset = {len(train_dataset)}")
    # logger.info(f"Size of combined validation dataset = {len(validation_dataset)}")
    
    # train_shards = {"train": [os.path.join(data_args.data_dir, f) for f in 
    #                           os.listdir(data_args.data_dir) if f.endswith('.arrow')]}
    EN_ARROW_FILES = 51 // 2 + 1
    HI_ARROW_FILES = 655 // 2 + 2
    UK_ARROW_FILES = 335 // 2 + 2
    ZH_ARROW_FILES = 505 // 2 + 2 
    
    #FIXME: fix these hardcoded paths
    en_path = "path/to/data/preprocessed-c4-train/preprocessed_c4_bigrams_529_sep_wspace_cleaned_rmCo"
    hi_path = "path/to/data/preprocessed-mc4-train/preprocessed_mc4_bigrams_529_hi"
    uk_path = "path/to/data/preprocessed-mc4-train/preprocessed_mc4_bigrams_529_uk"
    zh_path = "path/to/data/preprocessed-mc4-train/preprocessed_mc4_bigrams_529_zh"
    
    rng = np.random.default_rng(training_args.seed) # specify seed for reproducibility
    
    en_train_shards = [os.path.join(en_path, f) for f in os.listdir(en_path) if f.endswith('.arrow')]
    chosen_en_files = rng.choice(en_train_shards, EN_ARROW_FILES, replace=False).tolist()
    # Due to OOM errors, we need to split the training data into two batches
    second_chosen_en_files = rng.choice([f for f in en_train_shards if f not in chosen_en_files], EN_ARROW_FILES, replace=False).tolist() 
    # en_train_dataset = load_dataset("arrow", data_files=chosen_en_files, split="train")

    hi_train_shards = [os.path.join(hi_path, f) for f in os.listdir(hi_path) if f.endswith('.arrow')]
    chosen_hi_files = rng.choice(hi_train_shards, HI_ARROW_FILES, replace=False).tolist()
    second_chosen_hi_files = rng.choice([f for f in hi_train_shards if f not in chosen_hi_files], HI_ARROW_FILES, replace=False).tolist()
    # hi_train_dataset = load_dataset("arrow", data_files=chosen_hi_files, split="train")
    
    uk_train_shards = [os.path.join(uk_path, f) for f in os.listdir(uk_path) if f.endswith('.arrow')]
    chosen_uk_files = rng.choice(uk_train_shards, UK_ARROW_FILES, replace=False).tolist()
    second_chosen_uk_files = rng.choice([f for f in uk_train_shards if f not in chosen_uk_files], UK_ARROW_FILES, replace=False).tolist()
    # uk_train_dataset = load_dataset("arrow", data_files=chosen_uk_files, split="train")
    
    zh_train_shards = [os.path.join(zh_path, f) for f in os.listdir(zh_path) if f.endswith('.arrow')]
    chosen_zh_files = rng.choice(zh_train_shards, ZH_ARROW_FILES, replace=False).tolist()
    second_chosen_zh_files = rng.choice([f for f in zh_train_shards if f not in chosen_zh_files], ZH_ARROW_FILES, replace=False).tolist()
    # zh_train_dataset = load_dataset("arrow", data_files=chosen_zh_files, split="train")
    
    assert len(en_train_shards) > 0, "No en train shards found"
    assert len(hi_train_shards) > 0, "No hi train shards found"
    assert len(uk_train_shards) > 0, "No uk train shards found"
    assert len(zh_train_shards) > 0, "No zh train shards found"
    # not_chosen_arrow_files = [f for f in train_shards if f not in chosen_arrow_files]
    # second_batch_arrow_files = rng.choice(not_chosen_arrow_files, 96, replace=False).tolist()
    
    logger.info("***** Interleaving training datasets *****")
    # Probabilities
    probs = [0.25, 0.25, 0.25, 0.25]
    train_dataset = interleave_datasets(
        [
            load_dataset("arrow", data_files=second_chosen_en_files, split="train"), 
            load_dataset("arrow", data_files=second_chosen_hi_files, split="train"), 
            load_dataset("arrow", data_files=second_chosen_uk_files, split="train"), 
            load_dataset("arrow", data_files=second_chosen_zh_files, split="train")
        ], 
        probabilities=probs, 
        seed=training_args.seed,
        stopping_strategy="first_exhausted"
        )
    
    logger.info("***** Done interleaving datasets *****")
    # train_dataset = load_dataset("arrow", data_files=second_batch_arrow_files, split="train", keep_in_memory=False)
    validation_dataset = load_dataset("arrow", data_files=zh_train_shards[0], split="train", keep_in_memory=False)
        
    # Filter the datasets
    start_time = time()
    # NUM_CORES_FILTER = 12 # NOTE hardcoded for now during experiments
    train_dataset = train_dataset.filter(lambda x: bool(x['text']))
    validation_dataset = validation_dataset.filter(lambda x: bool(x['text']))
    logger.info(f"Filtering took {time() - start_time} seconds")
    
    def filter_empty(example) -> bool:
        return bool(example)
        # return bool(example['data'])
        
    # Filter the datasets
    # start_time = time()
    # NUM_CORES_FILTER = 98 # NOTE hardcoded for now during experiments
    # train_dataset = train_dataset.filter(filter_empty, num_proc=NUM_CORES_FILTER)
    # validation_dataset = validation_dataset.filter(filter_empty, num_proc=NUM_CORES_FILTER)
    # logger.info(f"Filtering took {time() - start_time} seconds")
    
    # Shuffle train dataset as well 
    train_dataset = train_dataset.shuffle(seed=training_args.seed) # NOTE also try without shuffling since it might slow down 

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token,
    }
    logger.info(f"Using dropout with probability {model_args.dropout_prob}")

    if model_args.config_name:
        config = PIXELConfig.from_pretrained(
            model_args.config_name,
            attention_probs_dropout_prob=model_args.dropout_prob,
            hidden_dropout_prob=model_args.dropout_prob,
            **config_kwargs,
        )
    elif model_args.model_name_or_path:
        config = PIXELConfig.from_pretrained(
            model_args.model_name_or_path,
            attention_probs_dropout_prob=model_args.dropout_prob,
            hidden_dropout_prob=model_args.dropout_prob,
            **config_kwargs,
        )
    else:
        config = PIXELConfig(
            attention_probs_dropout_prob=model_args.dropout_prob,
            hidden_dropout_prob=model_args.dropout_prob,
            **config_kwargs,
        )
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # Adapt config
    config.update(
        {
            "mask_ratio": model_args.mask_ratio,
            "norm_pix_loss": model_args.norm_pix_loss,
            "architectures": [PIXELForPreTraining.__name__],
        }
    )

    # Create model
    if model_args.model_name_or_path:
        model = PIXELForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            **config_kwargs,
        )
    else:
        logger.info("Training new model from scratch")
        if training_args.bf16 and training_args.deepspeed is not None:
            dtype = torch.bfloat16
        elif training_args.fp16 and training_args.deepspeed is not None:
            dtype = torch.float16
        else:
            dtype = torch.float32
        model = PIXELForPreTraining._from_config(config, torch_dtype=dtype)

    # Load text renderer
    #FIXME: fix the hardcoded path
    text_renderer = PangoCairoTextRenderer.from_pretrained(model_args.text_renderer_name_or_path, fallback_fonts_dir="path/to/data/fallback_fonts_dd2248copy", rgb=True, **config_kwargs)
    text_renderer.max_seq_length = 511

    feature_extractor = ViTFeatureExtractor()

    # Determine a max seq length that has an integer square root
    max_seq_length = 0
    for i in range(100):
        if i * i >= text_renderer.max_seq_length:
            max_seq_length = i * i
            break

    # logger.debug(f"{max_seq_length = }")
    # logger.debug(f"{text_renderer.max_seq_length = }")

    # Adjust image size
    image_height = text_renderer.pixels_per_patch
    image_width = text_renderer.pixels_per_patch * max_seq_length
    model.config.image_size = (image_height, image_width)
    model.image_size = (image_height, image_width)
    feature_extractor.size = (image_height, image_width)

    # logger.debug(f"{model.image_size = }")

    # Reinitialize embeddings
    if model_args.model_name_or_path is None and model_args.config_name is not None:
        logger.info("Reinitializing embeddings. Warning: This should not happen when continuing pretraining from a PIXEL model.")
        model.vit.embeddings = PIXELEmbeddings(model.config).to(dtype)
        model.decoder.decoder_pos_embed = torch.nn.Parameter(
            torch.zeros((1, max_seq_length + 1, 512)), requires_grad=False
        )
        decoder_pos_embed = get_2d_sincos_pos_embed(
            model.decoder.decoder_pos_embed.shape[-1], int(max_seq_length ** 0.5), add_cls_token=True
        )
        model.decoder.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).to(dtype).unsqueeze(0))

    logger.info(model)
    logger.info("***** Final model config *****")
    logger.info(config)

    total_params = sum([p.numel() for p in model.parameters()])
    logger.info(f"Total parameters count: {total_params}")
    encoder_params = sum([p.numel() for p in model.vit.parameters()])
    logger.info(f"Encoder parameters count: {encoder_params}")
    encoder_embedding_params = sum([p.numel() for p in model.vit.embeddings.parameters()])
    logger.info(f"Encoder embeddings parameters count: {encoder_embedding_params}")
    decoder_params = sum([p.numel() for p in model.decoder.parameters()])
    logger.info(f"Decoder parameters count: {decoder_params}")

    # Get patch mask generator if span masking
    if model_args.span_masking and model_args.masking_max_span_length and model_args.masking_cumulative_span_weights:
        logger.info(
            f'Applying span masking with "max_span_length = {model_args.masking_max_span_length}" '
            f', "cumulative_span_weights = {model_args.masking_cumulative_span_weights}" '
            f' and "spacing = {model_args.masking_spacing if model_args.masking_spacing else "span"}"'
        )
        patch_mask_generator = SpanMaskingGenerator(
            max_span_length=model_args.masking_max_span_length,
            spacing=model_args.masking_spacing if model_args.masking_spacing else "span",
            cumulative_span_weights=model_args.masking_cumulative_span_weights,
        )

    if data_args.do_normalize:
        image_mean = feature_extractor.image_mean
        image_std = feature_extractor.image_std
    else:
        image_mean, image_std = (None, None)
        feature_extractor.do_normalize = data_args.do_normalize

    # Set transformations --- resize by default and optionally apply normalization
    transforms = get_transforms(
        do_resize=False,
        do_normalize=data_args.do_normalize,
        image_mean=image_mean,
        image_std=image_std,
    )

    logger.info(f"Applied transformations: {transforms}")

    pad_to_multiple_of = 64  # if "a100" in torch.cuda.get_device_name().lower() else 8
    
    def string_to_ngrams(s:str, n:int=2):
        """
        Takes a string and returns a list of character n-grams by splitting `s` on every `n` character.
        Args:
            s (str): The input string to be converted to bigrams.
            n (int): The frequency of which the input string is split. Defaults to `n`=2
        Returns:
            list: A list of character n-grams.
        """
        return [s[i:i + n] for i in range(0, len(s), n)]

    def preprocess_images(examples):
        """Preprocess a batch of images by applying transforms."""

        # logger.debug(f"{examples = }")

        data = {"pixel_values": [], "attention_mask": []}
        if model_args.span_masking:
            data.update({"patch_mask": []})

        encodings = [text_renderer(text=text) for text in examples["text"]]

        longest_sequence_length = max([encoding.num_text_patches for encoding in encodings]) + 1
        padding = pad_to_multiple_of - (longest_sequence_length % pad_to_multiple_of)
        padded_seq_length = longest_sequence_length + padding
        padded_seq_length -= 1
        w_padded = padded_seq_length * text_renderer.pixels_per_patch

        # logger.debug(f"{padded_seq_length = }")
        # logger.debug(f"{w_padded = }")

        # wandb_images = []
        # real_ws = []
        for encoding in encodings:

            # Apply transforms and pad image
            img = transforms(Image.fromarray(encoding.pixel_values))
            c, h, w = img.shape
            w_real = (encoding.num_text_patches + 1) * text_renderer.pixels_per_patch
            # logger.debug(f"Unpadded {c = }, {h = }, {w = }, {w_real = }")
            pixel_values = torch.ones((c, h, w_padded))
            pixel_values[:, :, : w_real] = img[:, :, : w_real]

            # wandb_images.append(wandb.Image(pixel_values))
            # real_ws.append(w_real)

            data["pixel_values"].append(pixel_values)
            data["attention_mask"].append(get_attention_mask(encoding.num_text_patches, seq_length=padded_seq_length))
            if model_args.span_masking:
                data["patch_mask"].append(
                    torch.tensor(
                        patch_mask_generator(
                            num_patches_total=padded_seq_length,
                            num_text_patches=encoding.num_text_patches + 1,
                            num_patches_to_mask=math.ceil(padded_seq_length * model_args.mask_ratio)
                        ))
                )

        # wandb.log({"img": wandb_images})
        return data

    if training_args.do_train:
        # Set training transforms
        train_dataset.set_transform(preprocess_images)

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            validation_dataset = validation_dataset.shuffle(seed=training_args.seed).select(
                range(data_args.max_eval_samples)
            )
        # Set the validation transforms
        validation_dataset.set_transform(preprocess_images)

    # Compute absolute learning rate
    total_train_batch_size = (
        training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = training_args.base_learning_rate * total_train_batch_size / 256

    # Initialize our trainer
    trainer = PIXELTrainerForPretraining(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        tokenizer=text_renderer,
        data_collator=collate_fn,
    )
    #logger.info(f"{trainer.use_cuda_amp = }")
    #logger.info(f"{trainer.amp_dtype = }")
    #trainer.use_cuda_amp = True
    #trainer.amp_dtype = torch.bfloat16
    # new_dataloader = DataLoader(validation_dataset, batch_size=8, collate_fn=collate_fn)
    # logger.debug(next(iter(new_dataloader)))
    # logger.debug(next(iter(trainer.get_eval_dataloader())))

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        # Also save feature extractor together with model and text renderer
        feature_extractor.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "tasks": "masked-auto-encoding",
        "dataset": "wikipedia + bookcorpus",
        "tags": ["masked-auto-encoding"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
