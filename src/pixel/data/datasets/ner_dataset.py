# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
import os.path as osp
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, TextIO, Union

import torch
from filelock import FileLock
from PIL import Image
from transformers import PreTrainedTokenizer, is_torch_available
from datasets import load_dataset, load_from_disk

from ...utils import Modality, Split, get_attention_mask
from ..rendering import PangoCairoTextRenderer, PyGameTextRenderer

logger = logging.getLogger(__name__)


@dataclass
class NERInputExample:
    """
    A single training/test example for named entity recognition

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data import Dataset

    class NERDataset(Dataset):
        """
        PyTorch dataset for named entity recognition.
        """

        features: List[Dict[str, Union[int, torch.Tensor]]]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            data_dir: str,
            hf_dataset_name: Optional[str],
            hf_dataset_config: Optional[str],
            processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizer],
            labels: List[str],
            modality: Modality,
            transforms: Optional[Callable] = None,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.TRAIN,
            **kwargs,
        ):
            is_hf_dataset = hf_dataset_name is not None

            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}".format(mode.value, processor.__class__.__name__, str(max_seq_length)),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):
                does_cache_exist = os.path.exists(cached_features_file)
                # use_cached_data = does_cache_exist and not overwrite_cache
                if not is_hf_dataset:
                    logger.info(f"Loading examples from dir: {data_dir}")
                    self.examples = read_examples_from_file(data_dir=data_dir, mode=mode)
                else:
                    logger.info(f"Loading examples from the HuggingFace dataset: {hf_dataset_name}/{hf_dataset_config}")
                    self.examples = read_examples_from_huggingface(
                        dataset_name=hf_dataset_name,
                        dataset_config=hf_dataset_config,
                        mode=mode,
                    )

                if does_cache_exist and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    convert_examples_to_features_fn = get_examples_to_features_fn(modality)
                    self.features = convert_examples_to_features_fn(
                        self.examples, labels, max_seq_length, processor, transforms, **kwargs
                    )
                    logger.info(f"Saving features into cached file {cached_features_file}")
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> Dict[str, Union[int, torch.Tensor]]:
            return self.features[i]


    class NERDatasetJIT(Dataset):
        """
        PyTorch dataset for named entity recognition.
            - Just-in-time rendered on-the-fly.
            - Only works with HF datasets saved on the disk.
            - Particularly implemented for the Naamapadam dataset.
        """

        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            data_dir: str,
            processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizer],
            labels: List[str],
            modality: Modality,
            transforms: Optional[Callable] = None,
            max_seq_length: Optional[int] = None,
            mode: Split = Split.TRAIN,
            seed: Optional[int] = None,
            num_examples: Optional[int] = None,
            **kwargs,
        ):
            self.data_dir = osp.abspath(osp.expanduser(data_dir))
            self.processor = processor
            self.modality = modality
            self.transforms = transforms
            self.max_seq_length = max_seq_length
            self.kwargs = kwargs

            # Load the data.
            self.mode = mode
            self.split = "validation" if mode.value == "dev" else mode.value
            self.data = load_from_disk(self.data_dir)
            self.data = self.data[self.split]
            if self.split == "train" and num_examples is not None:
                assert seed is not None
                self.data = self.data.shuffle(seed=seed).select(range(num_examples))
            self.labels = labels
            self._init_feature_extractor()
            self.data = self.data.filter(lambda x: len(x["tokens"]) > 0)

        def _init_feature_extractor(self):
            if self.modality == Modality.IMAGE:
                self._extract_features = create_image_feature_extractor(
                    label_list=self.labels,
                    max_seq_length=self.max_seq_length,
                    processor=self.processor,
                    transforms=self.transforms,
                )
            elif self.modality == Modality.TEXT:
                self._extract_features = create_text_feature_extractor(
                    label_list=self.labels,
                    max_seq_length=self.max_seq_length,
                    processor=self.processor,
                    transforms=self.transforms,
                    **self.kwargs,
                )

        def get_ner_example(self, i):
            hf_example = self.data[i]
            label_ints = hf_example['ner_tags']
            labels = [self.data.features['ner_tags'].feature._int2str[x] for x in label_ints]
            ner_example = NERInputExample(
                guid=f"{self.mode}-{i}",
                words=hf_example["tokens"],
                labels=labels,
            )
            return ner_example

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i) -> Dict[str, Union[int, torch.Tensor]]:
            ner_example = self.get_ner_example(i)
            features = self._extract_features(ner_example)
            return features


def read_examples_from_file(data_dir, mode: Union[Split, str], label_idx=-1) -> List[NERInputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(NERInputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[label_idx].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(NERInputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
    return examples


# this is more particularly implemented for the Naamapadam dataset.
# NERInputExample(
#   guid='train-1750',
#   words=['በዓለም', 'ገበያ', 'የካካዎ', 'ምርት', 'በሚወድቅበት', 'ጊዜ', 'የነዚህ', 'ሀገር', 'መንግሥታት', 'ከገበሬዉ', 'የካካዎ', 'ምርቱን', 'የሚሸምቱት', 'በድሮዉ', 'ዋጋ', 'ብቻ', 'ነዉ', '።'],
#   labels=['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
def read_examples_from_huggingface(
    dataset_name: str,
    dataset_config: str,
    mode: Union[Split, str],
) -> List[NERInputExample]:
    def _convert_int_to_tags(example):
        example["labels"] = [data.features["ner_tags"].feature.int2str(tag) for tag in example["ner_tags"]]
        return example
    data_split = mode.value
    data_split = 'validation' if data_split == 'dev' else data_split

    # FIXME: this is not good, but okay for now.
    load_data_from_disk = dataset_name != 'ai4bharat/naamapadam'
    if not load_data_from_disk:
        data = load_dataset(dataset_name, dataset_config, split=data_split)
    else:
        data = load_from_disk(dataset_name)
        data = data[data_split]
    data = data.map(_convert_int_to_tags)
    examples = []
    for hf_idx, example in enumerate(data):
        if len(example["tokens"]) != 0:
            this = NERInputExample(
                guid=f"{mode}-{hf_idx}",
                words=example["tokens"],
                labels=example["labels"],
            )
            examples.append(this)
    return examples



def write_predictions_to_file(writer: TextIO, test_input_reader: TextIO, preds_list: List):
    example_id = 0
    for line in test_input_reader:
        if line.startswith("-DOCSTART-") or line == "" or line == "\n":
            writer.write(line)
            if not preds_list[example_id]:
                example_id += 1
        elif preds_list[example_id]:
            output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
            writer.write(output_line)
        else:
            logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])


def get_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        # Use MasakhaNER labels
        # https://github.com/masakhane-io/masakhane-ner
        return ["O", "B-DATE", "I-DATE", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def get_examples_to_features_fn(modality: Modality):
    if modality == Modality.IMAGE:
        return convert_examples_to_image_features
    if modality == Modality.TEXT:
        return convert_examples_to_text_features
    else:
        raise ValueError("Modality not supported.")


def convert_examples_to_image_features(
    examples: List[NERInputExample],
    label_list: List[str],
    max_seq_length: int,
    processor: Union[PyGameTextRenderer, PangoCairoTextRenderer],
    transforms: Optional[Callable] = None,
    pad_token_label_id=-100,
) -> List[Dict[str, Union[int, torch.Tensor, List[int]]]]:
    """Loads a data file into a list of `Dict` containing image features"""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info(f"Writing example {ex_index} of {len(examples)}")

        encoding = processor(example.words)
        image = encoding.pixel_values
        num_patches = encoding.num_text_patches
        word_starts = encoding.word_starts

        label_ids = [pad_token_label_id] * max_seq_length
        for idx, word_start in enumerate(word_starts[:-1]):
            label_ids[word_start] = label_map[example.labels[idx]]

        pixel_values = transforms(Image.fromarray(image))
        attention_mask = get_attention_mask(num_patches, seq_length=max_seq_length)

        # sanity check lengths
        assert len(attention_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"sentence: {' '.join(example.words)}")
            logger.info(f"attention_mask: {attention_mask}")
            logger.info(f"label_ids: {label_ids}")

        features.append({"pixel_values": pixel_values, "attention_mask": attention_mask, "label_ids": label_ids})

    return features


def create_image_feature_extractor(
    label_list: List[str],
    max_seq_length: int,
    processor: Union[PyGameTextRenderer, PangoCairoTextRenderer],
    transforms: Optional[Callable] = None,
    pad_token_label_id: int = -100,
):
    label_map = {label: i for i, label in enumerate(label_list)}
    def extract_image_features(example):
        encoding = processor(example.words)
        image = encoding.pixel_values
        num_patches = encoding.num_text_patches
        word_starts = encoding.word_starts

        label_ids = [pad_token_label_id] * max_seq_length
        for idx, word_start in enumerate(word_starts[:-1]):
            label_ids[word_start] = label_map[example.labels[idx]]

        pixel_values = transforms(Image.fromarray(image))
        attention_mask = get_attention_mask(num_patches, seq_length=max_seq_length)

        # sanity check lengths
        assert len(attention_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        return {
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
        }

    return extract_image_features


def convert_examples_to_text_features(
    examples: List[NERInputExample],
    label_list: List[str],
    max_seq_length: int,
    processor: PreTrainedTokenizer,
    # pass the ones below as kwargs to the dataset __init__
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    **kwargs,
) -> List[Dict[str, Union[int, torch.Tensor, List[int]]]]:
    """Loads a data file into a list of `InputFeatures`
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = processor.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = processor.num_special_tokens_to_add() + (1 if sep_token_extra else 0)
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            token_type_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = processor.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
            token_type_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if "token_type_ids" not in processor.model_input_names:
            token_type_ids = None

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"sentence: {' '.join(example.words)}")
            logger.info(f"tokens: {' '.join(tokens)}")
            logger.info(f"input_ids: {input_ids}")
            logger.info(f"attention_mask: {attention_mask}")
            logger.info(f"token_type_ids: {token_type_ids}")
            logger.info(f"label_ids: {label_ids}")

        features.append(
            {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "label_ids": label_ids,
            }
        )
    return features


def create_text_feature_extractor(
    label_list: List[str],
    max_seq_length: int,
    processor: PreTrainedTokenizer,
    # pass the ones below as kwargs to the dataset __init__
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    **kwargs,
):
    """Loads a data file into a list of `InputFeatures`
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    label_map = {label: i for i, label in enumerate(label_list)}
    def extract_text_features(example):
        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = processor.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = processor.num_special_tokens_to_add() + (1 if sep_token_extra else 0)
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            token_type_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = processor.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
            token_type_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if "token_type_ids" not in processor.model_input_names:
            token_type_ids = None

        return {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "label_ids": label_ids,
        }

    return extract_text_features
