"""
Script used to preprocess a Wikipedia .txt dataset file
Processes the dataset book-by-book and splits it into sequences of up to a target length for easy on-the-fly-rendering.
"""
# NOTE inspiration if change: split is defined outside __main__ is not enough
# https://discuss.huggingface.co/t/map-multiprocessing-issue/4085/25
import argparse
import logging
import os
import os.path as osp
import sys
import json
from tqdm import tqdm
import math
import unicodedata

import datasets
from datasets import Dataset, load_from_disk
from datasets import load_dataset

import gc   

logger = logging.getLogger(__name__)

#FIXME: fix this hardcoded path
hashmap_path = "path/to/data/unicode-hash-map/results.jsonl"

data = []
with open(hashmap_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))
width_lookup = {k: v for d in data for k, v in d.items()} # returns width in pixels

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

def string_to_ngrams_sep_wspace(s:str, n:int=2) -> list:
    """
    Takes a string and returns a list of character n-grams by splitting `s` on every `n` character,
    with whitespace included as separate elements.
    Args:
        s (str): The input string to be converted to bigrams.
        n (int): The frequency of which the input string is split. Defaults to `n`=2
    Returns:
        list: A list of character n-grams.
    TODO
        [x] speed test => `while` loop is faster (~50%) than *nested* list comprehension 
    """
    bigrams = []
    i = 0
    while i < len(s):
        if s[i] == ' ':
            bigrams.append(s[i])
            i += 1
        else:
            bigram = s[i:i + n]
            if ' ' in bigram:
                bigrams.append(s[i])
                i += 1
            else:
                bigrams.append(bigram)
                i += n
    return bigrams

def string_to_ngrams_sliding_window(s:str, n:int=2, w:int=1):
    """
    Takes a string and returns a list of character n-grams by splitting `s` on every `n` character.
    Args:
        s (str): The input string to be converted to bigrams.
        n (int): The frequency of which the input string is split. Defaults to `n`=2
        w (int): How far the window should slide across at each element. 
            (for bigrams (`n=2`), `w`=2 is no overlap, while `w`=1 results in single character overlap)
    Returns:
        list: A list of character n-grams.
    """
    return [s[i:i + n] for i in range(0, len(s), w)]    
    
# class PangoCairoWidthEstimator:
#     def __init__(self, text_renderer: PangoCairoTextRenderer):
#         self.text_renderer = text_renderer

#     def __call__(self, text: str):
#         text = text.replace("\n", " ")
#         return self.text_renderer(string_to_ngrams_sliding_window(text)).num_text_patches

def lookup_width_estimator(b:str, PPB:int=16) -> int:
    # NOTE change to have `PPB` be a function of the renderer config 
    # NOTE could potentially speed up by also making the width lookup a call to str.translate()
    # ... instead of a `dict` method
    if b == ' ':
        return 1
    else:
        return math.ceil(sum([width_lookup[x] for x in b]) / PPB)

# Identify all Unicode characters in the Other category e.g. 'Cn' (Not assigned) and 'Cs' (Surrogate)
chars_to_remove1 = {i: ' ' for i in range(0x0, 0x10FFFF+1) if unicodedata.category(chr(i)) in ('Cn', 'Cs', 'Co')}
# Create a translation table from the dictionary
removal_table = str.maketrans(chars_to_remove1)

#FIXME: fix this hardcoded path
with open("path/to/unrenderable_chars.json", "r") as f:
    unrenderable_chars = json.load(f)
    
unrenderable_chars_to_wspace = {str(k).encode("utf-8").decode("utf-8"): " " for k in unrenderable_chars.values()}
removal_table2 = str.maketrans(unrenderable_chars_to_wspace)

def preprocess_text(s:str, r:dict=removal_table) -> str:
    s = s.encode("utf-8").decode("utf-8") # Zero faith. Make sure that the string is in UTF-8
    # s = p.sub('', s)  # Cleaned text
    s = s.translate(removal_table) # Remove unwanted Unicode characters 
    s = s.translate(removal_table2) # Remove unrenderable Unicode characters 
    # s = s.replace("\n", " ")
    s = ' '.join(s.split()) # split() also removes NBSP chars (\xa0), \t, \n, etc
    return s

width_estimator = string_to_ngrams_sep_wspace

def split(examples):
    outputs = {
        # "title": [],
        "text": [],
    }

    for ex_text in examples["text"]:
    # for book_idx, (ex_title, ex_text) in enumerate(zip(examples["title"], examples["text"])):
        # ex_text = ex["text"]
        doc = [t for t in ex_text.split("\n") if t.strip()] # and t.strip() != ex_title.strip()]
        width = 0
        block = ""

        for line in doc:
            line = preprocess_text(line)

            # Block is empty
            if len(block) == 0:
                block = line
                width = sum(lookup_width_estimator(b) for b in width_estimator(line))
                # width = len(width_estimator(line, n=2)) # NOTE change 

                # If already long enough on its own, add it to data
                if width >= 529:
                    # outputs["title"].append(ex_title)
                    outputs["text"].append(block)
                    block = ""

            # Block is not empty
            else:
                # Estimate width when adding new line to block
                new_width = sum(lookup_width_estimator(b) for b in width_estimator(block + " " + line))
                # new_width = len(width_estimator(block + " " + line, n=1)) # NOTE change 

                # New line still fits; update block and width
                if new_width < 529:
                    block += f" {line}"
                    width = new_width
                # New line does not fit, add existing block to data if it is long enough, then reset block and width
                else:
                    if width >= 23:
                        # outputs["title"].append(ex_title)
                        outputs["text"].append(block)
                    block = line
                    width = sum(lookup_width_estimator(b) for b in width_estimator(line))
                    # width = len(width_estimator(line, n=1)) # NOTE change 

        # If block not empty and longer than one row, append to data
        if width >= 23:
            # outputs["title"].append(ex_title)
            outputs["text"].append(block)

    return outputs

def main(args: argparse.Namespace):
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    logging.disable(logging.WARNING)

    # width_estimator = string_to_ngrams_sep_wspace

    # Save test img
    # encoding = text_renderer("I like cats 🤗")
    # img = Image.fromarray(encoding.pixel_values)
    # img.save("test_bookcorpus.png")

    print(args.__dict__)

    # Create output folder if it does not exist yet
    os.makedirs(args.output_dir, exist_ok=True)
    arrow_cache_path ="arrow_cache96_cleaned"
    os.makedirs(os.path.join(args.output_dir, arrow_cache_path), exist_ok=True)
    
    for lang in ["en", "hi", "uk", "zh",]:
        output_dir = osp.abspath(osp.expanduser(args.output_dir))
        cache_dir = osp.abspath(osp.expanduser(args.cache_dir))
        dataset = load_dataset("mc4", lang, split="train", streaming=False, cache_dir=cache_dir, keep_in_memory=False)
        dataset_split = dataset.map(split, batched=True, batch_size=1, remove_columns=dataset.column_names, num_proc=96, 
                                    keep_in_memory=False, load_from_cache_file=False,
                                    cache_file_name=os.path.join(output_dir, arrow_cache_path,  f"preprocessed_mc4_bigrams_529_{lang}.arrow"))
        logger.info(f"Finished preprocessing C4, size: original = 364868892, split = {len(dataset_split)}")

        dataset_split.save_to_disk(os.path.join(args.output_dir, f"preprocessed_mc4_bigrams_529_{lang}"), num_shards=1024)
        
        # Free up memory
        del dataset_split
        del dataset
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory on disk",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Cache dir for downloading the data."
    )
    parser.add_argument(
        "--target_seq_length",
        type=int,
        default=529,
        help="Sequence length for rendering",
    )
    parsed_args = parser.parse_args()

    main(parsed_args)
