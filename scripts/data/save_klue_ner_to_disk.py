import argparse
import os
from datasets import load_dataset, get_dataset_config_names


def parse_args():
    parser = argparse.ArgumentParser(description="Save KLUE/NER dataset to disk.")
    parser.add_argument(
        "--save-dir",
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        required=True,
        help="Directory where results will be saved."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if set."
    )
    return parser.parse_args()


def create_preprocess_fn(data):
    ner_tags = data["train"].features["ner_tags"]
    def _convert_int_to_tags(example):
        example["labels"] = [ner_tags.feature.int2str(tag) for tag in example["ner_tags"]]
        return example
    return _convert_int_to_tags


if __name__ == "__main__":
    args = parse_args()
    print("Save Directory:", args.save_dir)
    print("Overwrite:", args.overwrite)

    data = load_dataset("klue/klue", "ner")
    fn = create_preprocess_fn(data)
    data = data.map(fn)
    data.save_to_disk(os.path.abspath(args.save_dir))
    print('done.')
