import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Universal NER datasets to the desired format.")
    parser.add_argument(
        "--input_path",
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        required=True,
        help="Path of the raw dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        required=True,
        help="The output path for the processed format.",
    )
    return parser.parse_args()


def preprocess_file(input_file, output_file, split):
    with open(input_file, 'r') as f:
        raw_lines = [line.strip() for line in f.readlines()]

    lines = list() 
    for raw_line in raw_lines:
        if raw_line.startswith("# "):
            continue

        if raw_line.strip() == '':
            lines.append('')
        else:
            tok, label = raw_line.split('\t')[1:3]
            lines.append(f"{tok} {label}")

    with open(output_file, 'w') as f:
        f.write("\n".join(lines) + "\n")


def find_input_file(path, split):
    file_list = os.listdir(path)
    filtered = [fname for fname in file_list if fname.endswith(f'-{split}.iob2')]
    assert len(filtered) == 1
    return os.path.join(path, filtered[0])

if __name__ == "__main__":
    args = parse_args()
    print(f"input: {args.input_path}")
    print(f"output: {args.output_path}")

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    splits = ['train', 'dev', 'test']
    for split in splits:
        input_file = find_input_file(args.input_path, split)
        output_file = os.path.join(args.output_path, f'{split}.txt')
        preprocess_file(input_file, output_file, split)

