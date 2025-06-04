#!/usr/bin/env python3
"""
Script to download and save specified language splits from the `Davlan/sib200` dataset to disk.
Usage:
    python save_sib200_splits.py /path/to/output_root
"""
import os
import os.path as osp
import argparse
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="Save language-specific splits from Davlan/sib200 to disk"
    )
    parser.add_argument(
        '--save-dir',
        help="Root directory where language folders will be created and saved"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # List of language configurations to process
    languages = [
        'arz_Arab',
        'ben_Beng',
        'bod_Tibt',
        'deu_Latn',
        'ell_Grek',
        'eng_Latn',
        'est_Latn',
        'fin_Latn',
        'fra_Latn',
        'heb_Hebr',
        'hin_Deva',
        'hye_Armn',
        'jpn_Jpan',
        'kir_Cyrl',
        'kor_Hang',
        'rus_Cyrl',
        'tam_Taml',
        'tel_Telu',
        'tur_Latn',
        'uig_Arab',
        'ukr_Cyrl',
        'urd_Arab',
        'uzn_Latn',
        'zho_Hans',
    ]

    for lang in languages:
        print(f"Loading split for language: {lang}...")
        # Each call returns a DatasetDict (e.g., with 'train', 'validation', etc.)
        ds = load_dataset('Davlan/sib200', lang)

        # Prepare output directory for this language
        out_dir = osp.abspath(osp.join(osp.expanduser(args.save_dir), lang))
        os.makedirs(out_dir, exist_ok=True)

        print(f"Saving dataset for {lang} to {out_dir}")
        # Save the entire DatasetDict in Hugging Face Arrow format
        ds.save_to_disk(out_dir)

    print("All specified language splits have been saved.")


if __name__ == '__main__':
    main()

