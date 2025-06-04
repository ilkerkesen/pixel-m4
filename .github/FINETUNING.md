## Finetuning PIXEL-M4

Here we provide instructions for finetuning PIXEL-M4. You can use these to reproduce our results or train your own PIXEL-based model on a different dataset. If your dataset is currently not supported and you don't know how to get started (because the renderer is missing a feature, a classification head is missing, etc.) feel free to open an issue about it. 

Requirements:
- An environment set up as described in our main [README.md](https://github.com/ilkerkesen/pixel-m4/blob/main/README.md)

### Downloading data and fallback fonts
<details>
  <summary><i>Show Instructions</i></summary>
&nbsp;

#### Fallback fonts
We provide a script to download fallback fonts for the `PangoCairoTextRenderer`. It is not necessary to use fallback fonts because our default `GoNotoCurrent.ttf` font already covers most languages/scripts. The renderer will log warnings if it encounters unknown glyphs. If that happens, you should definitely consider downloading the fallback fonts and passing the folder to the renderer via `--fallback_fonts_dir` so everything is rendered correctly:
  
```bash
python scripts/data/download_fallback_fonts.py <output_dir>
```

#### Data 

Use the provided scripts to download the data.

```bash
# Create a folder in which we keep the data
mkdir -p data
  
# UD data for parsing and POS tagging
wget -qO- https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4758/ud-treebanks-v2.10.tgz | tar xvz -C data
  
# SIB-200 language splits
python scripts/data/save_sib_to_disk.py --save-dir ./data/sib-200

# Indic NER data
python scripts/data/save_naamapadam.py --save-dir ./data/naamapadam

# KLUE/NER data
python scripts/data/save_klue_ner_to_disk.py ---save-dir ./data/klue/ner

```

For Universal NER, clone the repos and run the preprocessing script,

```bash
mkdir data/uner-raw

git clone https://github.com/UniversalNER/UNER_English-EWT ./data/uner-raw/eng
git clone https://github.com/UniversalNER/UNER_Serbian-SET ./data/uner-raw/srp
git clone https://github.com/UniversalNER/UNER_Chinese-GSD ./data/uner-raw/zho

python scripts/data/preprocess_uner.py --input_path ./data/uner-raw/eng/ --output_path ./data/uner/eng
python scripts/data/preprocess_uner.py --input_path ./data/uner-raw/srp/ --output_path ./data/uner/srp
python scripts/data/preprocess_uner.py --input_path ./data/uner-raw/zho/ --output_path ./data/uner/zho
```


</details>

### Training
 
Finetuning PIXEL-M4 and other PIXEL-based language models works almost in the same way as for any other model in the [transformers library](https://github.com/huggingface/transformers). There are, however, a few important differences:
- Instead of a tokenizer, we use a renderer for PIXEL. We use the bigrams renderer proposed in [this work](https://aclanthology.org/2023.emnlp-main.628/).
- The maximum sequence length in PIXEL-based models needs to have an integer square root, e.g. `256 = 16 * 16` or `400 = 20 * 20`. This is because of how the image is divided into patches

Note: All examples here use grayscale rendering. When using the pangocairo-based rendering backends, you can activate RGB rendering via `--render_rgb`. However, this will make rendering a little slower, so we recommend to only use it when you know you're working with color inputs.

Here are some examples for how to finetune PIXEL:

#### SIB-200
<details>
  <summary><i>Show Code Snippet</i></summary>
&nbsp;

```bash
export WANDB_DISABLED=true
export FONTCONFIG_PATH=/etc/fonts  # some systems needs this.

MODEL="Team-PIXEL/pixel-m4"
LANG="hin_Deva"

# fixed hyperparameters for SIB-200 experiments.
FALLBACK_FONTS_DIR="./fallback_fonts"  # let's say this is where we downloaded the fonts to
SEQ_LEN=256
BSZ=32
GRAD_ACCUM=1
NUM_STEPS=15000
FP16_OR_BF16_="bf16"

# sweeped hyperparameters.
SEED=0
LR=1e-5

# set this based on your compute.
NUM_WORKERS=8

RUN_NAME="${LANG}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"
OUTPUT_DIR="./logs/$(basename ${MODEL})/sib-200/${LANG}/${LR}--${SEED}"
DATA_DIR="./data/sib-200"
python ./scripts/training/run_sib_bigrams.py \
    --model_name_or_path=${MODEL} \
    --remove_unused_columns=False \
    --data_dir=${DATA_DIR} \
    --language ${LANG} \
    --do_train --do_eval --do_predict \
    --dropout_prob=0.1 \
    --max_seq_length=${SEQ_LEN} \
    --max_steps=${NUM_STEPS} \
    --early_stopping \
    --early_stopping_patience=20 \
    --per_device_train_batch_size=${BSZ} \
    --gradient_accumulation_steps=${GRAD_ACCUM} \
    --learning_rate=${LR} \
    --run_name=${RUN_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --overwrite_output_dir \
    --overwrite_cache \
    --logging_strategy=epoch \
    --logging_steps=1 \
    --evaluation_strategy=epoch \
    --save_strategy=epoch \
    --save_total_limit=2 \
    --report_to=none \
    --log_predictions \
    --load_best_model_at_end=True \
    --metric_for_best_model="eval_f1" \
    --bf16 \
    --half_precision_backend=cuda_amp \
    --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
    --seed=${SEED} \
    --dataloader_num_workers=${NUM_WORKERS} \
    --rendering_backend="bigrams"
```
</details>


#### Dependency Parsing
<details>
  <summary><i>Show Code Snippet</i></summary>
&nbsp;

```bash
export WANDB_DISABLED=true
export FONTCONFIG_PATH=/etc/fonts  # some systems need this declaration.

# model and treebank
MODEL="Team-PIXEL/pixel-m4"
TREEBANK="UD_Hindi-HDTB"

# fixed hyperparameters for dependency parsing experiments.
FALLBACK_FONTS_DIR="./fallback_fonts"  # let's say this is where we downloaded the fonts to
SEQ_LEN=256
BSZ=64
GRAD_ACCUM=1
NUM_STEPS=15000
FP16_OR_BF16_="bf16"

# sweeped hyperparameters.
SEED=0
LR=1e-5

# set this based on your compute.
NUM_WORKERS=8

RUN_NAME="${TREEBANK}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"
OUTPUT_DIR="./logs/$(basename ${MODEL})/udp/${TREEBANK}/${LR}--${SEED}"
DATA_DIR="./data/ud-treebanks-v2.10/$TREEBANK"
python ./scripts/training/run_ud_bigrams.py \
    --model_name_or_path=${MODEL} \
    --remove_unused_columns=False \
    --data_dir=${DATA_DIR} \
    --do_train --do_eval --do_predict \
    --dropout_prob=0.1 \
    --max_seq_length=${SEQ_LEN} \
    --max_steps=${NUM_STEPS} \
    --early_stopping \
    --early_stopping_patience=5 \
    --per_device_train_batch_size=${BSZ} \
    --per_device_eval_batch_size=${BSZ} \
    --gradient_accumulation_steps=${GRAD_ACCUM} \
    --learning_rate=${LR} \
    --warmup_steps=100 \
    --run_name=${RUN_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --overwrite_output_dir \
    --overwrite_cache \
    --logging_strategy=steps \
    --logging_steps=100 \
    --evaluation_strategy=steps \
    --eval_steps=500 \
    --save_strategy=steps \
    --save_steps=500 \
    --save_total_limit=2 \
    --report_to=none \
    --log_predictions \
    --load_best_model_at_end=True \
    --metric_for_best_model="eval_las" \
    --bf16 \
    --half_precision_backend=cuda_amp \
    --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
    --seed=${SEED} \
    --dataloader_num_workers=${NUM_WORKERS} \
    --rendering_backend="pangocairo"
```

</details>

#### NER
<details>
  <summary><i>Show Code Snippet</i></summary>
&nbsp;

```bash
export WANDB_DISABLED=true
export FONTCONFIG_PATH=/etc/fonts  # some systems need this declaration.

MODEL="Team-PIXEL/pixel-m4"
LANG="hi"

# predefined.
FALLBACK_FONTS_DIR="/gpfs/projects/ehpc137/fallback_fonts"  # let's say this is where we downloaded the fonts to
SEQ_LEN=196
BSZ=64
GRAD_ACCUM=1
NUM_STEPS=50000
LEARNING_RATES=(1e-5 3e-5 5e-5)
FP16_OR_BF16_="bf16"

# sweeped hyperparameters.
SEED=0
LR=1e-5

# set this based on your compute.
NUM_WORKERS=8

RUN_NAME="${LANG}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"
OUTPUT_DIR="./logs/$(basename ${MODEL})/naamapadam/${LANG}/${LR}--${SEED}"
DATA_DIR="./data/naamapadam/${LANG}"
python ./scripts/training/run_ner_bigrams.py \
    --model_name_or_path=${MODEL} \
    --data_dir=${DATA_DIR} \
    --do_train --do_eval --do_predict \
    --jit_rendering=True \
    --remove_unused_columns=False \
    --dropout_prob=0.1 \
    --max_seq_length=${SEQ_LEN} \
    --max_steps=${NUM_STEPS} \
    --early_stopping \
    --early_stopping_patience=5 \
    --per_device_train_batch_size=${BSZ} \
    --per_device_eval_batch_size=${BSZ} \
    --gradient_accumulation_steps=${GRAD_ACCUM} \
    --learning_rate=${LR} \
    --warmup_steps=100 \
    --run_name=${RUN_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --overwrite_output_dir \
    --overwrite_cache \
    --logging_strategy=steps \
    --logging_steps=100 \
    --evaluation_strategy=steps \
    --eval_steps=500 \
    --save_strategy=steps \
    --save_steps=500 \
    --save_total_limit=2 \
    --report_to=none \
    --log_predictions \
    --load_best_model_at_end=True \
    --metric_for_best_model="eval_f1" \
    --bf16 \
    --half_precision_backend=cuda_amp \
    --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
    --seed=${SEED} \
    --dataloader_num_workers=${NUM_WORKERS} \
    --rendering_backend="pangocairo"
 ```

Note that, the above script targets Naamapadam -- Indic NER benchmark.
It also works for KLUE/NER benchmark, yet you need to adjust `OUTPUT_DIR` and `DATA_DIR`.
To run experiments on Universal NER language splits, please disable JIT rendering (`--jit_rendering=False`).
</details>
