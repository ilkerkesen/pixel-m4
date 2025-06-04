## Analyzing PIXEL-M4

This file provides documentation about reproducing the PIXEL-M4 analyses reported in [our work](https://arxiv.org/abs/2505.21265).
We performed t-SNE and cross-lingual retrieval probing analyses using SIB-200 benchmark,
so a having working environment will be satisfactory to complete these studies.
For word-level probing, cloning the LINSPECTOR repository is necessary,

```bash
# on root repo dir
git clone https://github.com/UKPLab/linspector.git ./data/linspector
```

### Word-Level Probing on LINSPECTOR
Just run the following command for the desired language,

```bash
# optional: see possible command line options/arguments
python ./scripts/analysis/run_linspector.py -h

# first: run the experiment
for lang in arabic armenian modern-greek macedonian russian; do
    python ./scripts/analysis/run_linspector.py --lang $lang --model_id "Team-PIXEL/pixel-m4" 
    python ./scripts/analysis/run_linspector.py --lang $lang --model_id "Team-PIXEL/pixel-base-bigrams" 
done

# second: visualize the results
python ./scripts/analysis/plot_linspector.py --results-dir ./outputs/analysis/linspector
```

Note that, `plot_linspector.py` scripts only produces the main LINSPECTOR figure for the Arabic, Armenian, Greek, Macedonian languages
on the tasks Case, POS, SameFeat and TagCount.

### t-SNE Visualizations
```bash
# optional: see possible command line options/arguments
python ./scripts/analysis/run_tsne.py -h

# first: run the experiment
python ./scripts/analysis/run_tsne.py --model_id "Team-PIXEL/pixel-m4"
python ./scripts/analysis/run_tsne.py --model_id "Team-PIXEL/pixel-base-bigrams"

# second: visualize the results
python ./script/analysis/plot_tsne.py --results-dir ./outputs/analysis/tsne
```

### Cross-Lingual Retrieval
```
# optional: see possible command line options/arguments
python ./scripts/analysis/run_retrieval.py -h

# first: run the experiment
python ./scripts/analysis/run_retrieval.py --model_id "Team-PIXEL/pixel-m4"

# second: visualize the results
python ./script/analysis/plot_retrieval.py --results-dir ./outputs/analysis/retrieval
```
