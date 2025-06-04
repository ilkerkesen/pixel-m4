import argparse
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#E69F00', '#0072B2']
# plt.style.use('bmh')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'

MODEL_NAME_DICT = {
    "pixel-m4": "PIXEL-M4",
    "pixel-bigrams": "PIXEL-BIGRAMS",
}

LANG_CODE_DICT = {
    "eng_Latn": "ENG",
    "hin_Deva": "HIN",
    "ukr_Cyrl": "UKR",
    "zho_Hans": "ZHO",
}

def recall_at_k(sim: np.ndarray, k: int = 1) -> float:
    # sim: (n, n) score matrix for one split (rows = queries, cols = targets)
    topk = np.argsort(-sim, axis=1)[:, :k]
    n = sim.shape[0]
    hits = sum(1 for i in range(n) if i in topk[i])
    return hits / n


def compute_all_recalls(model: str, path: str, k: int = 1, n_per_lang: int = 701):
    layers = list(range(0, 13))
    example_data = np.load(osp.join(path, f"{model}-layer=12.npz"))
    labels = example_data["labels"]
    lang_codes = [LANG_CODE_DICT[labels[i*n_per_lang]] for i in range(4)]
    pairs = list(combinations(range(4), 2))  # [(0,1),(0,2)...(2,3)]
    rec = np.zeros((len(layers), len(pairs)))
    
    for li, layer in enumerate(layers):
        data = np.load(f"{path}/{model}-layer={layer}.npz")
        S = data['embeddings']  # shape (2804,2804)
        
        for pi, (i, j) in enumerate(pairs):
            # row-block for lang i, col-block for lang j
            start_i, end_i = i*n_per_lang, (i+1)*n_per_lang
            start_j, end_j = j*n_per_lang, (j+1)*n_per_lang
            sim_ij = S[start_i:end_i, start_j:end_j]
            rec[li, pi] = recall_at_k(sim_ij, k=k)
    
    pair_labels = [f"{lang_codes[i]}–{lang_codes[j]}" for i,j in pairs]
    return layers, rec, pair_labels


def plot_recalls(layers, rec, pair_labels, k: int):
    plt.figure(figsize=(8,5))
    for pi, label in enumerate(pair_labels):
        # print(np.array(rec[:,pi]).max())
        plt.plot(layers, rec[:,pi], marker='o', linestyle='-', label=label)
    plt.xticks(layers)
    plt.xlabel("Layer", fontsize=20)
    plt.ylabel(f"Recall@{k}", fontsize=20)
    # plt.title(f"{model_name}")
    plt.legend(loc="best", ncol=2, fontsize=16)
    plt.grid(True, linestyle='--', alpha=1.0)
    plt.axhline(0.056, linestyle='--')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-id", default="Team-PIXEL/pixel-m4", type=str)
    parser.add_argument("--results-dir", default="outputs/analysis/retrieval", type=str)
    parser.add_argument("--K", default=5, type=int, help="K value for recall@K metric.")
    args = parser.parse_args()
    model = args.model_id.split('/')[-1]
    layers, rec, labels = compute_all_recalls(model, args.results_dir, k=args.K)
    plot_recalls(layers, rec, labels, args.K)

