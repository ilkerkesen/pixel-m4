from __future__ import absolute_import, division, unicode_literals

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from pixel.analysis.splitclassifier import SplitClassifier


def get_split_sim(S, i, j, n=701):
    """
        Return the cosine-similarity block between split i and split j.
        Splits are zero-indexed, each of size n rows.
    """
    start_i, end_i = i*n, (i+1)*n
    start_j, end_j = j*n, (j+1)*n
    return S[start_i:end_i, start_j:end_j]


class RetrievalExperiment:
    def __init__(self, params, batcher, layer):
        self.seed = "1111"
        self.params = params
        self.batcher = batcher
        self.layer = layer
        self.loadData()
        self.extractEmbeddings(batcher)

    def loadData(self):
        self.data = list() 
        self.y = list()
        self.indices = list()
        languages = self.params['languages']
        for lang in languages:
            data = load_dataset('Davlan/sib200', lang, split='train')
            for x in data:
                self.data.append(x['text'])
                self.y.append(lang)
                self.indices.append(x['index_id'])

    def extractEmbeddings(self, batcher):
        params = self.params
        bsize = self.params['batch_size']
        embed = {}
        layer_embs = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [],
                      8: [], 9: [], 10: [], 11: [], 12: []}
        for i in tqdm(range(0, len(self.data), bsize)):
            batch = self.data[i:i+bsize]
            embs = batcher(params, batch)
            for k, v in embs.items():
                layer_embs[k].append(embs[k])
        for layer in range(0, 13):
            embed[layer] = np.vstack(layer_embs[layer])
        self.embed = embed

    def run(self):
        results = {k: None for k in range(0, 13)}
        for layer in tqdm(range(0, 13)):
            X = self.embed[layer]
            mean_vec = X.mean(axis=0, keepdims=True)
            X_centered = X - mean_vec
            norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
            X_normalized = X_centered / norms
            scores = X_normalized @ X_normalized.T
            results[layer] = scores
        return results, self.y, self.indices
