from __future__ import absolute_import, division, unicode_literals

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer


class TSNEExperiment:
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
        languages = self.params['languages']
        for lang in languages:
            data = load_dataset('Davlan/sib200', lang, split='train')
            for x in data:
                self.data.append(x['text'])
                self.y.append(lang)

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

    def run(self, params, batcher, n_components=2, perplexity=50, learning_rate=500, n_iter=1000):
        embed = self.embed
        results = {k: None for k in range(0, 13)}
        for layer in tqdm(range(0, 13)):
            X = embed[layer]
            normalizer = Normalizer(norm='l2')
            X = normalizer.fit_transform(X)
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=42
            )
            X_tsne = tsne.fit_transform(X)
            results[layer] = X_tsne
        return results, self.y

