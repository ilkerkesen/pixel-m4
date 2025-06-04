from __future__ import absolute_import, division, unicode_literals

import logging
import numpy as np
from pixel.analysis.classifier import MLP
import sklearn
assert(sklearn.__version__ >= "0.18.0"), \
    "need to update sklearn to version >= 0.18.0"
from sklearn.linear_model import LogisticRegression

def get_classif_name(classifier_config, usepytorch):
    if not usepytorch:
        modelname = 'sklearn-LogReg'
    else:
        nhid = classifier_config['nhid']
        optim = 'adam' if 'optim' not in classifier_config else classifier_config['optim']
        bs = 64 if 'batch_size' not in classifier_config else classifier_config['batch_size']
        modelname = 'pytorch-MLP-nhid%s-%s-bs%s' % (nhid, optim, bs)
    return modelname


class SplitClassifier(object):
    """
    (train, valid, test) split classifier.
    """
    def __init__(self, X, y, config):
        self.X = X
        self.y = y
        self.nclasses = config['nclasses']
        self.featdim = self.X['train'].shape[1]
        self.seed = config['seed']
        self.usepytorch = config['usepytorch']
        self.classifier_config = config['classifier']
        self.cudaEfficient = False if 'cudaEfficient' not in config else \
            config['cudaEfficient']
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)
        self.noreg = False if 'noreg' not in config else config['noreg']
        self.config = config

    def run(self):
        logging.info('Training {0} with standard validation..'
                     .format(self.modelname))
        regs = [10**t for t in range(-5, -1)] if self.usepytorch else \
               [2**t for t in range(-2, 4, 1)]
        if self.noreg:
            regs = [1e-9 if self.usepytorch else 1e9]
        scores = []
        for reg in regs:
            if self.usepytorch:
                clf = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=self.nclasses, l2reg=reg,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)

                clf.fit(self.X['train'], self.y['train'],
                        validation_data=(self.X['valid'], self.y['valid']))
            else:
                clf = LogisticRegression(C=reg, random_state=self.seed)
                clf.fit(self.X['train'], self.y['train'])
            scores.append(round(100*clf.score(self.X['valid'],
                                self.y['valid']), 2))
        logging.info([('reg:'+str(regs[idx]), scores[idx])
                      for idx in range(len(scores))])
        optreg = regs[np.argmax(scores)]
        devaccuracy = np.max(scores)
        logging.info('Validation : best param found is reg = {0} with score \
            {1}'.format(optreg, devaccuracy))
        clf = LogisticRegression(C=optreg, random_state=self.seed)
        logging.info('Evaluating...')
        if self.usepytorch:
            clf = MLP(self.classifier_config, inputdim=self.featdim,
                      nclasses=self.nclasses, l2reg=optreg,
                      seed=self.seed, cudaEfficient=self.cudaEfficient)

            clf.fit(self.X['train'], self.y['train'],
                    validation_data=(self.X['valid'], self.y['valid']))
        else:
            clf = LogisticRegression(C=optreg, random_state=self.seed)
            clf.fit(self.X['train'], self.y['train'])

        testaccuracy = clf.score(self.X['test'], self.y['test'])
        testaccuracy = round(100*testaccuracy, 2)
        y_hat = clf.predict(self.X['test']).squeeze(-1).astype(int)
        assert len(self.y['test']) == len(y_hat)
        return devaccuracy, testaccuracy, y_hat
