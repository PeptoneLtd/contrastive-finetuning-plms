import numpy as np
import os
import sys
import pandas as pd

from collections import Counter
import itertools
import json
import requests
import operator
import torch

from transformers import AutoTokenizer, EsmModel, AutoModel
from itertools import islice
import time

from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score, f1_score, precision_recall_fscore_support, average_precision_score, \
    precision_recall_curve, roc_curve
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import ElasticNet
import sklearn.metrics as metrics

import time
import torch

import torch
import numpy as np
import random
import pandas as pd
from collections import OrderedDict, Counter
import itertools
from itertools import groupby

import os
import sys
# from Bio import SeqIO
import copy
from typing import Dict, List, Tuple, TypeVar
import string
import scipy

from sklearn.linear_model import ElasticNet
import sklearn.metrics

import transformers
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, models, datasets, \
    evaluation, util
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from bisect import bisect
import math

from huggingface_hub import snapshot_download

'''
This script is a brief demo, that shows how to find ESM-based disorder predictors that use SentenceTransformers.
The idea is to concatenate the residue level representations with sequence level representations and use
these combined embeddings with a prediction head (a simple ElasticNet e.g.) to predict z-scores.

The residue level representations are extracted from the esm model "esm2_t33_650M_UR50D".

The sequence level representations are coming from a BiEncoder, which is based on a fine-tuned tiny esm model,
esm2_t6_8M_UR50D.

Here we rely on huggingface and use the esm ecosystem through huggingface.

The routine extracts residue level embeddings on the fly and does not use cached objects, hence it is not
necessarily suitable for computationally intense random search.
'''

# prepare the data using the same train and test set that appear in the final version of the ADOPT paper
# ------------------------------------------------------------------------------------------------------
# the two datasets can be found in the repo's 'data/disorder' folder

path_chezod_1159_raw = 'data/disorder/chezod_1159.json'
path_chezod_117_raw = 'data/disorder/117_dataset_raw.json'
df_1159 = pd.read_json(path_chezod_1159_raw)
df_117 = pd.read_json(path_chezod_117_raw)

# just in case, remove any overlaps from the train set (1159)
overlaps = list(set(list(df_1159["brmid"])) & set(list(df_117["brmid"])))
df_cleared = df_1159[~df_1159["brmid"].isin(overlaps)]

# bucket sequences in terms of sequence level disorder profiles - more details on this will be provided separately
# ----------------------------------------------------------
df_cleared['mean-z-score'] = df_cleared['z-score'].apply(lambda x: np.mean(np.array(x)[np.where(np.array(x) != 999)]))
df_cleared['std-z-score'] = df_cleared['z-score'].apply(lambda x: np.std(np.array(x)[np.where(np.array(x) != 999)]))


def mean_std_bucketing(df_row):
    """
    The numbers used for cutoffs here come from 50% quantiles of the
    relevant quantities, i.e. mean and std
    """

    if df_row['std-z-score'] < 4.046:
        if df_row['mean-z-score'] < 12.55:
            label = 0
        else:
            label = 2
    else:
        if df_row['mean-z-score'] < 10.51:
            label = 1
        else:
            label = 3
    return label


def z_score_bucketing(df_row, std_cutoff=4.046, mean_cutoffs=None):
    """
    The numbers used for cutoffs here come from 50% quantiles of the
    relevant quantities, i.e. mean and std
    """

    if mean_cutoffs is None:
        mean_cutoffs = [12.55, 10.51]
    if df_row['std-z-score'] < std_cutoff:
        if df_row['mean-z-score'] < mean_cutoffs[0]:
            label = 0
        else:
            label = 2
    else:
        if df_row['mean-z-score'] < mean_cutoffs[1]:
            label = 1
        else:
            label = 3
    return label


df_cleared['bucket-z-score'] = df_cleared.apply(mean_std_bucketing, axis=1)

# bucketing the test set, i.e. df_117 is not relevant as we do not use this at this stage
mean_z_scores_buckets = [np.quantile(np.array(df_cleared['mean-z-score']), q) for q in [0.25, 0.5, 0.75]]
df_117['mean-z-score'] = df_117['zscore'].apply(lambda x: np.mean(np.array(x)[np.where(np.array(x) != 999)]))
df_117['bucket-z-score'] = df_117['mean-z-score'].apply(lambda x: bisect(mean_z_scores_buckets, x))


def set_seed(seed: int):
    """
    Set the seeds in python, numpy and torch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# SetFit approach
# ---------------
def sequence_pairs_generation(sequences: List[str], labels: List[int], pairs: List) -> List:
    # initialize two empty lists to hold the (sequence, sequence) pairs and
    # labels to indicate if a pair is positive or negative

    numClassesList = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in numClassesList]

    for idxA in range(len(sequences)):
        currentSequence = sequences[idxA]
        label = labels[idxA]
        idxB = np.random.choice(idx[np.where(numClassesList == label)[0][0]])
        posSequence = sequences[idxB]
        # prepare a positive pair and update the sentences and labels
        # lists, respectively
        pairs.append(InputExample(texts=[currentSequence, posSequence], label=1.0))

        negIdx = np.where(labels != label)[0]
        negSequence = sequences[np.random.choice(negIdx)]
        # prepare a negative pair of sequences and update our lists
        pairs.append(InputExample(texts=[currentSequence, negSequence], label=0.0))

        # return a 2-tuple of our sequence pairs and labels
    return (pairs)


def sequence_triplets_generation(sequences: List[str], labels: List[int], triplets: List) -> List:
    # initialize two empty lists to hold the (sequence, sequence) pairs and
    # labels to indicate if a pair is positive or negative

    numClassesList = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in numClassesList]

    for idxA in range(len(sequences)):
        currentSequence = sequences[idxA]
        label = labels[idxA]
        idxB = np.random.choice(idx[np.where(numClassesList == label)[0][0]])
        posSequence = sequences[idxB]
        # prepare a positive pair and update the sentences and labels
        # lists, respectively
        # pairs.append(InputExample(texts=[currentSequence, posSequence], label=1.0))

        negIdx = np.where(labels != label)[0]
        negSequence = sequences[np.random.choice(negIdx)]
        # prepare a negative pair of sequences and update our lists
        triplets.append(InputExample(texts=[currentSequence, posSequence, negSequence]))

        # return a 2-tuple of our sequence pairs and labels
    return (triplets)


seq_col = 'sequence'
category_col = 'bucket-z-score'

esm_model_name = 'facebook/esm2_t33_650M_UR50D'
esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
esm_model_8bit = AutoModel.from_pretrained(esm_model_name, device_map="auto", load_in_8bit=True)


def get_embeddings(indexes: List[str], df, st_repr: List, esm_model_8bit, esm_tokenizer, z_col='z-score',
                   drop_missing=True, add_st_repr=True) -> Tuple:
    """
    Prepares the input using residue and also sequence level representations by concatenating them

    More detail later. Clean up will be required as well.
    """
    status_update = False
    zeds = []
    exes = []

    print('extracting embeddings...')
    for k in range(len(indexes)):
        seq = df[df['brmid'] == indexes[k]].sequence.item()

        # retrieve residue level embeddings from esm using huggingface
        inputs = esm_tokenizer(seq, return_tensors="pt", padding=True)
        outputs = esm_model_8bit(**inputs)
        repr_esm = outputs.last_hidden_state[0, 1:1 + len(seq), ...].clone().detach().cpu().numpy()

        z_s = np.array(df[df['brmid'] == indexes[k]][z_col].to_numpy()[0])

        if drop_missing:
            idxs = np.where(z_s != 999)[0]
        else:
            idxs = np.arange(len(z_s))

        for i in idxs:
            zeds.append(z_s[i])
            if add_st_repr:
                exes.append(np.concatenate([repr_esm[i], st_repr[seq]]))
            else:
                exes.append(repr_esm[i])
        if k / len(indexes) > 0.5 and not status_update:
            print('half way through, ...hang in there')
            status_update = True

    return np.array(exes), np.array(zeds)


# example seeds
seed_list = [983115, 759167, 821769, 573297, 331205]  # np.random.choice(10000, 1)

res = {}
# SetFit params
st_model = 'facebook/esm2_t6_8M_UR50D'
num_training = 256
num_itr = 5

run_times = []

for _idx, _seed in enumerate(seed_list):

    start = time.time()
    # set seed
    set_seed(_seed)
    # equal samples per class training
    train_df_sample = pd.concat(
        [df_cleared[df_cleared[category_col] == k].sample(num_training) for k in df_cleared[category_col].unique()])
    x_train = train_df_sample[seq_col].values.tolist()
    y_train = train_df_sample[category_col].values.tolist()

    train_examples = []
    for x in range(num_itr):
        train_examples = sequence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)

    model = SentenceTransformer(st_model)  # load it from cache

    # S-BERT adaptation
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=24)
    train_loss = losses.CosineSimilarityLoss(model)

    num_epochs = 4
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up

    # train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps,
              show_progress_bar=True)

    # prepare the data
    ex_train, zed_train = {}, {}
    ex_test, zed_test = {}, {}

    # first the sequence level representations extracted from the fine-tuned BiEncoder
    train_st_repr = dict(zip(df_cleared['sequence'], model.encode(list(df_cleared['sequence']))))
    test_st_repr = dict(zip(df_117['sequence'], model.encode(list(df_117['sequence']))))

    for model_type in ['esm-st']:
        ex_train[model_type], zed_train[model_type] = get_embeddings(list(df_cleared['brmid']),
                                                                     df_cleared,
                                                                     train_st_repr,
                                                                     esm_model_8bit,
                                                                     esm_tokenizer,
                                                                     z_col='z-score',
                                                                     drop_missing=True,
                                                                     add_st_repr=True)

        ex_test[model_type], zed_test[model_type] = get_embeddings(list(df_117['brmid']),
                                                                   df_117,
                                                                   test_st_repr,
                                                                   esm_model_8bit,
                                                                   esm_tokenizer,
                                                                   z_col='zscore',
                                                                   drop_missing=True,
                                                                   add_st_repr=True)

    X_train = copy.deepcopy(ex_train['esm-st'])
    y_train = copy.deepcopy(zed_train['esm-st'])
    X_test = copy.deepcopy(ex_test['esm-st'])
    y_test = copy.deepcopy(zed_test['esm-st'])

    # prediction head to predict the z-scores
    regr = ElasticNet(alpha=0.0001, max_iter=500000, tol=0.0005, precompute=True)
    regr.fit(X_train, y_train)

    # metrics
    predict_skl = regr.predict(X_test)
    r2_score_skl = sklearn.metrics.r2_score(y_test, predict_skl)
    print('r2 with skl:', r2_score_skl)
    print(
        f"MAE/std with skl - {sklearn.metrics.mean_absolute_error(y_test, predict_skl):.4f} / {np.std(abs(y_test - predict_skl)):.4f}")
    # correlation
    print(f"Number: {_idx}; Seed: {_seed} ElasticNet - Correlation between the predicted and the ground \
        truth on the test set:  {scipy.stats.spearmanr(y_test, predict_skl).correlation:.4f}")
    print('---------------------------------')

    y_train_bin = np.where(y_train < 8., 1., 0, )
    y_test_bin = np.where(y_test < 8., 1., 0, )

    clf = SGDClassifier(loss='log_loss', max_iter=500000, tol=0.0005, penalty='elasticnet', alpha=0.0001,
                        random_state=72)
    clf.fit(X_train, y_train_bin)

    fpr, tpr, thresholds = metrics.roc_curve(y_test_bin, clf.decision_function(X_test))

    print('AUC_prc: ', metrics.auc(fpr, tpr))
    print('APS: ', metrics.average_precision_score(y_test_bin, clf.decision_function(X_test)))

    end = time.time()
    run_times.append(end - start)

    res[_seed] = {
        'seed': _seed,
        'r2': r2_score_skl,
        'MAE': sklearn.metrics.mean_absolute_error(y_test, predict_skl),
        'corr': scipy.stats.spearmanr(y_test, predict_skl).correlation,
        'AUC_prc': metrics.auc(fpr, tpr),
        'APS': metrics.average_precision_score(y_test_bin, clf.decision_function(X_test)), }

print()
print('ST-avg results; avg of top 5')
print('----------------------------')
for _metric in ['r2', 'MAE', 'corr', 'AUC_prc', 'APS']:
    _aux = [res[_key][_metric] for _key in res.keys()]
    print(f"{_metric}, mean: {np.mean(_aux):.3f}, std: {np.std(_aux):.3f}")

print()
