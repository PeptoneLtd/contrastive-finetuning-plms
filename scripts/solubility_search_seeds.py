import gc
import copy
import math
import random

import numpy as np
import pandas as pd
from bisect import bisect
from sklearn import metrics
from sklearn.linear_model import SGDClassifier

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses


# the code will require GPU
# -------------------------

# choose which ST based model you wish to run
# ---------------------------------------------------------------------------------------------------------------------------
#     'facebook/esm2_t12_35M_UR50D' (small);
#     'facebook/esm2_t30_150M_UR50D' (large);
#     'facebook/esm2_t33_650M_UR50D' (largest) - currently will require an 80GB GPU;
# ---------------------------------------------------------------------------------------------------------------------------
esm_model_tag = (  # or to get the small model choose replace this with 'facebook/esm2_t12_35M_UR50D'
    "facebook/esm2_t30_150M_UR50D"
)

# set the data path - data files can be found in the repo's 'data/solubility' folder
# ---------------------------------------------------------------------------------

DATA_PATH = "data/solubility/psi_data.json"
VAL_DATA_PATH = "data/solubility/NESG_testset.json"

if esm_model_tag == "facebook/esm2_t30_150M_UR50D":
    seed_list = np.random.choice(
        10000, 5
    )  # generate random seeds, one for each partition in the 5 fold CV
    BATCH_SIZE = 6
else:
    seed_list = np.random.choice(
        10000, 5
    )  # generate random seeds, one for each partition in the 5 fold CV
    BATCH_SIZE = 16

sol_col = "solublility|0=Insoluble|1=Soluble"
category_col = sol_col
seq_col = "fasta"
val_sol_col = "solubility"


def set_seed(seed: int):
    """
    Set the seeds in python, numpy and torch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sequence_pairs_generation(
    dataframe: pd.DataFrame, number_of_sentences_per_sequence: int = 5
) -> list:
    sequences = dataframe[seq_col].values
    labels = dataframe[category_col].values

    numClassesList = np.unique(labels)
    bucket_indexes = [np.where(labels == i)[0] for i in numClassesList]

    sentences = []

    for i in range(number_of_sentences_per_sequence):
        for sequence_index in range(len(sequences)):
            currentSequence = sequences[sequence_index]
            label = labels[sequence_index]

            concord_sequence_index = np.random.choice(
                bucket_indexes[np.where(numClassesList == label)[0][0]]
            )
            posSequence = sequences[concord_sequence_index]
            sentences.append(InputExample(texts=[currentSequence, posSequence], label=1.0))

            discord_sequence_index = np.where(labels != label)[0]
            negSequence = sequences[np.random.choice(discord_sequence_index)]
            sentences.append(InputExample(texts=[currentSequence, negSequence], label=0.0))

    return sentences


class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()


def main():
    df_psi = pd.read_json(f"{DATA_PATH}")
    df_val = pd.read_json(f"{VAL_DATA_PATH}")

    st_model_tag = esm_model_tag
    _st_tag_save = st_model_tag.split("/")[1]
    num_training = 512
    num_epochs = 3
    empty_metric_dict = {
        "AUC_prc": [],
        "APS": [],
        "acc": [],
        "precision": [],
        "recall": [],
        "auc_roc": [],
        "seed": [],
    }

    # iterate over folds
    # ------------------

    res_fold = {
        i: {"fold": copy.deepcopy(empty_metric_dict), "val_fold": copy.deepcopy(empty_metric_dict)}
        for i in df_psi["Partition"].unique()
    }

    for partition in range(5):  # df_psi['Partition'].unique():

        # setting the right optimal seed
        _seed = seed_list[partition]

        df_sub = df_psi[df_psi["Partition"] != partition]

        fold_test_iterator = list(
            zip(
                df_psi[df_psi["Partition"] == partition][seq_col],
                df_psi[df_psi["Partition"] == partition][category_col],
            )
        )
        test_seqs, y_test = list(zip(*fold_test_iterator))

        fold_train_iterator = list(
            zip(
                df_psi[df_psi["Partition"] != partition][seq_col],
                df_psi[df_psi["Partition"] != partition][category_col],
            )
        )
        train_seqs, y_train = list(zip(*fold_train_iterator))

        set_seed(_seed)

        train_df_sample = pd.concat(
            [
                df_sub[df_sub[category_col] == k].sample(num_training)
                for k in df_sub[category_col].unique()
            ]
        )
        sentences = sequence_pairs_generation(train_df_sample)

        st_model = SentenceTransformer(st_model_tag)
        train_dataloader = DataLoader(sentences, shuffle=True, batch_size=BATCH_SIZE)
        train_loss = losses.CosineSimilarityLoss(st_model)

        warmup_steps = math.ceil(
            len(train_dataloader) * num_epochs * 0.1
        )  # 10% of train data for warm-up

        # train the model
        device = torch.device("cuda:0")
        # put the model and data on GPUs
        st_model.to(device)

        try:
            with ClearCache():
                st_model.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=num_epochs,
                    warmup_steps=warmup_steps,
                    show_progress_bar=True,
                    use_amp=True,
                )
        except:
            continue

        X_train = []
        X_train = st_model.encode(train_seqs, show_progress_bar=True)

        X_test = []
        X_test = st_model.encode(test_seqs, show_progress_bar=True)

        X_train = np.array(X_train)
        X_test = np.array(X_test)

        # validation set
        X_val = []
        val_seqs = list(df_val[seq_col])
        y_val = list(df_val[val_sol_col])
        X_val = st_model.encode(val_seqs, show_progress_bar=True)

        clf = SGDClassifier(
            loss="log_loss",
            max_iter=500000,
            tol=0.0005,
            penalty="elasticnet",
            alpha=0.0001,
            random_state=72,
        )
        clf.fit(X_train, y_train)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.decision_function(X_test))
        print(f"on the split - {_seed}")
        print("AUC_prc", metrics.auc(fpr, tpr))
        print("APS", metrics.average_precision_score(y_test, clf.decision_function(X_test)))
        print("--------------------------------")
        print()

        print(clf.score(X_test, y_test))
        print(
            metrics.average_precision_score(
                y_test, clf.predict_proba(X_test)[:, 1], average="micro"
            )
        )
        print(metrics.recall_score(y_test, clf.predict(X_test), average="micro"))
        print(metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

        res_fold[partition]["fold"]["AUC_prc"].append(metrics.auc(fpr, tpr))
        res_fold[partition]["fold"]["APS"].append(
            metrics.average_precision_score(y_test, clf.decision_function(X_test))
        )
        res_fold[partition]["fold"]["acc"].append(clf.score(X_test, y_test))
        res_fold[partition]["fold"]["precision"].append(
            metrics.average_precision_score(
                y_test, clf.predict_proba(X_test)[:, 1], average="weighted"
            )
        )
        res_fold[partition]["fold"]["recall"].append(
            metrics.recall_score(y_test, clf.predict(X_test), average="weighted")
        )
        res_fold[partition]["fold"]["auc_roc"].append(
            metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        )
        res_fold[partition]["fold"]["seed"].append(_seed)

        print("-------------------------------------------------------")

        # check the validation set
        fpr_val, tpr_val, thresholds_val = metrics.roc_curve(y_val, clf.decision_function(X_val))
        print("Validation set results: ")
        print("--------------------------------")
        print(f"on the split - {_seed}")
        print("AUC_prc", metrics.auc(fpr_val, tpr_val))
        print("APS", metrics.average_precision_score(y_val, clf.decision_function(X_val)))
        print("--------------------------------")
        print()

        print(clf.score(X_val, y_val))
        print(
            metrics.average_precision_score(y_val, clf.predict_proba(X_val)[:, 1], average="micro")
        )
        print(metrics.recall_score(y_val, clf.predict(X_val), average="micro"))
        print(metrics.roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))

        res_fold[partition]["val_fold"]["AUC_prc"].append(metrics.auc(fpr_val, tpr_val))
        res_fold[partition]["val_fold"]["APS"].append(
            metrics.average_precision_score(y_val, clf.decision_function(X_val))
        )
        res_fold[partition]["val_fold"]["acc"].append(clf.score(X_val, y_val))
        res_fold[partition]["val_fold"]["precision"].append(
            metrics.average_precision_score(
                y_val, clf.predict_proba(X_val)[:, 1], average="weighted"
            )
        )
        res_fold[partition]["val_fold"]["recall"].append(
            metrics.recall_score(y_val, clf.predict(X_val), average="weighted")
        )
        res_fold[partition]["val_fold"]["auc_roc"].append(
            metrics.roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        )
        res_fold[partition]["val_fold"]["seed"].append(_seed)

        del st_model
        gc.collect()
        torch.cuda.empty_cache()
        print("emptied cache...")

    print()
    print("5 fold CV results: ")

    for key in empty_metric_dict.keys():
        _aux = [res_fold[partition]["fold"][key][0] for partition in range(5)]
        print(f"{key}, mean: {np.mean(_aux):.3f}, std: {np.std(_aux):.3f}")

    print()
    print("Validation set - 5 fold CV results: ")

    for key in empty_metric_dict.keys():
        _aux = [res_fold[partition]["val_fold"][key][0] for partition in range(5)]
        print(f"{key}, mean: {np.mean(_aux):.3f}, std: {np.std(_aux):.3f}")


if __name__ == "__main__":
    main()
