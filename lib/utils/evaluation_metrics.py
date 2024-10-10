from lib.pycocoevalcap.bleu.bleu import Bleu
from lib.pycocoevalcap.rouge import Rouge
from lib.pycocoevalcap.meteor import Meteor
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def classificationMetrics(labels, predictions):
    labels = np.array(labels).reshape(-1, 1)
    predictions = np.array(predictions).reshape(-1, 1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(
        labels, predictions, average='macro', zero_division=1)
    recall = recall_score(labels, predictions,
                          average='macro', zero_division=1)
    f1 = f1_score(labels, predictions, average='macro', zero_division=1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def computeScores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        # (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res
