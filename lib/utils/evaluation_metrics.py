from lib.pycocoevalcap.bleu.bleu import Bleu
from lib.pycocoevalcap.rouge import Rouge
from lib.pycocoevalcap.meteor import Meteor
import evaluate
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")


def NLU_metrics(labels, predictions):
    # ground_truth_captions = [caption.split() for caption in labels]
    ground_truth_captions = labels
    # generated_captions = [caption.split() for caption in predictions]
    generated_captions = predictions
    bleu0 = corpus_bleu(ground_truth_captions, generated_captions)
    bleu1 = corpus_bleu(ground_truth_captions,
                        generated_captions, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(ground_truth_captions,
                        generated_captions, weights=(0, 1, 0, 0))
    bleu3 = corpus_bleu(ground_truth_captions,
                        generated_captions, weights=(0, 0, 1, 0))
    bleu4 = corpus_bleu(ground_truth_captions,
                        generated_captions, weights=(0, 0, 0, 1))
    rougeTF = rouge.compute(
        predictions=predictions, references=labels)
    # bleuTF = bleu.compute(predictions=predictions, references=labels)
    return {
        "bleu0": bleu0,
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "rouge1": rougeTF['rouge1'],
        "rouge2": rougeTF['rouge2'],
        "rougeL": rougeTF['rougeL'],
        "rougeLsum": rougeTF['rougeL'],
        # "bleuTF": bleuTF['bleu'],
    }


def classification_metrics(labels, predictions):
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


def compute_scores(gts, res):
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
