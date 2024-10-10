import os
import random
import re
from PIL.Image import Image
from transformers import BertTokenizer, AutoTokenizer
import pandas as pd
import numpy as np
import torch

from torch import nn


def shiftTokensRight(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError(
            "Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError(
            "Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def setSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def cleanFinding(finding):
    if type(finding) != str:
        return ""
    finding = finding.lower()
    # remove sentence before two new lines (remove sentence description view position)
    finding = re.sub(r".*\n*\s\n", "", finding)
    # remove new lines to space
    finding = re.sub(r"\n", " ", finding)
    # remove multiple spaces to single space
    finding = re.sub(r"\s+", " ", finding)
    finding = finding.strip()
    return finding


def preprocessData():
    captions = pd.read_csv('mimic-cxr/txt/data/mimic_cxr_sectioned.csv')
    splits = pd.read_csv('files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv')
    splits['study_id'] = 's' + splits['study_id'].astype(str)
    chexpert = pd.read_csv(
        'files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv')
    chexpert['study_id'] = 's' + chexpert['study_id'].astype(str)
    metadata = pd.read_csv(
        'files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv')
    metadata['study_id'] = 's' + metadata['study_id'].astype(str)
    data = pd.merge(splits, captions, left_on='study_id', right_on='study')
    data = data.drop(columns=['study'])
    data = pd.merge(data, metadata, left_on=['dicom_id', 'study_id', 'subject_id'], right_on=[
        'dicom_id', 'study_id', 'subject_id'])
    data = pd.merge(data, chexpert, left_on=['study_id', 'subject_id'], right_on=[
                    'study_id', 'subject_id'])
    dataNew = data.loc[data['ViewPosition'].isin(["PA", "AP"])]
    dataNew['findings'] = dataNew['findings'].apply(cleanFinding)
    dataNew = dataNew[dataNew['findings'].notnull()]
    dataNew = dataNew[dataNew['findings'] != '']
    dataNew = dataNew.reset_index(drop=True)
    dataNew.to_csv("data.csv", index=False)


def buildInputsWithSpecialTokens(self, token_ids_0):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


def trainTokenizer(checkpointPath, allData, batchSize, vocabSize, datasetName):
    def batch_iterator():
        for i in range(0, len(allData), batchSize):
            yield allData[i:i+batchSize]
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print(f"Tokenizer is Fast: {tokenizer.is_fast}")
    new_tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(), vocab_size=vocabSize)
    # new_tokenizer.model_max_length = 128
    new_tokenizer.pad_token = tokenizer.bos_token
    new_tokenizer.padding_side = "right"
    new_tokenizer.save_pretrained(f"{checkpointPath}/tokenizer/{datasetName}/")

# Tagging loss function
# copy from https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


def downIuDataset():
    idFile = '1N4m_eEXG1M6qRYBwIrrl3M6P4_d67m_X'
    FILENAME = 'iu-dataset.zip'
    url = f'''wget --load-cookies /tmp/cookies.txt "https://drive.usercontent.google.com/download?confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.usercontent.google.com/download?id={
        idFile}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={idFile}" -O {FILENAME} && rm -rf /tmp/cookies.txt'''
    os.system(url)
    os.system(
        'mkdir iu-dataset && mv iu-dataset.zip iu-dataset/iu-dataset.zip && cd iu-dataset/ && unzip iu-dataset.zip && rm iu-dataset.zip')
    print('Downloaded')
