import logging
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from lib.config import Config
from lib.datasets.dataset import collate_fn
from lib.experiment import Experiment
from lib.utils.evaluation_metrics import classification_metrics, compute_scores
from lib.utils.utils import AsymmetricLoss, setSeed
from pycocoevalcap.bleu.bleu import Bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

Bleu_scorer = None


def init_scorer():
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)


seed = 42
warnings.filterwarnings("ignore")


class Runner:
    def __init__(self,
                 config: Config,
                 experiment: Experiment,
                 epochs=50,
                 trainBatchSize=4,
                 testBatchSize=4,
                 valOnEpochs=1,
                 testOnEpochs=1,
                 saveBest=True,
                 deviceIds=None):

        self.config = config
        self.max_length = self.config["datasets"]["seq_length"]
        self.exp = experiment
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.deviceIds = deviceIds
        self.logger = logging.getLogger(__name__)
        self.epochs = epochs
        self.valOnEpochs = valOnEpochs
        self.testOnEpochs = testOnEpochs
        self.saveBest = saveBest
        self.bestBleu1 = 0
        self.bestRougeL = 0
        self.trainBatchSize = trainBatchSize
        self.testBatchSize = testBatchSize
        self.accum_iter = 4
        self.lossTag = AsymmetricLoss(gamma_neg=7, gamma_pos=0, clip=0.05)
        setSeed(seed)

    def train(self):
        max_epochs = self.epochs
        self.exp.train_start_callback()
        trainLoader, listTag = self.getTrainDataloader()
        model = self.config.getModel(listTag=listTag).to(self.device)
        self.exp.model_params_callback(model)
        optimizer = self.config.getOptimizer(model)
        scheduler = self.config.getLrScheduler(optimizer)
        loss_fn = self.config.getLossFunction(
            reduction='mean', label_smoothing=0.1)
        bestLoss = np.inf
        for epoch in range(max_epochs):
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.train()
            train_loss = 0
            pbar = tqdm(trainLoader)
            for batch_idx, (img, caption, mask, label) in enumerate(pbar):
                img, caption, mask, label = map(lambda x: x.to(self.device),
                                                (img, caption, mask, label))
                if len(self.deviceIds) > 1:
                    predictClass, tokens = model.data_parallel(
                        pixel_values=img, decoder_input_ids=caption, device_ids=self.deviceIds, tagInput=mask)
                else:
                    predictClass, tokens = model(
                        pixel_values=img, decoder_input_ids=caption, tagInput=label)
                lossTag = self.lossTag(predictClass, label)
                outputs = F.log_softmax(tokens, dim=-1)
                lossCap = loss_fn(outputs, caption, mask)
                loss = lossCap + lossTag
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.set_postfix(epoch=epoch, Total_Loss=loss.item(), loss_cls=lossTag.item(),
                                 loss_cap=lossCap.item(), lr=scheduler.get_last_lr()[0])
                train_loss += loss.item()
                self.exp.iter_end_callback(epoch, max_epochs, batch_idx,
                                           len(trainLoader), lossTag.item(), lossCap.item(), loss.item(), scheduler.get_last_lr()[0])
            train_loss = train_loss/len(trainLoader)
            self.exp.epoch_end_callback(
                epoch, max_epochs, train_loss, model, optimizer, scheduler, bestLoss)

            if (epoch + 1) % self.valOnEpochs == 0:
                lossTotalVal = self.val(model, loss_fn, epoch)
                bestLoss = min(bestLoss, lossTotalVal)
                self.exp.delete_model(lossTotalVal == bestLoss, epoch)
            if (epoch + 1) % self.testOnEpochs == 0:
                self.test(model, epoch)
                self.test(model, epoch, guide=True)

    def val(self, model=None, loss_fn=None, epoch=None, on_val=True):
        if on_val:
            dataLoader, listTag = self.getValDataloader()
        else:
            dataLoader, listTag = self.getTestDataloader()
        if model is None and loss_fn is None and epoch is None:
            model = self.config.getModel(
                listTag=listTag).to(self.device)
            lastCheckpointEpoch = self.exp.get_last_checkpoint_epoch()
            modelCheckpoint = self.exp.get_epoch_model(lastCheckpointEpoch)
            model.load_state_dict(torch.load(
                modelCheckpoint), map_location=self.device)
            loss_fn = self.config.getLossFunction(
                reduction='mean', label_smoothing=0.1)
        lossClassification = 0
        lossCaption = 0
        lossTotal = 0
        pbar = tqdm(dataLoader)
        model.eval()
        with torch.no_grad():
            for batch_idx, (img, caption, mask, label) in enumerate(pbar):
                img, caption, mask, label = map(lambda x: x.to(self.device),
                                                (img, caption, mask, label))
                if len(self.deviceIds) > 1:
                    predictClass, tokens = model.data_parallel(
                        pixel_values=img, decoder_input_ids=caption, device_ids=self.deviceIds, tagInput=label)
                else:
                    predictClass, tokens = model(
                        pixel_values=img, decoder_input_ids=caption, tagInput=label)
                loss1 = self.lossTag(predictClass, label)
                lossClassification += loss1.item()
                outputs = F.log_softmax(tokens, dim=-1)
                lossCap = loss_fn(outputs, caption, mask)
                loss = lossCap + loss1
                lossCaption += lossCap.item()
                lossTotal += loss.item()
                pbar.set_postfix(Total_Loss=loss.item(), loss_cls=loss1.item(),
                                 loss_cap=lossCap.item())
        lossClassification /= len(dataLoader)
        lossCaption /= len(dataLoader)
        lossTotal /= len(dataLoader)
        self.exp.eval_end_callback(
            lossCaption, lossClassification, lossTotal, epoch, on_val)

        return lossTotal

    def test(self, model=None, epoch=None, guide=False):
        dataLoader, listTag = self.getTestDataloader()
        if model is None and epoch is None:
            model = self.config.getModel(
                listTag=listTag).to(self.device)
            modelCheckpoint = self.exp.get_best_model()
            model.load_state_dict(modelCheckpoint)
        pbar = tqdm(dataLoader)
        model.eval()
        labels = []
        predictClasses = []
        captionsGT = []
        predictCaptions = []
        with torch.no_grad():
            for batch_idx, (img, caption, mask, label) in enumerate(pbar):
                img, caption, label, mask = map(lambda x: x.to(self.device),
                                                (img, caption, label, mask))
                predictClass, captions = model.generate(
                    img, max_length=self.max_length, tags=label if guide else None)
                predictCaptions.extend(captions)
                captionLabels = self.config["datasets"]["tokenizer"].decode_batch(
                    caption[:, 1:].cpu().numpy())
                captionsGT.extend(captionLabels)
                predictClasses.extend(
                    predictClass.detach().cpu().numpy().astype(int).tolist())
                labels.extend(
                    label.detach().cpu().numpy().astype(int).tolist())
        metricsNLU = compute_scores({i: [gt] for i, gt in enumerate(captionsGT)},
                                    {i: [re] for i, re in enumerate(predictCaptions)})
        metricsClassification = classification_metrics(
            labels, predictClasses)
        self.exp.end_test_callback(
            metricsNLU, metricsClassification, epoch)
        if self.saveBest:
            if metricsNLU["BLEU_1"] > self.bestBleu1:
                self.bestBleu1 = metricsNLU["BLEU_1"]
                self.exp.save_best_model(
                    model, epoch, metricsNLU, metricsClassification, name="bleu1")
            if metricsNLU["ROUGE_L"] > self.bestRougeL:
                self.bestRougeL = metricsNLU["ROUGE_L"]
                self.exp.save_best_model(
                    model, epoch, metricsNLU, metricsClassification, name="rougeL")

    def getTrainDataloader(self):
        trainDataset = self.config.getDataset(mode="train")
        return DataLoader(dataset=trainDataset,
                          batch_size=self.trainBatchSize,
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=collate_fn), trainDataset.tags

    def getValDataloader(self):
        valDataset = self.config.getDataset(mode="validate")
        return DataLoader(dataset=valDataset,
                          batch_size=self.testBatchSize,
                          shuffle=False,
                          num_workers=8,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=collate_fn), valDataset.tags

    def getTestDataloader(self):
        testDataset = self.config.getDataset(mode="test")
        return DataLoader(dataset=testDataset,
                          batch_size=self.testBatchSize,
                          shuffle=False,
                          num_workers=8,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=collate_fn), testDataset.tags
