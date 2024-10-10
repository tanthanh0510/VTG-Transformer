import os
import re
import logging

import torch
from torch.utils.tensorboard import SummaryWriter


class Experiment:
    def __init__(self,
                 expName: str,
                 modelCheckpointInterval: int = 100,
                 mode: str = 'train',
                 expsBasedir: str = 'experiments',
                 tensorboardDir: str = 'tensorboard',
                 pretrain: str = 'pretrain') -> None:
        self.name = expName
        self.pretrain = pretrain
        self.expDirpath = os.path.join(expsBasedir, expName)
        self.modelsDirpath = os.path.join(self.expDirpath, 'models')
        self.bestModelsDirpath = os.path.join(
            self.expDirpath, 'best_models')
        self.resultsDirpath = os.path.join(self.expDirpath, 'results')
        self.logPath = os.path.join(self.expDirpath, f'log_{mode}.txt')
        self.tensorboard_writer = SummaryWriter(
            os.path.join(tensorboardDir, expName))
        self.modelCheckpointInterval = modelCheckpointInterval
        self.setupExpDir()
        self.setupLogging()

    def setupExpDir(self) -> None:
        if not os.path.exists(self.expDirpath):
            os.makedirs(self.expDirpath)
            os.makedirs(self.modelsDirpath)
            os.makedirs(self.resultsDirpath)
            os.makedirs(self.bestModelsDirpath)
        if not os.path.exists(self.pretrain):
            os.makedirs(self.pretrain)

    def setupLogging(self) -> None:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        fileHandler = logging.FileHandler(self.logPath)
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(logging.INFO)
        streamHandler.setFormatter(formatter)
        logging.basicConfig(level=logging.DEBUG, handlers=[
                            fileHandler, streamHandler])
        self.logger = logging.getLogger(__name__)

    def getLastCheckpointEpoch(self) -> int:
        pattern = re.compile('model_(\\d+).pt')
        lastEpoch = -1
        for ckptFile in os.listdir(self.modelsDirpath):
            result = pattern.match(ckptFile)
            if result is not None:
                epoch = int(result.groups()[0])
                if epoch > lastEpoch:
                    lastEpoch = epoch
        return lastEpoch

    def getBestCheckpointEpoch(self, name: str = 'bleu1') -> int:
        return os.path.join(self.bestModelsDirpath, f"best_{name}_model.pt")

    def getCheckpointPath(self, epoch: int):
        return os.path.join(self.modelsDirpath, 'model_{:04d}.pt'.format(epoch))

    def getEpochModel(self, epoch: int):
        return torch.load(self.getCheckpointPath(epoch))['model']

    def getBestModel(self, name: str = 'bleu1'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(os.path.join(
            self.bestModelsDirpath, f"best_{name}_model.pt"), map_location=device)
        self.logger.info(
            f"Best model loaded from {os.path.join(self.bestModelsDirpath, f'best_{name}_model.pt')}")
        self.logger.info(f"Metrics NLU: {checkpoint['metricsNLU']}")
        self.logger.info(
            f"Metrics Classification: {checkpoint['metricsClassification']}")
        return checkpoint['model']

    def loadLastTrainState(self,
                           model: torch.nn.Module,
                           optimizer: torch.optim,
                           scheduler: torch.optim.lr_scheduler,
                           best_score: dict) -> tuple:
        epoch = self.getLastCheckpointEpoch()
        train_state_path = self.getCheckpointPath(epoch)
        train_state = torch.load(train_state_path)
        model.load_state_dict(train_state['model'])
        optimizer.load_state_dict(train_state['optimizer'])
        scheduler.load_state_dict(train_state['scheduler'])
        if train_state['best_score']:
            best_score = train_state['best_score']
        return epoch, model, optimizer, scheduler, best_score

    def saveBestModel(self,
                      model: torch.nn.Module,
                      metricsNLU: dict,
                      metricsClassification: dict,
                      name: str
                      ) -> None:
        torch.save({
            'model': model.state_dict(),
            'metricsNLU': metricsNLU,
            'metricsClassification': metricsClassification
        }, os.path.join(self.bestModelsDirpath, f"best_{name}_model.pt"))

    def saveTrainState(self,
                       epoch: int,
                       model: torch.nn.Module,
                       optimizer: torch.optim,
                       scheduler: torch.optim.lr_scheduler,
                       best_score: dict) -> None:
        train_state_path = self.getCheckpointPath(epoch)
        torch.save(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_score': best_score,
            }, train_state_path)

    def iterEndCallback(self,
                        epoch: int,
                        max_epochs: int,
                        iter_nb: int,
                        max_iter: int,
                        loss_caption: float,
                        loss_classification: float,
                        loss: float,
                        lr: float) -> None:
        line = 'Epoch [{}/{}] - Iter [{}/{}] - Total_Loss: {:.5f} - loss_caption: {:.5f} - loss_classification: {:.5} - Lr: {:.5f}'.format(
            epoch, max_epochs, iter_nb + 1, max_iter, loss, loss_caption, loss_classification, lr)
        self.logger.debug(line)
        overall_iter = (epoch * max_iter) + iter_nb
        self.tensorboard_writer.add_scalar(
            f'loss/loss_caption', loss_caption, overall_iter)
        self.tensorboard_writer.add_scalar(
            f'loss/loss_classification', loss_classification, overall_iter)
        self.tensorboard_writer.add_scalar(
            f'loss/total_loss', loss, overall_iter)
        self.tensorboard_writer.add_scalar(f'lr', lr, overall_iter)

    def epochStartCallback(self, epoch: int, max_epochs: int) -> None:
        self.logger.debug('Epoch [%d/%d] starting.', epoch, max_epochs)

    def epochEndCallback(self,
                         epoch: int,
                         max_epochs: int,
                         train_loss: float,
                         model: torch.nn.Module,
                         optimizer: torch.optim,
                         scheduler: torch.optim.lr_scheduler,
                         best_mse: float) -> None:
        self.logger.debug(
            'Epoch [%d/%d] finished with train_loss [%.2f].\n', epoch, max_epochs, train_loss)
        self.logger.info(
            'Epoch [%d/%d] finished with train_loss [%.2f].\n', epoch, max_epochs, train_loss)

        if epoch % self.modelCheckpointInterval == 0:
            self.saveTrainState(epoch, model, optimizer,
                                scheduler, best_mse)

    def deleteModel(self,
                    is_not_delete_model: bool,
                    epoch: int) -> None:
        current_model = 'model_{:04d}.pt'.format(epoch)

        if len(os.listdir(self.modelsDirpath)):
            for content in os.listdir(self.modelsDirpath):
                if (content != current_model and is_not_delete_model) or (content == current_model and not is_not_delete_model):
                    os.remove(os.path.join(self.modelsDirpath, content))

    def trainStartCallback(self) -> None:
        self.logger.debug('Beginning training session.\n')

    def modelParamsCallback(self, model: torch.nn.Module) -> None:
        self.logger.info('Model architecture:\n')
        self.logger.debug('Model architecture:\n')
        self.logger.info(model)
        self.logger.debug(model)
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f'Total number of parameters: {total_params}\n')
        self.logger.debug(f'Total number of parameters: {total_params}\n')

    def trainEndCallback(self) -> None:
        self.logger.debug('Training session finished.\n')

    def evalStartCallback(self) -> None:
        self.logger.debug('Beginning testing session.\n')

    def evalEndCallback(self,
                        lossCaption: float,
                        lossClassification: float,
                        totalLoss: float,
                        epochEvaluated: int,
                        onVal: bool) -> None:
        mode = 'validation' if onVal else 'test'

        # log tensorboard metrics
        self.tensorboard_writer.add_scalar(
            f'loss_caption_{mode}', lossCaption, epochEvaluated)
        self.tensorboard_writer.add_scalar(
            f'loss_classification_{mode}', lossClassification, epochEvaluated)
        self.tensorboard_writer.add_scalar(
            f'total_loss_{mode}', totalLoss, epochEvaluated)

        if onVal:
            self.logger.debug(
                f'{mode} session finished on model after epoch {epochEvaluated}.\n')
        else:
            self.logger.debug(f'{mode} session finished.\n')

        self.logger.info(
            f"loss_caption {mode}: {lossCaption:.5f}, loss_classification {mode}:{lossClassification:.5f}, total_loss_ {mode}:{totalLoss:.5f}.\n")

    def endTestCallback(self,
                        metricsNLU,
                        metricsClassification,
                        epochEvaluated) -> None:
        log = f"Epoch {epochEvaluated} - "
        for key, value in metricsNLU.items():
            self.tensorboard_writer.add_scalar(
                f"{key}", value, epochEvaluated)
            log += f"{key}: {value:.5f}, "

        for key, value in metricsClassification.items():
            self.tensorboard_writer.add_scalar(
                f"{key}", value, epochEvaluated)
            log += f"{key}: {value:.5f}, "
        self.logger.info(log)
