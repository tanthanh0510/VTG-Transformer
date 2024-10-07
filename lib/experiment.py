import os
import re
import logging

import torch
from torch.utils.tensorboard import SummaryWriter


class Experiment:
    def __init__(self,
                 exp_name: str,
                 model_checkpoint_interval: int = 100,
                 mode: str = 'train',
                 exps_basedir: str = 'experiments',
                 tensorboard_dir: str = 'tensorboard',
                 pretrain: str = 'pretrain') -> None:
        self.name = exp_name
        self.pretrain = pretrain
        self.exp_dirpath = os.path.join(exps_basedir, exp_name)
        self.models_dirpath = os.path.join(self.exp_dirpath, 'models')
        self.best_models_dirpath = os.path.join(
            self.exp_dirpath, 'best_models')
        self.results_dirpath = os.path.join(self.exp_dirpath, 'results')
        self.log_path = os.path.join(self.exp_dirpath, f'log_{mode}.txt')
        self.tensorboard_writer = SummaryWriter(
            os.path.join(tensorboard_dir, exp_name))
        self.model_checkpoint_interval = model_checkpoint_interval
        self.setup_exp_dir()
        self.setup_logging()

    def setup_exp_dir(self) -> None:
        if not os.path.exists(self.exp_dirpath):
            os.makedirs(self.exp_dirpath)
            os.makedirs(self.models_dirpath)
            os.makedirs(self.results_dirpath)
            os.makedirs(self.best_models_dirpath)
        if not os.path.exists(self.pretrain):
            os.makedirs(self.pretrain)

    def setup_logging(self) -> None:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.DEBUG, handlers=[
                            file_handler, stream_handler])
        self.logger = logging.getLogger(__name__)

    def get_last_checkpoint_epoch(self) -> int:
        pattern = re.compile('model_(\\d+).pt')
        last_epoch = -1
        for ckpt_file in os.listdir(self.models_dirpath):
            result = pattern.match(ckpt_file)
            if result is not None:
                epoch = int(result.groups()[0])
                if epoch > last_epoch:
                    last_epoch = epoch
        return last_epoch

    def get_best_checkpoint_epoch(self, name: str = 'bleu1') -> int:
        return os.path.join(self.best_models_dirpath, f"best_{name}_model.pt")

    def get_checkpoint_path(self, epoch: int):
        return os.path.join(self.models_dirpath, 'model_{:04d}.pt'.format(epoch))

    def get_epoch_model(self, epoch: int):
        return torch.load(self.get_checkpoint_path(epoch))['model']

    def get_best_model(self, name: str = 'bleu1'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(os.path.join(
            self.best_models_dirpath, f"best_{name}_model.pt"), map_location=device)
        self.logger.info(
            f"Best model loaded from {os.path.join(self.best_models_dirpath, f'best_{name}_model.pt')}")
        self.logger.info(f"Metrics NLU: {checkpoint['metricsNLU']}")
        self.logger.info(
            f"Metrics Classification: {checkpoint['metricsClassification']}")
        return checkpoint['model']

    def load_last_train_state(self,
                              model: torch.nn.Module,
                              optimizer: torch.optim,
                              scheduler: torch.optim.lr_scheduler,
                              best_score: dict) -> tuple:
        epoch = self.get_last_checkpoint_epoch()
        train_state_path = self.get_checkpoint_path(epoch)
        train_state = torch.load(train_state_path)
        model.load_state_dict(train_state['model'])
        optimizer.load_state_dict(train_state['optimizer'])
        scheduler.load_state_dict(train_state['scheduler'])
        if train_state['best_score']:
            best_score = train_state['best_score']
        return epoch, model, optimizer, scheduler, best_score

    def save_best_model(self,
                        model: torch.nn.Module,
                        epoch: int,
                        metricsNLU: dict,
                        metricsClassification: dict,
                        name: str
                        ) -> None:
        torch.save({
            'model': model.state_dict(),
            'metricsNLU': metricsNLU,
            'metricsClassification': metricsClassification
        }, os.path.join(self.best_models_dirpath, f"best_{name}_model.pt"))

    def save_train_state(self,
                         epoch: int,
                         model: torch.nn.Module,
                         optimizer: torch.optim,
                         scheduler: torch.optim.lr_scheduler,
                         best_score: dict) -> None:
        train_state_path = self.get_checkpoint_path(epoch)
        torch.save(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_score': best_score,
            }, train_state_path)

    def iter_end_callback(self,
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

    def epoch_start_callback(self, epoch: int, max_epochs: int) -> None:
        self.logger.debug('Epoch [%d/%d] starting.', epoch, max_epochs)

    def epoch_end_callback(self,
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

        if epoch % self.model_checkpoint_interval == 0:
            self.save_train_state(epoch, model, optimizer,
                                  scheduler, best_mse)

    def delete_model(self,
                     is_not_delete_model: bool,
                     epoch: int) -> None:
        current_model = 'model_{:04d}.pt'.format(epoch)

        if len(os.listdir(self.models_dirpath)):
            for content in os.listdir(self.models_dirpath):
                if (content != current_model and is_not_delete_model) or (content == current_model and not is_not_delete_model):
                    os.remove(os.path.join(self.models_dirpath, content))

    def train_start_callback(self) -> None:
        self.logger.debug('Beginning training session.\n')

    def model_params_callback(self, model: torch.nn.Module) -> None:
        self.logger.info('Model architecture:\n')
        self.logger.debug('Model architecture:\n')
        self.logger.info(model)
        self.logger.debug(model)
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f'Total number of parameters: {total_params}\n')
        self.logger.debug(f'Total number of parameters: {total_params}\n')

    def train_end_callback(self) -> None:
        self.logger.debug('Training session finished.\n')

    def eval_start_callback(self) -> None:
        self.logger.debug('Beginning testing session.\n')

    def eval_end_callback(self,
                          loss_caption: float,
                          loss_classification: float,
                          total_loss: float,
                          epoch_evaluated: int,
                          on_val: bool) -> None:
        mode = 'validation' if on_val else 'test'

        # log tensorboard metrics
        self.tensorboard_writer.add_scalar(
            f'loss_caption_{mode}', loss_caption, epoch_evaluated)
        self.tensorboard_writer.add_scalar(
            f'loss_classification_{mode}', loss_classification, epoch_evaluated)
        self.tensorboard_writer.add_scalar(
            f'total_loss_{mode}', total_loss, epoch_evaluated)

        if on_val:
            self.logger.debug(
                f'{mode} session finished on model after epoch {epoch_evaluated}.\n')
        else:
            self.logger.debug(f'{mode} session finished.\n')

        self.logger.info(
            f"loss_caption {mode}: {loss_caption:.5f}, loss_classification {mode}:{loss_classification:.5f}, total_loss_ {mode}:{total_loss:.5f}.\n")

    def end_test_callback(self,
                          metricsNLU,
                          metricsClassification,
                          epoch_evaluated) -> None:
        log = f"Epoch {epoch_evaluated} - "
        for key, value in metricsNLU.items():
            self.tensorboard_writer.add_scalar(
                f"{key}", value, epoch_evaluated)
            log += f"{key}: {value:.5f}, "

        for key, value in metricsClassification.items():
            self.tensorboard_writer.add_scalar(
                f"{key}", value, epoch_evaluated)
            log += f"{key}: {value:.5f}, "
        self.logger.info(log)
