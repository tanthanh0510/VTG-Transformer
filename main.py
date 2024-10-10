import argparse
import torch

from lib.experiment import Experiment
from lib.config import Config
from lib.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description='Image captioning with Transformer')
    parser.add_argument(
        'mode', choices=['train', 'test'], default='train', help='Train or test')
    parser.add_argument('dataset', choices=['iu', 'ad-hoc'], default='iu',
                        help='Choose dataset (ad-hoc or iu')
    parser.add_argument('--dataFile', help='path file data', required=True)
    parser.add_argument('--pretrain', default='checkpoints',
                        help='path pretrain model')

    parser.add_argument('--exp_name', help='Experiment name', required=True)

    parser.add_argument('--train_epochs', type=int, default=50,
                        help='Epochs to train the model (Default: 50)')
    parser.add_argument('--train_batch', type=int, default=160,
                        help='Traning batch size (Default: 160)')
    parser.add_argument('--test_batch', type=int, default=160,
                        help='Testing batch size (Default: 160)')
    parser.add_argument('--val_on_epoch', type=int, default=1,
                        help='Validation on epoch (Default: 1)')
    parser.add_argument('--test_on_epoch', type=int, default=1,
                        help='Validation on epoch (Default: 1)')
    parser.add_argument('--encoder_name', choices=['vit', 'swin'], default='vit',
                        help='Encoder name (Default: vit)')
    parser.add_argument('--pretrained', choices=['vit', 'swin', None], default=None,
                        help='Pretrained model (Default: None)')

    parser.add_argument('--imageSize', type=int, default=768,
                        help='Image size (Default: 768)')
    parser.add_argument('--max_seq_length', type=int, default=60)
    parser.add_argument('--deterministic',
                        action='store_true',
                        help='set cudnn.deterministic = True and cudnn.benchmark = False')
    parser.add_argument('--encoder_num_hidden_layers', type=int, default=3,
                        help='Number of hidden layers in encoder (Default: 3)')
    parser.add_argument('--encoder_num_attention_heads', type=int, default=12,
                        help='Number of attention heads in encoder (Default: 12)')
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=3,
                        help='Number of hidden layers in decoder (Default: 3)')
    parser.add_argument('--decoder_num_attention_heads', type=int, default=8,
                        help='Number of attention heads in decoder (Default: 8)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device('cpu') if not torch.cuda.is_available(
    ) else torch.device('cuda')
    deviceIds = [i for i in range(torch.cuda.device_count())]
    exp = Experiment(args.exp_name, mode=args.mode,
                     modelCheckpointInterval=args.val_on_epoch, pretrain=args.pretrain)
    cfg = Config(args, device)
    runner = Runner(cfg, exp, epochs=args.train_epochs, testBatchSize=args.test_batch, valOnEpochs=args.val_on_epoch,
                    trainBatchSize=args.train_batch, testOnEpochs=args.test_on_epoch, deviceIds=deviceIds)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'test':
        runner.test()
    else:
        runner.val()


if __name__ == '__main__':
    main()
