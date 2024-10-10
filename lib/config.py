import json

import torch
from lib.datasets.dataset import IuDataset, AtHotDataset
from lib.models import decoder, encoder
from lib.models.encoder.bert import BertConfig, BertModel
from lib.models.tokenizer import Tokenizer
from lib.models.visonencdec import VisonEncDecModel
from transformers import AutoModel

pretrained = {
    "vit_224": 'pretrain/vit-base-patch16-224',
    "vit_384": 'pretrain/vit-base-patch16-384',
    "swin_224": 'pretrain/swin-base-patch4-window7-224',
    "swin_384": 'pretrain/swin-base-patch4-window12-384',
}


def readJson(rpath):
    with open(rpath, 'r') as f:
        return json.load(f)


class LanguageModelCriterion(torch.nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


class Config:
    def __init__(self, args, device):
        self.config = {}
        self.load_config(args)
        self.device = device

    def load_config(self, args):
        self.config["datasetName"] = args.dataset
        self.config["datasets"] = {}
        self.config["datasets"]["dataFile"] = args.dataFile
        self.config["datasets"]["imageSize"] = (args.imageSize, args.imageSize)
        self.config["datasets"]["tokenizer"] = self.get_tokenizer(args.dataset)
        self.config["datasets"]["seq_length"] = args.max_seq_length

        self.config["model"] = {}
        encoderConfig = {}
        encoderConfig["image_size"] = (args.imageSize, args.imageSize)
        self.config["model"]["encoder"] = {
            "config": encoderConfig,
            "model_checkpoint": f"{args.pretrain}/encoder/{args.encoder_name}/pytorch_model.bin",
        }
        self.config["pretrained"] = args.pretrained
        if args.encoder_name == "vit":
            self.config["model"]["encoder"]["name"] = (
                "VisionTransformer", "ViTConfig")
            self.config["model"]["encoder"]["config"] = {k.replace(
                "encoder_", ""): v for k, v in args.__dict__.items() if k.startswith("encoder_")}
        elif args.encoder_name == "swin":
            self.config["model"]["encoder"]["name"] = (
                "SwinTransformer", "SwinConfig")
        self.config["model"]["decoder"] = {
            "name": ("GPT2LMHeadModel", "GPT2Config"),
            "config": {k.replace("decoder_", ""): v for k, v in args.__dict__.items() if k.startswith("decoder_")},
            "model_checkpoint": f"{args.pretrain}/decoder/pytorch_model.bin",
        }

        self.config["optimizer"] = {}
        self.config["optimizer"]["name"] = "Adam"
        self.config["optimizer"]["parameters"] = {}
        self.config["optimizer"]["parameters"]["lr"] = 5e-4

        self.config["loss"] = "CrossEntropyLoss"

        self.config["optimizer"]["parameters"]["weight_decay"] = 1e-5

        self.config['lr_scheduler'] = {}
        self.config['lr_scheduler']['name'] = "CosineAnnealingWarmRestarts"
        self.config["lr_scheduler"]["parameters"] = {'T_0': 10}

        self.config["exp_name"] = args.exp_name

    def get_tokenizer(self, datasetName):
        if datasetName == "iu":
            return Tokenizer('iu-dataset/iu_xray_data.json', 'iu_xray', **self.config["datasets"])
        return Tokenizer('vn-dataset/ann.json', 'vn_xray', **self.config["datasets"])

    def getDataset(self, **kwargs):
        if self.config["datasetName"] == "iu":
            return IuDataset(**kwargs, **self.config["datasets"])
        return AtHotDataset(**kwargs, **self.config["datasets"])

    def getEncoder(self, **kwargs):
        pretrained_model = self.config['pretrained']
        if pretrained_model:
            imageSize = self.config["datasets"]["imageSize"][0]
            encoderName = pretrained[f"{pretrained_model}_{imageSize}"]
            return AutoModel.from_pretrained(encoderName)

        name = self.config["model"]["encoder"]["name"][0]
        config = getattr(encoder, self.config["model"]["encoder"]["name"][1])(
            **self.config["model"]["encoder"]["config"])
        return getattr(encoder, name)(config=config, **kwargs)

    def getTagEncoder(self, **kwargs):
        encoder_config = BertConfig.from_json_file(
            "./lib/models/encoder/config/bert_config.json")
        encoder_config.vocab_size = self.config["datasets"]["tokenizer"].get_vocab_size(
        )
        encoder_config.hidden_size = kwargs['hidden_size']
        encoder_config.encoder_width = kwargs['hidden_size']
        tagEncoder = BertModel(config=encoder_config,
                               add_pooling_layer=False)
        return tagEncoder

    def getTagPredict(self, **kwargs):
        encoder_config = BertConfig.from_json_file(
            "./lib/models/encoder/config/bert_config.json")
        encoder_config.vocab_size = self.config["datasets"]["tokenizer"].get_vocab_size(
        )
        encoder_config.hidden_size = kwargs['hidden_size']
        encoder_config.encoder_width = kwargs['hidden_size']
        tagHead = BertModel(config=encoder_config,
                            add_pooling_layer=False)
        return tagHead

    def getDecoder(self, **kwargs):
        name = self.config["model"]["decoder"]["name"][0]
        config = getattr(decoder, self.config["model"]["decoder"]["name"][1])(
            **self.config["model"]["decoder"]["config"])
        config.vocab_size = self.config["datasets"]["tokenizer"].get_vocab_size(
        )
        config.bos_token_id = 0
        config.eos_token_id = 0
        decoderModel = getattr(decoder, name)(config=config, **kwargs)
        return decoderModel

    def getModel(self, **kwargs):
        encoder = self.getEncoder(**kwargs)
        parameter_model = {"encoder": encoder}
        self.config["model"]["decoder"]["config"]['hidden_size'] = encoder.config.hidden_size
        decoder = self.getDecoder(**kwargs)
        parameter_model["decoder"] = decoder
        tagEncoder = self.getTagEncoder(
            hidden_size=encoder.config.hidden_size, encoder_width=encoder.config.hidden_size)
        tagPredict = self.getTagPredict(
            hidden_size=encoder.config.hidden_size, encoder_width=encoder.config.hidden_size)
        parameter_model["tagEncoder"] = tagEncoder
        parameter_model["tagPredict"] = tagPredict
        parameter_model["tokenizer"] = self.config["datasets"]["tokenizer"]
        parameter_model["pretrain"] = True if self.config["pretrained"] else False
        ex_path = ""
        if self.config["datasetName"] == "iu":
            with open(f"tags/iu{ex_path}.txt", 'r') as f:
                parameter_model['listTag'] = f.read().splitlines()
            with open(f'tags/thresh_iu{ex_path}.txt', 'r') as f:
                parameter_model['threshold'] = [
                    float(i) for i in f.readlines()]
        else:
            with open("tags/vn.txt", 'r') as f:
                parameter_model['listTag'] = f.read().splitlines()
            with open('tags/thresh_vn.txt', 'r') as f:
                parameter_model['threshold'] = [
                    float(i) for i in f.readlines()]
        return VisonEncDecModel(**parameter_model)

    def getOptimizer(self, model):
        ve_params = list(map(id, model.encoder.parameters()))
        ed_params = filter(lambda x: id(
            x) not in ve_params, model.parameters())
        model_parameters = [
            {'params': model.encoder.parameters(
            ), 'lr': 5e-5},
            {'params': ed_params,
                'lr': 1e-4},
        ]
        optimizer = getattr(torch.optim, 'Adam')(
            model_parameters, weight_decay=5e-5, amsgrad=True)
        return optimizer

    def compute_loss(self, output, reports_ids, reports_masks):
        criterion = LanguageModelCriterion()
        loss = criterion(
            output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
        return loss

    def getLossFunction(self):
        return self.compute_loss

    def getLrScheduler(self, optimizer):
        return getattr(torch.optim.lr_scheduler,
                       self.config['lr_scheduler']['name'])(optimizer, **self.config['lr_scheduler']['parameters'])

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config
