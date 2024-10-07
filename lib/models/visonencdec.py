import math
import os
import warnings
from math import ceil
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")


def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module,
                                base_model_prefix: str, skip_key: str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        print(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name + ' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([
                module_name + "/" + sub_name
                for sub_name in encoder_modules.keys()
            ])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(
                            decoder_modules[decoder_name],
                            type(encoder_modules[encoder_name])) and len(
                                encoder_modules) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix,
                                       uninitialized_encoder_weights, skip_key)


class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = x[:, None, :]
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class VisonEncDecModel(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 tagEncoder: nn.Module,
                 tagPredict: nn.Module,
                 listTag: list,
                 threshold: list,
                 tokenizer: Optional[nn.Module] = None,
                 **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if "config" in self.encoder.__dict__:
            self.numFeature = self.encoder.config.hidden_size
        else:
            self.numFeature = self.encoder.num_features
        if (self.numFeature != self.decoder.embed_dim):
            self.enc_to_dec_proj = nn.Linear(
                self.numFeature, self.decoder.embed_dim)
        self.listTag = np.array(listTag)
        self.numTag = len(listTag)
        self.tagEncoder = tagEncoder
        self.tagPredict = tagPredict
        self.label_embed = nn.Embedding(
            self.numTag, self.tagEncoder.config.hidden_size)
        if self.tagEncoder.config.hidden_size != self.numFeature:
            self.tag_to_enc_proj = nn.Linear(
                self.tagEncoder.config.hidden_size, self.numFeature)
        self.tagHead = GroupWiseLinear(self.numTag,
                                       self.numFeature,
                                       bias=True)
        self.del_selfattention()
        tie_encoder_decoder_weights(self.tagEncoder, self.tagPredict, '',
                                    ' ')
        self.tokenizer = tokenizer
        self.threshold = torch.FloatTensor(threshold)
        for key, value in kwargs.items():
            setattr(self, key, value)

    # delete self-attention layer of image-tag recognition decoder to reduce computation, follower Query2Label
    def del_selfattention(self):
        del self.tagPredict.embeddings
        for layer in self.tagPredict.encoder.layer:
            del layer.attention

    def data_parallel(self, pixel_values: torch.Tensor, device_ids, output_device=None, **kwargs):
        if not device_ids or len(device_ids) == 1:
            return self(pixel_values, **kwargs)
        if output_device is None:
            output_device = device_ids[0]

        replicas = nn.parallel.replicate(self, device_ids)

        inputs = nn.parallel.scatter(pixel_values, device_ids)
        kwargs = nn.parallel.scatter(kwargs, device_ids)

        replicas = replicas[:len(inputs)]
        kwargs = kwargs[:len(inputs)]
        outputs = nn.parallel.parallel_apply(replicas, inputs, kwargs)

        return nn.parallel.gather(outputs, output_device)

    def forward(self,
                pixel_values: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                tagInput: Optional[torch.LongTensor] = None,
                ) -> dict:
        bs = pixel_values.shape[0]
        if getattr(self, 'pretrain', None) is None:
            imageEmbeddings, pooler = self.encoder(
                pixel_values=pixel_values)
        else:
            output = self.encoder(
                pixel_values=pixel_values, return_dict=True)
            imageEmbeddings = output.last_hidden_state

        imageAtts = torch.ones(imageEmbeddings.size()[:-1],
                               dtype=torch.long).to(pixel_values.device)
        label_embed = self.label_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        tagging_embed = self.tagPredict(
            encoder_embeds=label_embed,
            encoder_hidden_states=imageEmbeddings,
            encoder_attention_mask=imageAtts,
            return_dict=False,
            mode='tagging',
        )
        logits = self.tagHead(tagging_embed[0])

        tag = tagInput.cpu().numpy()
        tagInput = []
        for b in range(bs):
            index = np.argwhere(tag[b] == 1)
            token = self.listTag[index].squeeze(axis=1)
            tagInput.append(' | '.join(token))

        tag_input_tokenzier, attention_mask = self.tokenizer.batch_encode(
            tagInput)
        tag_input_tokenzier = tag_input_tokenzier.to(pixel_values.device)
        attention_mask = attention_mask.to(pixel_values.device)

        output_tagembedding = self.tagEncoder(
            tag_input_tokenzier,
            attention_mask=attention_mask,
            encoder_hidden_states=imageEmbeddings,
            encoder_attention_mask=imageAtts,
            return_dict=True,
        )

        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=output_tagembedding.last_hidden_state,
            inputs_embeds=decoder_inputs_embeds
        )

        return logits, outputs[0]

    def generate(self,
                 pixel_values: torch.Tensor,
                 max_length: int,
                 min_length: int = 20,
                 tags: torch.LongTensor = None,
                 **kwargs) -> torch.LongTensor:
        device = pixel_values.device
        batch, height, width, channel = pixel_values.shape
        bos_token_ids = (torch.ones(
            (batch, 1), dtype=torch.long, device=device) * 0).to(device)
        self.eval()
        self.decoder.eval()
        self.encoder.eval()
        self.tagEncoder.eval()
        if getattr(self, 'pretrain', None) is None:
            imageEmbeddings, pooler = self.encoder(
                pixel_values=pixel_values)
        else:
            output = self.encoder(
                pixel_values=pixel_values, return_dict=True)
            imageEmbeddings = output.last_hidden_state

        image_atts = torch.ones(imageEmbeddings.size()[:-1],
                                dtype=torch.long).to(pixel_values.device)
        bs = imageEmbeddings.shape[0]
        label_embed = self.label_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        tagging_embed = self.tagPredict(
            encoder_embeds=label_embed,
            encoder_hidden_states=imageEmbeddings,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )

        logits = self.tagHead(tagging_embed[0])

        targets = torch.where(
            torch.sigmoid(logits) > self.threshold.to(
                pixel_values.device),
            torch.tensor(1.0).to(pixel_values.device),
            torch.zeros(self.numTag).to(pixel_values.device))
        targets = targets.view(bs, -1)
        tag = targets.cpu().numpy()
        tag_input = []
        for b in range(bs):
            index = np.argwhere(tag[b] == 1)
            token = self.listTag[index].squeeze(axis=1)
            tag_input.append(' | '.join(token))
        tag_input_tokenzier, attention_mask = self.tokenizer.batch_encode(
            tag_input)
        tag_input_tokenzier = tag_input_tokenzier.to(pixel_values.device)
        attention_mask = attention_mask.to(pixel_values.device)

        output_tagembedding = self.tagEncoder(
            tag_input_tokenzier,
            attention_mask=attention_mask,
            encoder_hidden_states=imageEmbeddings,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return_dict_in_generate = kwargs.get("return_dict_in_generate", False)
        model_kwargs = {
            "encoder_hidden_states": output_tagembedding.last_hidden_state,
            "encoder_attention_mask": None,
            "return_dict_in_generate": return_dict_in_generate,
            "output_scores": kwargs.get("output_scores", False),
            "do_sample": False,
        }
        outputs = self.decoder.generate(
            input_ids=bos_token_ids,
            max_length=max_length,
            min_length=min_length,
            eos_token_id=0,
            pad_token_id=0,
            repetition_penalty=1.0,
            **model_kwargs)

        if return_dict_in_generate:
            return targets, outputs["sequences"], outputs['scores']
        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output.cpu().numpy()[1:])
            captions.append(caption)
        return targets, captions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(
        sorted_indices_to_remove, (1, -1), value=False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def top_k(logits, frac_num_tokens=0.1, k=None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def top_a(logits, min_p_pow=2.0, min_p_ratio=0.02):
    probs = F.softmax(logits, dim=-1)
    max_probs = torch.amax(probs, dim=-1, keepdim=True)
    limit = torch.pow(max_probs, min_p_pow) * min_p_ratio
    return torch.where(probs < limit, float('-inf'), logits)
