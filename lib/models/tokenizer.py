import json
import re
from collections import Counter

import torch


class Tokenizer(object):
    def __init__(self, ann_path, dataset_name, threshold=3, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.ann_path = ann_path
        self.dataset_name = dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
            self.threshold = threshold
        else:
            self.clean_report = self.clean_report_vn_cxr
            self.threshold = 2
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        only_findings = getattr(self, 'only_findings', False)
        if self.dataset_name == 'iu_xray':
            total_tokens = []
            for example in self.ann['train']:
                if only_findings and 'normal' in self.ann['train'][example]['tags_key']:
                    continue
                report = self.ann['train'][example]['caption'].replace(
                    '<#findings#>', ' ')
                tokens = self.clean_report(report).split()
                for token in tokens:
                    total_tokens.append(token)
        else:
            total_tokens = []
            for example in self.ann['train']:
                report = self.ann['train'][example]['finding']
                impression = self.ann['train'][example]['impression']
                report = report + ' ' + impression
                tokens = self.clean_report(report).split()
                for token in tokens:
                    total_tokens.append(token)
        total_tokens.append('|'*(self.threshold+1))
        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + \
            ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_vn_cxr(self, report):
        def report_cleaner(t): return t.replace('\n', ' ').replace('~', ' ').replace('-', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')

        def sent_cleaner(t): return re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                           .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(
            report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_iu_xray(self, report):
        def report_cleaner(t): return t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        def sent_cleaner(t): return re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                           replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(
            report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def batch_encode(self, reports, max_seq_length=100, pad_to_max_length=True, return_tensors="pt"):
        out, max_length = [], 0
        for report in reports:
            ids = self(report)
            out.append(ids)
            max_length = max(max_length, len(ids))
        if max_length < max_seq_length:
            attention_mask = []
            if pad_to_max_length:
                for i, ids in enumerate(out):
                    out[i] = ids + [0] * (max_length - len(ids))
                    attention_mask.append([1] * len(ids) + [0] * (max_length - len(ids))
                                          )
        else:
            attention_mask = []
            if pad_to_max_length:
                for i, ids in enumerate(out):
                    if len(ids) > max_seq_length:
                        out[i] = ids[:max_seq_length]
                        attention_mask.append([1] * (max_seq_length))
                    else:
                        out[i] = ids + [0] * (max_seq_length - len(ids))
                        attention_mask.append([1] * len(ids) + [0] * (max_seq_length - len(ids))
                                              )
        if return_tensors == "pt":
            out = torch.LongTensor(out)
            attention_mask = torch.LongTensor(attention_mask)
        return out, attention_mask

    def get_vocab_size(self):
        return len(self.token2idx)+1

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
