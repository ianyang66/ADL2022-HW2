# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exact Match & F1 metric."""

import datasets
import collections


import spacy
from tqdm import tqdm

_CITATION = ''
_DESCRIPTION = ''
_KWARGS_DESCRIPTION = ''

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class EM_F1(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {"id": datasets.Value("string"), "prediction_text": datasets.Value("string")},
                    "references": {
                        "id": datasets.Value("string"),
                        "answers": datasets.features.Sequence(
                            {
                                "text": datasets.Value("string"),
                                "answer_start": datasets.Value("int32"),
                            }
                        ),
                    },
                }
            ),
        )

    def _compute(self, predictions, references):
        pred_dict = {prediction["id"]: prediction["prediction_text"] for prediction in predictions}
        ref_dict = {reference["id"]: {"answers": reference["answers"]["text"]} for reference in references}
        tokenizer = Tokenizer()

        score = compute_metrics(ref_dict, pred_dict, tokenizer)
        return score



class Tokenizer:
    def __init__(self):
        self.nlp = spacy.load('zh_core_web_md', disable=['ner', 'parser', 'tagger'])

    def __call__(self, text, remove_punc=False):
        tokens = list(self.nlp(text))
        if remove_punc:
            tokens = [e for e in tokens if not e.is_punct]
        tokens = [e.text for e in tokens]
        return tokens


def compute_em(ans, pred):
    def em(a, p):
        return int(''.join(a) == ''.join(p))

    return max([em(a, pred) for a in ans])


def compute_f1(ans, pred):
    def f1(a, p):
        common = collections.Counter(a) & collections.Counter(p)
        tp = sum(common.values())
        if tp == 0:
            return 0
        precision = tp / len(p)
        recall = tp / len(a)

        return (2 * precision * recall) / (precision + recall)

    return max([f1(a, pred) for a in ans])


def compute_metric(ans, pred, tokenizer):
    ans = [tokenizer(a, remove_punc=True) for a in ans]
    pred = tokenizer(pred, remove_punc=True)

    return {
        'em': compute_em(ans, pred),
        'f1': compute_f1(ans, pred)
    }


def compute_metrics(answers, predictions, tokenizer):
    metrics = []
    for id_ in tqdm(list(answers.keys()), desc='[*] Evaluating', dynamic_ncols=True):
        if id_ not in predictions:
            print(f'[!] Cannot find answer for id {id_} in model predictions')
            continue
        prediction = predictions[id_]
        metric = compute_metric(answers[id_]['answers'], prediction, tokenizer)
        metrics.append(metric)

    num_total = len(metrics)
    result = {
        'count': num_total,
        'em': sum([m['em'] for m in metrics]) / num_total,
        'f1': sum([m['f1'] for m in metrics]) / num_total
    }

    return result