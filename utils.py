from typing import Dict, Optional
import os
import time

from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from mlflow.pyfunc import PythonModel, PythonModelContext
from sklearn.metrics import classification_report, f1_score


TOKENS = "tokens"
NER_TAGS = "ner_tags"
IGNORE_INDEX = -100
LABELS = "labels"


def encode_with_label(
    examples,
    tokenizer,
    padding=True,
    max_length=512,
):
    tokenized_inputs = tokenizer(
        examples[TOKENS],
        padding=padding,
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[NER_TAGS]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is not None and word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(IGNORE_INDEX)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs[LABELS] = labels
    return tokenized_inputs


def evaluate(model, dataloader, device, target_names):
    losses = AverageMeter()
    model.eval()
    ground_true = []
    prediction = []
    start_time = time.time()
    for step, batch in enumerate(tqdm(dataloader, desc="Evaluating ... ", position=2)):
        labels = batch[LABELS]
        batch_size = labels.size(0)
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss

        losses.update(loss.item(), batch_size)
        pred_ids = outputs.logits.argmax(dim=-1)[labels != IGNORE_INDEX].cpu().numpy().tolist()
        labels = labels[labels != IGNORE_INDEX].cpu().numpy().tolist()
        ground_true.extend(labels)
        prediction.extend(pred_ids)

    cls_report = classification_report(ground_true, prediction, target_names=target_names)
    f1 = f1_score(ground_true, prediction, average="macro")
    duration = time.time() - start_time

    return losses.avg, f1, cls_report, duration


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TokenCLSForParagraphSegmentationAPI(PythonModel):
    def __init__(self, id2label: Optional[Dict[int, str]] = None) -> None:
        super().__init__()
        self.id2label = {
            1: "Others",
            3: "Findings",
            5: "Impression",
        }
        if id2label is not None:
            self.id2label = id2label

    def load_context(self, context: PythonModelContext):
        model_path = os.path.dirname(context.artifacts["config"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.eval()

    def predict(self, context, df):
        results = []
        data = df.text.apply(lambda x: x.split(" ")).values.tolist()
        inputs = self.tokenizer(
            data,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )

        if self.model.device.index != None:
            torch.cuda.empty_cache()
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.model.device.index)

        with torch.no_grad():
            prediction = self.model(**inputs).logits.argmax(dim=-1)

        for i, pred_ids in enumerate(prediction):
            output = {
                v: []
                for k, v in self.id2label.items()
            }
            point = None
            paragraph_type = None
            previous_word_idx = 0
            for pred_idx, word_idx in zip(pred_ids, inputs.word_ids(batch_index=i)):
                if word_idx is not None and word_idx == previous_word_idx:
                    pred_idx = pred_idx.item()
                    if pred_idx in self.id2label:
                        if paragraph_type is not None:
                            output[paragraph_type].append(" ".join(data[i][point:word_idx]))
                        paragraph_type = self.id2label[pred_idx]
                        point = word_idx
                    previous_word_idx += 1

            paragraph_type = "Others" if paragraph_type is None else paragraph_type
            output[paragraph_type].append(" ".join(data[i][point:len(data[i])]))
            for key in output:
                output[key] = "\n".join(output[key])
            results.append(output)

        return pd.DataFrame(results)
