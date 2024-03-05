from typing import Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class TokenCLSForParagraphSegmentationPipeline:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        device: Optional[torch.device] = None,
        id2label: Optional[Dict[int, str]] = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path).to(self.device)
        self.model.eval()
        self.id2label = {
            1: "Others",
            3: "Findings",
            5: "Impression",
        }
        if id2label is not None:
            self.id2label = id2label

    def __call__(self, text: str) -> str:
        results = {
            v: []
            for k, v in self.id2label.items()
        }
        words = text.split(" ")
        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
        )

        torch.cuda.empty_cache()
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)

        with torch.no_grad():
            pred_ids = self.model(**inputs).logits.argmax(dim=-1)[0]

        point = None
        paragraph_type = None
        previous_word_idx = 0
        for pred_idx, word_idx in zip(pred_ids, inputs.word_ids()):
            if word_idx is not None and word_idx == previous_word_idx:
                pred_idx = pred_idx.item()
                if pred_idx in self.id2label:
                    if paragraph_type is not None:
                        results[paragraph_type].append(" ".join(words[point:word_idx]))
                    paragraph_type = self.id2label[pred_idx]
                    point = word_idx
                previous_word_idx += 1

        paragraph_type = "Others" if paragraph_type is None else paragraph_type
        results[paragraph_type].append(" ".join(words[point:len(words)]))

        for key in results:
            results[key] = "\n".join(results[key])

        return results


if __name__ == "__main__":
    pipeline = TokenCLSForParagraphSegmentationPipeline("models/best_model")
    text = "Findings: 1. A 2 cm mass in the right upper lobe, highly suspicious for primary lung cancer. 2. Scattered ground-glass opacities in both lungs, possible early sign of interstitial lung disease. 3. No significant mediastinal lymph node enlargement. 4. Mild pleural effusion on the left side. 5. No evidence of bone metastasis in the visualized portions of the thorax. Conclusion: A. Right upper lobe mass suggestive of lung cancer; biopsy recommended. B. Ground-glass opacities; suggest follow-up CT in 3 months. C. Mild pleural effusion; may require thoracentesis if symptomatic."
    results = pipeline(text)
    for key in results:
        print(f"===== {key} =====")
        print(f"{results[key]}\n")
