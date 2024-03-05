# Paragraph Segmentation with TokenCLS
透過 Token Classification 實現 Chest CT 影像文字報告的段落區分。

## 資料格式
+ 原始標註資料
```
[@FIND@]Findings :
[@FIND@]1. A 2 cm mass in the right upper lobe , highly suspicious for primary lung cancer .
[@FIND@]2. Scattered ground-glass opacities in both lungs , possible early sign of interstitial lung disease .
[@FIND@]3. No significant mediastinal lymph node enlargement .
[@FIND@]4. Mild pleural effusion on the left side .
[@FIND@]5. No evidence of bone metastasis in the visualized portions of the thorax .
[@IMPR@]Conclusion :
[@IMPR@]A. Right upper lobe mass suggestive of lung cancer ; biopsy recommended .
[@IMPR@]B. Ground-glass opacities ; suggest follow-up CT in 3 months .
[@IMPR@]C. Mild pleural effusion ; may require thoracentesis if symptomatic .
```
+ CoNLL 標註資料

標籤的種類共 7 種，分別是 `O`、`B-S`、`B-E`、`B-FIN-S`、`B-FIN-E`、`B-IMP-S`、`B-IMP-E`。

原始標註資料經由 `1_preparing.py` 轉換成 CoNLL 標註資料。用法請參考 `run_preparing.sh` 腳本。
```
Findings B-FIN-S
: B-FIN-E
1. B-FIN-S
A O
2 O
...
cancer O
. B-FIN-E
2. B-FIN-S
Scattered O
...
```

## 模型訓練
```
python 2_train.py --experiments_path PATH_TO_EXPERIMENTS \
                  --experiment_name EXPERIMENT_NAME \
                  --run_name RUN_NAME \
                  --model_path MODEL_SAVE_PATH \
                  --script_file DATASET_SCRIPT_PATH \
                  --train_file TRAIN_DATASET_PATH \
                  --validation_file VALIDATION_DATASET_PATH \
                  [--log_file LOG_FILE] \
                  [--cache_dir CACHE_DIRECTORY] \
                  [--pretrained_model_name_or_path PRETRAINED_MODEL] \
                  [--batch_size BATCH_SIZE] \
                  [--max_length MAX_TOKEN_LENGTH] \
                  [--learning_rate LEARNING_RATE] \
                  [--weight_decay WEIGHT_DECAY] \
                  [--epochs NUMBER_OF_EPOCHS] \
                  [--warmup_ratio WARMUP_RATIO] \
                  [--accum_steps GRADIENT_ACCUMULATION_STEPS] \
                  [--max_norm MAX_GRADIENT_NORM] \
                  [--seed RANDOM_SEED]
```
### Required Arguments
+ **experiments_path** *(str)* ─ Directory to store MLflow experiment tracking data.
+ **experiment_name** *(str)* ─ Name of the MLflow experiment.
+ **run_name** *(str)* ─ Name of the MLflow run.
+ **model_path** *(str)* ─ Path to save the trained model.
+ **script_file** *(str)* ─ Path to the dataset script (in Hugging Face's datasets library format).
+ **train_file** *(str)* ─ Training dataset file path.
+ **validation_file** *(str)* ─ Validation dataset file path.

### Optional Arguments
+ **log_file** *(str, defaults to `train.log`)* ─ Log file path.
+ **cache_dir** *(str, defaults to `cache_dir`)* ─ Cache directory for Hugging Face's datasets.
+ **pretrained_model_name_or_path** *(str, defaults to `prajjwal1/bert-tiny`)* ─ Pretrained model name or path.
+ **batch_size** *(int, defaults to `16`)* ─ Training and validation batch size.
+ **max_length** *(int, defaults to `256`)* ─ Maximum token length for the model.
+ **learning_rate** *(float, defaults to `1e-4`)* ─ Learning rate for the optimizer.
+ **weight_decay** *(float, defaults to `0.0`)* ─ Weight decay for regularization.
+ **epochs** *(int, defaults to `10`)* ─ Number of training epochs.
+ **warmup_ratio** *(float, defaults to `0.0`)* ─ Ratio of warmup steps in the learning rate scheduler.
+ **accum_steps** *(int, defaults to `1`)* ─ Gradient accumulation steps for larger batch effects with limited memory.
+ **max_norm** *(float, defaults to `1.0`)* ─ Maximum gradient norm for gradient clipping.
+ **seed** *(int, defaults to `2330`)* ─ Random seed for reproducibility.

## 模型評估
```
python 3_test.py --script_file SCRIPT_PATH \
                 --test_file TEST_DATASET_PATH \
                 --pretrained_model_name_or_path MODEL_PATH_OR_NAME \
                 [--cache_dir CACHE_DIRECTORY] \
                 [--max_length MAX_TOKEN_LENGTH] \
                 [--eval_batch_size EVAL_BATCH_SIZE]
```
### Required Arguments
+ **script_file** *(str)* ─ Path to the dataset script or dataset loading script in the Hugging Face datasets library.
+ **test_file** *(str)* ─ Path to the file containing the test data.
+ **pretrained_model_name_or_path** *(str)* ─ The name or path of the pretrained model to be evaluated.

### Optional Arguments
+ **cache_dir** *(str, defaults to `cache_dir`)* ─ Cache directory for storing the datasets.
+ **max_length** *(str, defaults to `512`)* ─ Maximum length of the input sequences.
+ **eval_batch_size** *(str, defaults to `100`)* ─ Batch size for evaluation.

### Evaluation Metrics
+ Test Loss: The average loss of the model on the test dataset.
+ Test F1 (Macro): The F1 score calculated across all classes, giving equal weight to each class.
+ Test Classification Report: A detailed report showing the precision, recall, and F1 score for each class.
+ Test Duration: The time taken to complete the evaluation.

## 模型推理
```python
pipeline = TokenCLSForParagraphSegmentationPipeline("models/best_model")
text = "Findings: 1. A 2 cm mass in the right upper lobe, highly suspicious for primary lung cancer. 2. Scattered ground-glass opacities in both lungs, possible early sign of interstitial lung disease. 3. No significant mediastinal lymph node enlargement. 4. Mild pleural effusion on the left side. 5. No evidence of bone metastasis in the visualized portions of the thorax. Conclusion: A. Right upper lobe mass suggestive of lung cancer; biopsy recommended. B. Ground-glass opacities; suggest follow-up CT in 3 months. C. Mild pleural effusion; may require thoracentesis if symptomatic."
results = pipeline(text)
```
輸出結果：
```
===== Others =====


===== Findings =====
Findings:
1. A 2 cm mass in the right upper lobe, highly suspicious for primary lung cancer.
2. Scattered ground-glass opacities in both lungs, possible early sign of interstitial lung disease.
3. No significant mediastinal lymph node enlargement.
4. Mild pleural effusion on the left side.
5. No evidence of bone metastasis in the visualized portions of the thorax.

===== Impression =====
Conclusion:
A. Right upper lobe mass suggestive of lung cancer; biopsy recommended.
B. Ground-glass opacities; suggest follow-up CT in 3 months.
C. Mild pleural effusion; may require thoracentesis if symptomatic.
```

## 模型服務
透過指定的主機和端口啟動一個網絡伺服器，來部署一個用 MLflow 保存的模型。
```bash
#!/usr/bin/env sh

# Set environment variable for the tracking URL where the Model Registry resides
export MLFLOW_TRACKING_URI=http://localhost:9487

# Serve the production model from the model registry
mlflow models serve -m "models:/paragraph-segmentation/1" -h 0.0.0.0 -p 9490
```

API 使用方式請參考 `post.sh`。
