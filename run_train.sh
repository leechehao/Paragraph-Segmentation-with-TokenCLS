python 2_train.py --experiments_path /home/mlflow-experiments \
                  --experiment_name Paragraph-Segmentation \
                  --run_name deberta-v3-large-2 \
                  --model_path models/best_model \
                  --script_file paragraph_segmentation_dataset.py \
                  --train_file program_data/train/train.conll \
                  --validation_file program_data/validation/valid.conll \
                  --log_file train.log \
                  --cache_dir cache_dir \
                  --pretrained_model_name_or_path microsoft/deberta-v3-large \
                  --batch_size 4 \
                  --max_length 256 \
                  --learning_rate 5e-5 \
                  --weight_decay 0.0 \
                  --epochs 10 \
                  --warmup_ratio 0.0 \
                  --accum_steps 1 \
                  --max_norm 1.0 \
                  --seed 2330



