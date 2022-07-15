#!/bin/bash
# RUns representative pretrained model evaluations on SQuAD-v2.0 using example script run_qa

python run_qa.py --model_name_or_path twmkn9/distilbert-base-uncased-squad2 \
                --dataset_name squad_v2 --version_2_with_negative --do_eval \
                --max_seq_length 384 --doc_stride 128 \
                --output_dir save/squad_v2/twmkn9/distilbert-base-uncased-squad2/101 \
                --per_device_eval_batch_size 48 --evaluation_strategy epoch \
                --n_best_size 10 --logging_steps 10 --preprocessing_num_workers 6 

python run_qa.py --model_name_or_path navteca/roberta-base-squad2 \
                --dataset_name squad_v2 --version_2_with_negative --do_eval \
                --max_seq_length 384 --doc_stride 128 \
                --output_dir save/squad_v2/navteca/roberta-base-squad2/101 \
                --per_device_eval_batch_size 48 --evaluation_strategy epoch \
                --n_best_size 10 --logging_steps 10 --preprocessing_num_workers 6 

python run_qa.py --model_name_or_path navteca/electra-base-squad2 \
                --dataset_name squad_v2 --version_2_with_negative --do_eval \
                --max_seq_length 384 --doc_stride 128 \
                --output_dir save/squad_v2/navteca/electra-base-squad2/101 \
                --per_device_eval_batch_size 48 --evaluation_strategy epoch \
                --n_best_size 10 --logging_steps 10 --preprocessing_num_workers 6 

python run_qa.py --model_name_or_path deepset/bert-large-uncased-whole-word-masking-squad2 \
                --dataset_name squad_v2 --version_2_with_negative --do_eval \
                --max_seq_length 384 --doc_stride 128 \
                --output_dir save/squad_v2/deepset/bert-large-uncased-whole-word-masking-squad2/101 \
                --per_device_eval_batch_size 48 --evaluation_strategy epoch \
                --n_best_size 10 --logging_steps 10 --preprocessing_num_workers 6 

python run_qa.py --model_name_or_path phiyodr/bert-base-finetuned-squad2 \
                --dataset_name squad_v2 --version_2_with_negative --do_eval \
                --max_seq_length 384 --doc_stride 128 \
                --output_dir save/squad_v2/phiyodr/bert-base-finetuned-squad2/101 \
                --per_device_eval_batch_size 48 --evaluation_strategy epoch \
                --n_best_size 10 --logging_steps 10 --preprocessing_num_workers 6 

python run_qa.py --model_name_or_path mrm8488/longformer-base-4096-finetuned-squadv2 \
                --dataset_name squad_v2 --version_2_with_negative --do_eval \
                --max_seq_length 1024 --doc_stride 128 --attention_window 384 \
                --output_dir save/squad_v2/mrm8488/longformer-base-4096-finetuned-squadv2/101 \
                --per_device_eval_batch_size 48 --evaluation_strategy epoch \
                --n_best_size 10 --logging_steps 10 --preprocessing_num_workers 6 

