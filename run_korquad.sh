#!/bin/bash

if [ $1 = "train" ];then
	echo "train";
	sleep 3;

	python3 src/train.py \
	  --train_file=$2 \
	  --output_dir=$3 \
	  --checkpoint=$4 \
	  --model_config=$BERT_BASE_DIR/bert_config.json \
	  --vocab=$BERT_BASE_DIR/vocab.txt \
	  --max_seq_length=512 \
	  --max_query_length=96 \
	  --max_answer_length=30 \
	  --doc_stride=128 \
	  --train_batch_size=16 \
	  --learning_rate=5e-5 \
	  --num_train_epochs=4.0 \
	  --grad_noise=$5 \
	  --gs_noise=$6 \
	  --seed=42;
else
	echo "test";
	sleep 3;

	python3 src/eval_ensemble.py \
	  --predict_file=$2 \
	  --output_dir=$3 \
	  --ensemble=$4 \
	  --checkpoint=$5 \
	  --model_config=$BERT_BASE_DIR/bert_config.json \
	  --vocab=$BERT_BASE_DIR/vocab.txt \
	  --max_seq_length=512 \
	  --max_query_length=96 \
	  --max_answer_length=30 \
	  --doc_stride=128 \
	  --batch_size=16 \
	  --n_best_size=20 \
	  --seed=42;
fi

