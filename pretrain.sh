#!/usr/bin/env bash

export BUCKET=gs://koursaros
export TEACHER_DIR=../uncased_L-12_H-768_A-12
export STUDENT_DIR=../uncased_L-4_H-768_A-12

python3 run_pretraining.py \
  --input_file=${BUCKET}/bert_pretrain_data/*.tf_record \
  --output_dir=${BUCKET}/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=${STUDENT_DIR}/bert_config.json \
  --teacher_config_file=${TEACHER_DIR}/bert_config.json \
  --teacher_checkpoint=${BUCKET}/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --train_batch_size=128 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=10000 \
  --num_warmup_steps=100 \
  --learning_rate=2e-5 \
  --distill \
  --use_tpu \
  --tpu_name pretrain


# BERT BASE BASELINE EVAL
#***** Eval results *****
#INFO:tensorflow:  global_step = 0
#INFO:tensorflow:  loss = 2.284875
#INFO:tensorflow:  masked_lm_accuracy = 0.6143619
#INFO:tensorflow:  masked_lm_loss = 2.2041852
#INFO:tensorflow:  next_sentence_accuracy = 0.98125
#INFO:tensorflow:  next_sentence_loss = 0.07830824
python3 run_pretraining.py \
  --input_file=${BUCKET}/bert_pretrain_data/*.tf_record \
  --output_dir=${BUCKET}/baseline_pretrain \
  --do_eval=True \
  --bert_config_file=${TEACHER_DIR}/bert_config.json \
  --train_batch_size=128 \
  --max_seq_length=128 \
  --use_tpu \
  --tpu_name pretrain \
  --init_checkpoint ${BUCKET}/uncased_L-12_H-768_A-12/bert_model.ckpt

# SANITY CHECK EVAL TEACHER. Should be the same, e.g.
# ***** Eval results *****
#INFO:tensorflow:  global_step = 0
#INFO:tensorflow:  loss = 53.61523
#INFO:tensorflow:  masked_lm_accuracy = 0.6143619
#INFO:tensorflow:  masked_lm_loss = 2.2041852
#INFO:tensorflow:  next_sentence_accuracy = 0.98125
#INFO:tensorflow:  next_sentence_loss = 0.07830824

python3 run_pretraining.py   \
--input_file=${BUCKET}/bert_pretrain_data/*.tf_record   \
--output_dir=${BUCKET}/tmp   --do_train=False   --do_eval=True  \
 --bert_config_file=${STUDENT_DIR}/bert_config.json   \
 --teacher_config_file=${TEACHER_DIR}/bert_config.json \
 --teacher_checkpoint=${BUCKET}/uncased_L-12_H-768_A-12/bert_model.ckpt   \
 --train_batch_size=128   --max_seq_length=128   \
 --max_predictions_per_seq=20   --num_train_steps=100000   \
 --num_warmup_steps=1000   --learning_rate=2e-5   --distill   \
 --use_tpu   --tpu_name pretrain --eval_teacher --pred_distill

GLUE_DIR=../glue_data
MODEL=preds_distill/model.ckpt-50000
# GET MNLI for student
python3 run_classifier.py   --task_name=MNLI  \
 --do_train=false   --do_eval=true   \
 --data_dir=$GLUE_DIR/MNLI   --vocab_file=${TEACHER_DIR}/vocab.txt   \
 --bert_config_file=$STUDENT_DIR/bert_config.json   \
 --init_checkpoint=${BUCKET}/${MODEL}   \
 --max_seq_length=128   --train_batch_size=32   --learning_rate=2e-5   \
 --num_train_epochs=3.0   --output_dir=gs://koursaros/tmp/mnli_output/ \
 --use_tpu --tpu_name pretrain