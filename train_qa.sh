# train
accelerate launch question-answering/run_qa_no_trainer.py \
--do_train \
--do_eval \
--train_file ./temp/QA_train.json \
--validation_file ./temp/QA_valid.json \
--test_file QA_test_TA_ro_cosine.json \
--pad_to_max_length \
--model_name_or_path hfl/chinese-lert-base \
--output_dir ./qaLert_epo10 \

# --max_train_samples 100 \
# --max_eval_samples 100


# --model_name_or_path bert-base-chinese \
