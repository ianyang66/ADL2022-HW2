accelerate launch question-answering/run_qa_no_trainer.py \
--do_train \
--do_eval \
--train_file ./data/QA_train.json \
--validation_file ./data/QA_valid.json \
--pad_to_max_length \
--model_name_or_path ./QAlertlargegr64linearepo6 \
--output_dir ./data \

# --max_predict_samples 100
# --do_debug