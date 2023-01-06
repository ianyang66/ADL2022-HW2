# train
accelerate launch multiple-choice/run_mc_no_trainer_test.py \
--do_train \
--do_eval \
--train_file ./temp/train.json \
--validation_file ./temp/valid.json \
--context_file ./data/context.json \
--pad_to_max_length \
--model_name_or_path hfl/chinese-roberta-wwm-ext \
--output_dir ./Rogr1linearepo3 \
# --do_debug

