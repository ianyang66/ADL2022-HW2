mkdir ./data
# preprocess test.json  
python preprocess_cs.py \
--data_dir $2 \
--output_dir ./data/cs_test.json

# context-selection prediction 
python multiple-choice/run_cs_no_trainer.py \
--do_test \
--test_file ./data/cs_test.json \
--output_file ./data/qa_test.json \
--context_file $1 \
--pad_to_max_length \
--model_name_or_path ./best-cs

# question-answering prediction
python question-answering/run_qa_no_trainer.py \
--do_predict \
--test_file ./data/qa_test.json \
--output_csv $3 \
--pad_to_max_length \
--model_name_or_path ./best-qa