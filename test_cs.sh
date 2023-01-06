# test
accelerate launch multiple-choice/run_cs_no_trainer.py \
--do_test \
--test_file ./temp/cs_test.json \
--output_file ./temp/QA_test_macbertlarge_gr8_epo2.json \
--context_file ./data/context.json \
--pad_to_max_length \
--model_name_or_path ./CSmacbertlarge_gr8_epo2
