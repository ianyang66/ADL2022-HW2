mkdir ./best-cs
mkdir ./best-qa

# context-seleciton
wget https://www.dropbox.com/s/e0rll5din2rhbqz/config.json?dl=1 -O ./best-cs/config.json
wget -t 3 -T 30 -c https://www.dropbox.com/s/yutaskuibh8k211/pytorch_model.bin?dl=1 -O ./best-cs/pytorch_model.bin
wget -t 3 -T 30 -c https://www.dropbox.com/s/yutaskuibh8k211/pytorch_model.bin?dl=1 -O ./best-cs/pytorch_model.bin
wget -t 3 -T 30 -c https://www.dropbox.com/s/yutaskuibh8k211/pytorch_model.bin?dl=1 -O ./best-cs/pytorch_model.bin
wget https://www.dropbox.com/s/pz9bvwoy1lghb78/special_tokens_map.json?dl=1 -O ./best-cs/special_tokens_map.json
wget https://www.dropbox.com/s/7u0i940oacfn0zk/tokenizer.json?dl=1 -O ./best-cs/tokenizer.json
wget https://www.dropbox.com/s/3wee2g64va92i85/tokenizer_config.json?dl=1 -O ./best-cs/tokenizer_config.json
wget https://www.dropbox.com/s/jjwi8y22f4fyzfb/vocab.txt?dl=1 -O ./best-cs/vocab.txt

# question-answering
wget https://www.dropbox.com/s/c94ost7agqq537d/config.json?dl=1 -O ./best-qa/config.json
wget -t 3 -T 30 -c https://www.dropbox.com/s/23u354rly5reoeu/pytorch_model.bin?dl=1 -O ./best-qa/pytorch_model.bin
wget -t 3 -T 30 -c https://www.dropbox.com/s/23u354rly5reoeu/pytorch_model.bin?dl=1 -O ./best-qa/pytorch_model.bin
wget -t 3 -T 30 -c https://www.dropbox.com/s/23u354rly5reoeu/pytorch_model.bin?dl=1 -O ./best-qa/pytorch_model.bin
wget https://www.dropbox.com/s/31d7e0fv1fiatdz/special_tokens_map.json?dl=1 -O ./best-qa/special_tokens_map.json
wget https://www.dropbox.com/s/olt0noxinnmjslp/tokenizer.json?dl=1 -O ./best-qa/tokenizer.json
wget https://www.dropbox.com/s/lf1dysyerf4ybtd/tokenizer_config.json?dl=1 -O ./best-qa/tokenizer_config.json
wget https://www.dropbox.com/s/lubde69a4yd0952/vocab.txt?dl=1 -O ./best-qa/vocab.txt