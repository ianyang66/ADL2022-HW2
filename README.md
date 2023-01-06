# Homework 2 - NTU ADL 2022 FALL

## Reproduce testing process

### Use run.sh to predict testing data

```shell
bash download.sh
bash run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```

example:

```shell
bash download.sh
bash run.sh ./data/context.json ./data/test.json ./data/test_submission.csv
```

<br>

## **Reproduce training process**

### **Install Spacy**

You should install spacy and zh_core_web_md, or you can not use my function to compute exact match and f1 score in question answering.

```shell
bash spacy&dependency_install.sh
```

### **Context-Selection Data Preprocessing**

We need to prepare proper data format for multiple-choice and question-answering (store preprocessed data to ./data/cs_train.json, ./data/cs_valid.json, ./data/qa_train.json, ./data/qa_valid.json)

```shell
bash preprocess_train.sh /path/to/train.json /path/to/valid.json /path/to/context.json
```

example:

```shell
bash preprocess_train.sh ./data/cs_train.json ./data/cs_valid.json ./data/context.json
```

### **Context-Selection train**

```shell
bash train_cs.sh  /path/to/preprocessed_train.json /path/to/preprocessed_valid.json  /path/to/context.json
```

example:

```shell
bash train_cs.sh ./data/cs_train.json ./data/cs_valid.json ./data/context.json
```

### **Question-Answering train**

```shell
bash train_qa.sh /path/to/preprocessed_train.json /path/to/preprocessed_valid.json  /path/to/context.json
```

example:

```shell
bash train_qa.sh ./data/qa_train.json ./data/qa_valid.json ./data/context.json
```

## Experiment Result

| Model Name (CS)                   | Model Name (QA)             | Gradient Accumulation Steps (MC) | Gradient Accumulation Steps (QA) | Num Train Epochs (CS) | Num Train Epochs (QA) | lr scheduler type (CS) | lr scheduler type (QA) | Public Acc.<br/>on Kaggle |
| --------------------------------- | --------------------------- | -------------------------------- | -------------------------------- | --------------------- | --------------------- | ---------------------- | ---------------------- | ------------------------- |
| bert-base-chinese                 | bert-base-chinese           | 2                                | 2                                | 3                     | 1                     | linear                 | linear                 | 0.74231                   |
| hfl/chinese-roberta-wwm-ext       | hfl/chinese-roberta-wwm-ext | 2                                | 2                                | 3                     | 3                     | linear                 | linear                 | 0.76401                   |
| hfl/chinese-roberta-wwm-ext       | hfl/chinese-roberta-wwm-ext | 2                                | 2                                | 3                     | 3                     | linear                 | cosine                 | 0.783                     |
| hfl/chinese-roberta-wwm-ext       | hfl/chinese-roberta-wwm-ext | 2                                | 2                                | 3                     | 20                    | linear                 | cosine                 | 0.77215                   |
| hfl/chinese-roberta-wwm-ext       | hfl/chinese-roberta-wwm-ext | 2                                | 2                                | 3                     | 20                    | linear                 | linear                 | 0.7613                    |
| hfl/chinese-roberta-wwm-ext       | hfl/chinese-pert-base       | 2                                | 8                                | 3                     | 10                    | linear                 | linear                 | 0.76672                   |
| hfl/chinese-roberta-wwm-ext       | hfl/chinese-lert-base       | 2                                | 8                                | 3                     | 10                    | linear                 | cosine                 | 0.79023                   |
| hfl/chinese-roberta-wwm-ext       | hfl/chinese-lert-base       | 2                                | 8                                | 3                     | 10                    | linear                 | linear                 | 0.79023                   |
| hfl/chinese-roberta-wwm-ext       | hfl/chinese-lert-base       | 2                                | 8                                | 3                     | 15                    | linear                 | linear                 | 0.79023                   |
| hfl/chinese-roberta-wwm-ext       | hfl/chinese-lert-base       | 2                                | 8                                | 3                     | 20                    | linear                 | linear                 | 0.78119                   |
| hfl/chinese-roberta-wwm-ext       | hfl/chinese-lert-base       | 2                                | 8                                | 5                     | 10                    | linear                 | linear                 | 0.78661                   |
| hfl/chinese-roberta-wwm-ext       | hfl/chinese-lert-large      | 2                                | 8                                | 3                     | 6                     | linear                 | linear                 | 0.79475                   |
| hfl/chinese-roberta-wwm-ext-large | hfl/chinese-lert-large      | 8                                | 8                                | 2                     | 6                     | linear                 | linear                 | 0.79385                   |
| hfl/chinese-macbert-base          | hfl/chinese-lert-large      | 8                                | 8                                | 6                     | 6                     | linear                 | linear                 | 0.79927                   |
| hfl/chinese-macbert-large         | hfl/chinese-lert-large      | 8                                | 8                                | 2                     | 6                     | linear                 | linear                 | **0.80831**               |
| hfl/chinese-macbert-large         | hfl/chinese-lert-large      | 8                                | 64                               | 2                     | 10                    | linear                 | linear                 | 0.80289                   |

## Q5. bonus

### **Reproduce training&prediction process**

Use train_intent.sh to train bert model, and it will also predict the testing data.

```shell
bash bonus/train_intent.sh /path/to/data_directory_contain_train.json&eval.json /path/to/test.json /path/to/prediction.csv
```

example:

```shell
bash bonus/train_intent.sh data_hw1/intent data_hw1/intent/test.json intent_pred.csv
```



Use train_slot.sh to train bert model, and it will also predict the testing data.

```shell
bash bonus/train_slot.sh /path/to/data_directory_contain_train.json&eval.json /path/to/test.json /path/to/prediction.csv
```

example:

```shell
bash bonus/train_slot.sh data_hw1/slot data_hw1/slot/test.json slot_pred.csv
```

