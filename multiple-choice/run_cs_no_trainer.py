import argparse
from transformers import SchedulerType

import logging
import math
import os
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional, Union
import json
import numpy as np

import datasets
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

#import evaluate
import transformers
from accelerate import Accelerator

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

# from transformers.utils import get_full_repo_name
from transformers.file_utils import PaddingStrategy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task")
    # Data
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default='./temp/mc_train.json', help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default='./temp/mc_valid.json', help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default='./temp/mc_test.json', help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--output_file", type=str, default='./temp/QA_test.json', help="A csv or a json file containing the predicted test data."
    )
    parser.add_argument(
        "--context_file", type=str, default='./data/context.json', help="A csv or a json file containing the context data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="If passed, go through the trian process",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="If passed, go through the validate process",
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
        help="If passed, go through the test process",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="fnlp/elasticbert-chinese-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use."
    )
    parser.add_argument("--num_train_epochs",
        type=int,
        default=6,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument("--seed",
        type=int, 
        default=50,
        help="A seed for reproducible training."
    )

    parser.add_argument(
        "--do_debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )

    args = parser.parse_args()
    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)

        return batch


def main():
    args = parse_args()
    print("="*100 + "\n\n")
    print("Args:\n")
    print(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    logger = logging.getLogger(__name__)
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    print("\n\n" + "="*20 + "\n\n")
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()


   
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # Load dataset from the hub.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None and args.do_train:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None and args.do_eval:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if args.test_file is not None and args.do_test:
            data_files["test"] = args.test_file
            extension = args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field='data')

    # Take 100 data to use, and Comment out them during formal training or prediction.
    if args.do_debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))


    # load context list
    context_dir = Path(args.context_file)
    context_list = json.loads(context_dir.read_text())

    # download model & vocab
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForMultipleChoice.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )



    # Check max sequence length 
    # Make sure that the maximum sequence length does not exceed the maximum sequence length provided by the model
    if args.max_length is None:
        max_length = tokenizer.model_max_length
        if max_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_length = 1024
    else:
        if args.max_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_length = min(args.max_length, tokenizer.model_max_length)

    # ## Preprocessing the datasets
    # Preprocessing the datasets.
    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    # ending_names = [f"ending{i}" for i in range(4)]
    question_name = "question"
    paragraphs_name = "paragraphs"
    relevant_name = "relevant"
    def preprocess_function(examples):
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        # question: previous context
        # [[question_0, question_0, question_0, question_0], [question_1, question_1, question_1, question_1], ... [question_N, question_N, question_N, question_N]]
        first_sentences = [[question] * 4 for question in examples[question_name]]

        # Grab all paragraph context possible for each question.
        paragraphsIdx = examples[paragraphs_name]

        # paragraph: after context
        # [[paragraphs0_0, paragraphs1_0, paragraphs2_0, paragraphs3_0], [paragraphs0_1, paragraphs1_1, paragraphs2_1, paragraphs3_1] ...]
        second_sentences = [
            [f"{context_list[idx]}" for idx in selections] for i, selections in enumerate(paragraphsIdx)  # i: 第 i 個 data， 每個 data 有4個 paragraph selection 
        ]

        # Flatten everything
        # [question_0, question_0, question_0, question_0, question_1, question_1, question_1, question_1, ... question_N, question_N, question_N, question_N]
        first_sentences = sum(first_sentences, [])

        # [paragraphs0_0, paragraphs1_0, paragraphs2_0, paragraphs3_0, paragraphs0_1, paragraphs1_1, paragraphs2_1, paragraphs3_1 ...]
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=args.max_length,
            padding="max_length" if args.pad_to_max_length else False,
        )
        # Tokenize
        # {
        #  [token_dataId_options]: [CLS] Question [SEP] Paragraph (option i) [SEP] (list to idx)  
        # 'input_ids': [[tokens_0_0], [tokens_0_1], [tokens_0_2], [tokens_0_3], ... [tokens_N_0], [tokens_N_1], [tokens_N_2], [tokens_N_3]]} => (N-1) * 4
        # 'token_type_ids: [[tokens_0_0], [tokens_0_1], [tokens_0_2], [tokens_0_3], ... [tokens_N_0], [tokens_N_1], [tokens_N_2], [tokens_N_3]]} => (N-1) * 4
        # 'attention_mask': [[tokens_0_0], [tokens_0_1], [tokens_0_2], [tokens_0_3], ... [tokens_N_0], [tokens_N_1], [tokens_N_2], [tokens_N_3]]} => (N-1) * 4
        # }

        # Un-flatten
        # input_ids': [[[tokens_0_0], [tokens_0_1], [tokens_0_2], [tokens_0_3]], ... [[tokens_N_0], [tokens_N_1], [tokens_N_2], [tokens_N_3]]]} => (N-1)
        encoded_examples = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

        # encoded_examples join to label
        if relevant_name in examples.keys():
            labels = examples[relevant_name]
            encoded_examples['label'] = [selections.index(labels[i]) for i, selections in enumerate(paragraphsIdx)]
        else:
            # for test data
            encoded_examples['label'] = [0 for i, selections in enumerate(paragraphsIdx)]
        return encoded_examples

    # ## Preprocess train dataset / valid dataset
    # do train
    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        raw_train_dataset = raw_datasets["train"]

        with accelerator.main_process_first():
            train_dataset = raw_train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names
            )
    # do validate
    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        raw_eval_dataset = raw_datasets["validation"]
        with accelerator.main_process_first():
            eval_dataset = raw_eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["validation"].column_names
            )

    # do test
    if args.do_test:
        if "test" not in raw_datasets:
            raise ValueError("--do_test requires a test dataset")
        raw_test_dataset = raw_datasets["test"]
        with accelerator.main_process_first():
            test_dataset = raw_test_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["test"].column_names
            )

    if args.do_train:
        print("\n\n" + "="*20 + "\n\n")
        print("raw train_dataset's keys: \n")
        print(list(raw_train_dataset[0].keys()))
        print("train_dataset's keys: \n")
        print(list(train_dataset[0].keys()))
    if args.do_eval:
        print("\n\n" + "="*20 + "\n\n")
        print("raw eval_dataset's keys: \n")
        print(list(raw_eval_dataset[0].keys()))
        print("eval_dataset's keys: \n")
        print(list(eval_dataset[0].keys()))
    if args.do_test:
        print("\n\n" + "="*20 + "\n\n")
        print("raw test_dataset's keys: \n")
        print(list(raw_test_dataset[0].keys()))
        print("test_dataset's keys: \n")
        print(list(test_dataset[0].keys()))

    # DataLoaders
    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    if args.do_train:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )

    if args.do_eval:
        eval_dataloader = DataLoader(
            eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
    if args.do_test:
        test_dataloader = DataLoader(
            test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )


    # ## Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # ## device
    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    if args.do_train and args.do_eval:
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )
    if args.do_test:
        model, optimizer, test_dataloader = accelerator.prepare(model, optimizer, test_dataloader)

    # ## learning rate Scheduler
    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    if args.do_train:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    
        

    # ## Train
    if args.do_train and args.do_eval:   
        # Metrics
        #metric = evaluate.load("accuracy")
        metric = load_metric("accuracy")
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("**** Running training ****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        print("\n\n" + "="*20 + "\n\n")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch_data in enumerate(train_dataloader):
                outputs = model(**batch_data)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break

            model.eval()
            eval_Accuracy = []
            for step, batch_data in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch_data)
                predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch_data["labels"]),
                )
            eval_metric = metric.compute()
            logger.info(f"Valid Accuracy: {eval_metric}")
            eval_Accuracy.append(eval_metric["accuracy"])
            
            # eval_metric = metric.compute()
            accelerator.print(f"\nepoch {epoch}: {eval_metric}\n")

            # if args.push_to_hub and epoch < args.num_train_epochs - 1:
            if epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
            

        if args.output_dir is not None:
            np.save(f'{args.output_dir}/eval_accuracy', np.array(eval_Accuracy))
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)

    # ## Predict
    if args.do_test:
        output_json = {"data":[]}
        predictions_idx = []
        model.eval()
        print("\n....Predict....")
        for step, batch_data in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                outputs = model(**batch_data)
            predictions = outputs.logits.argmax(dim=-1)
            predictions = predictions.cpu().tolist()
            for prediction_idx in predictions:
                predictions_idx.append(prediction_idx)
        
        print("\n....Write test QA json file....")
        for i, pred in enumerate(tqdm(predictions_idx)):
            context_id = raw_test_dataset["paragraphs"][i][pred]
            data = {
                "id": raw_test_dataset["id"][i],
                "question": raw_test_dataset["question"][i],
                "context": context_list[context_id]
            }
            output_json["data"].append(data)
        json.dump(output_json, open(args.output_file, 'w'), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
