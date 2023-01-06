import json
import time
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from dataset import SeqClsDataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizerFast

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, intent2idx, args.max_len, tokenizer, "train")
        for split, split_data in data.items()
    }
    # TODO: create DataLoader for train / dev datasets
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) 
        for split, split_dataset in datasets.items()
    }

    # TODO: init model and move model to target device(cpu / gpu)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=150)
    device = args.device
    try:
        ckpt = torch.load("./bonus/ckpt/intent/best-model.ckpt")
        model.load_state_dict(ckpt)
    except:
        pass
    batch_size = args.batch_size
    # TODO: init optimizer
    optimizer = AdamW(model.parameters(), lr = args.lr)

    model.to(device)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    epoch_times = []
    best_acc = 0.0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_loss, train_acc, val_loss, val_acc = 0.0, 0.0, 0.0, 0.0
        start_time = time.perf_counter()

        for i, data in enumerate(tqdm(dataloaders[TRAIN])):
            data = [torch.tensor(j).to(device) for j in data[1:]] 
            optimizer.zero_grad()
            out = model(input_ids = data[0], token_type_ids = data[1], attention_mask=data[2], labels=data[3])
            loss = out.loss
            _, train_pred = torch.max(out.logits, 1)
            loss.backward()
            optimizer.step()

            train_acc += (train_pred.cpu() == data[3].cpu()).sum().item()
            train_loss += loss.item()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        with torch.no_grad():
            #h = model.init_hidden(batch_size, device)
            model.eval()
            for i, dev_data in enumerate(tqdm(dataloaders[DEV])):
                dev_data = [torch.tensor(j).to(device) for j in dev_data[1:]] 
                out = model(input_ids = dev_data[0], token_type_ids = dev_data[1], attention_mask=dev_data[2], labels=dev_data[3])

                loss = out.loss
                _, val_pred = torch.max(out.logits, 1)
                val_acc += (val_pred.cpu() == dev_data[3].cpu()).sum().item()
                val_loss += loss.item()
            
            print(f"Epoch {epoch + 1}: Train Acc: {train_acc / len(dataloaders[TRAIN].dataset)}, Train Loss: {train_loss / len(dataloaders[TRAIN])}, Val Acc: {val_acc / len(dataloaders[DEV].dataset)}, Val Loss: {val_loss / len(dataloaders[DEV])}")
            ckp_dir = "./bonus/ckpt/intent/"
            if val_acc >= best_acc:
                best_acc = val_acc
                ckp_path = ckp_dir + '{}-model.ckpt'.format(epoch + 1)
                best_ckp_path = ckp_dir + 'best-model.ckpt'.format(epoch + 1)
                torch.save(model.state_dict(), ckp_path)
                torch.save(model.state_dict(), best_ckp_path)
                print(f"Save model with acc {val_acc / len(dataloaders[DEV].dataset)}")

            current_time = time.perf_counter()
        epoch_times.append(current_time-start_time)
        print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))     

    # TODO: Inference on test set
    test_data = None
    with open(args.test_file, "r") as fp:
        test_data = json.load(fp)
    test_dataset = SeqClsDataset(test_data, intent2idx, args.max_len, tokenizer, "test")
    # Create DataLoader for test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    model.eval()

    # TODO: predict dataset
    preds = []
    with open(args.output_csv, "w") as fp:
        fp.write("id,intent\n")
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader)):
                ids=data[0]
                data = [torch.tensor(j).to(device) for j in data[1:]] 
                out = model(input_ids = data[0], token_type_ids = data[1], attention_mask=data[2])
                _, pred = torch.max(out.logits, 1)
                for j, p in enumerate(pred):
                    fp.write(f"{ids[j]},{test_dataset.idx2label(p.item())}\n")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./bonus/data_hw1/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./bonus/cache_hw1/intent",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./bonus/ckpt/intent/",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Directory to save the model file.",
        default="./bonus/data_hw1/intent/test.json",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        help="Directory to save the model file.",
        default="intent_pred.csv",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # training
    parser.add_argument(
            "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=8)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
