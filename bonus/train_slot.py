import time
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from dataset import SeqSlotDataset
from transformers import AdamW, BertForTokenClassification, BertTokenizerFast

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    torch.manual_seed(1)
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased")
    datasets: Dict[str, SeqSlotDataset] = {
        split: SeqSlotDataset(split_data, tag2idx, args.max_len, tokenizer, "train")
        for split, split_data in data.items()
    }
    # TODO: create DataLoader for train / dev datasets
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) 
        for split, split_dataset in datasets.items()
    }

    # TODO: init model and move model to target device(cpu / gpu)
    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=10)
    device = args.device

    try:
        ckpt = torch.load("./bonus/ckpt/slot/model.ckpt")
        model.load_state_dict(ckpt)
    except:
        print("Can't load model!")
    batch_size = args.batch_size
    # TODO: init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)

    model.to(device)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    epoch_times = []
    best_acc = 0.0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_loss, train_acc, val_loss, val_acc = 0.0, 0.0, 0.0, 0.0
        start_time = time.perf_counter()
        tr = tqdm(dataloaders[TRAIN])
        for i, d in enumerate(tr):
            data = [torch.tensor(j).to(device) for j in d[1:]]
            optimizer.zero_grad()
            out = model(input_ids = data[0], token_type_ids=data[1],attention_mask=data[2],labels=data[3])
            _, train_pred = torch.max(out.logits, 2)
            loss = out.loss
            loss.backward()
            optimizer.step()

            for j, label in enumerate(data[3]):
                if 9 in train_pred[j][1:].tolist():
                    end_index = train_pred[j][1:].cpu().tolist().index(9) + 1
                else:
                    end_index = len(train_pred[j].tolist())
                data_end_index = data[3][j][1:].cpu().tolist().index(9) + 1
                train_acc += (train_pred[j][1:end_index].cpu().tolist() == data[3][j][1:data_end_index].cpu().tolist())
            train_loss += loss.item()
            tr.clear()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        with torch.no_grad():
            #h = model.init_hidden(batch_size, device)
            model.eval()
            for i, dev_d in enumerate(tqdm(dataloaders[DEV])):
                dev_data = [torch.tensor(j).to(device) for j in dev_d[1:]]
                out = model(input_ids = dev_data[0], token_type_ids=dev_data[1],attention_mask=dev_data[2],labels=dev_data[3])

                loss = out.loss 
                _, val_pred = torch.max(out.logits, 2)
                for j, label in enumerate(dev_data[3]):
                    end_index = val_pred[j][1:].cpu().tolist().index(9) + 1
                    data_end_index = dev_data[3][j][1:].cpu().tolist().index(9) + 1
                    val_acc += (val_pred[j][1:end_index].cpu().tolist() == dev_data[3][j][1:data_end_index].cpu().tolist())
                val_loss += loss.item()
            
            print(f"Epoch {epoch + 1}: Train Acc: {train_acc / len(dataloaders[TRAIN].dataset)}, Train Loss: {train_loss / len(dataloaders[TRAIN])}, Val Acc: {val_acc / len(dataloaders[DEV].dataset)}, Val Loss: {val_loss / len(dataloaders[DEV])}")
            ckp_dir = "./bonus/ckpt/slot/"
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
    test_dataset = SeqSlotDataset(test_data, tag2idx, args.max_len, tokenizer, "test")
    # TODO: create DataLoader for test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle = False, batch_size = 150, collate_fn = test_dataset.collate_fn)
    model.eval()
    model.load_state_dict(torch.load("./bonus/ckpt/slot/best-model.ckpt"))
    # load weights into model

    # TODO: predict dataset
    preds = []
    with open(args.output_csv, "w") as fp:
        fp.write("id,tags\n")
        with torch.no_grad():
            for i, d in enumerate(tqdm(test_loader)):
                ids = d[0]
                data = [torch.tensor(j).to(device) for j in d[1:]]
                out = model(input_ids = data[0], token_type_ids=data[1],attention_mask=data[2])
                _, pred = torch.max(out.logits, 2)
                for j, p in enumerate(pred):
                    data_end_index = p[1:].cpu().tolist().index(9) + 1
                    fp.write(f"{ids[j]},{' '.join(list(map(lambda x:test_dataset.idx2label(x), list(filter(lambda x: (x != 9), p[1:data_end_index].tolist())))))}\n")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./bonus/data_hw1/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./bonus/cache_hw1/slot",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./bonus/ckpt/slot/",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Directory to save the model file.",
        default="./data_hw1/slot/test.json",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        help="Directory to save the model file.",
        default="slot_pred.csv",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
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
