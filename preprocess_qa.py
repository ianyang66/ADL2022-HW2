import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
import os
import sys

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="Directory to the dataset.",
        default="./data/valid.json",
    )
    parser.add_argument(
        "--context_dir",
        help="Directory to the dataset.",
        default="./data/context.json",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save the processed file.",
        default="./data/qa_valid.json",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Data Path
    data_dir = Path(args.data_dir)
    context_dir = Path(args.context_dir)
    # make directory and json file
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    # load data from data path
    list_data = json.loads(data_dir.read_text())
    list_context = json.loads(context_dir.read_text())

    # Preprocess
    new_list_data = []
    for data in list_data:
        data["answers"] = {"answer_start":[data["answer"]["start"]], "text":[data["answer"]["text"]]}
        data["context"] = list_context[data["relevant"]]
        data.pop("answer")
        new_list_data.append(data)

    # List to Dict, and storage
    json.dump({"data": new_list_data}, open(args.output_dir, 'w'),indent=2, ensure_ascii=False)  # ensure_ascii=False 包含中文