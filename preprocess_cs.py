import json
from argparse import ArgumentParser, Namespace
import os


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="Directory to the dataset.",
        default="./data/test.json",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save the processed file.",
        default="./data/cs_test.json",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # open and read data
    with open(args.data_dir, 'r') as f:
        data = json.load(f)
    #　make directory and json file
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    # List to Dict, and storage
    json.dump({'data': data}, open(args.output_dir, 'w',encoding='utf-8'), indent=2, ensure_ascii=False)