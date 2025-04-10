"""This script processes full MRPC data"""

import argparse
import os
import numpy as np

def get_label(task, line):
    return line.strip().split('\t')[0]

def load_mrpc_data(data_dir):
    """Loads MRPC train and dev data."""
    dataset = {}
    dirname = os.path.join(data_dir, "MRPC")
    for split in ["train", "dev"]:
        filename = os.path.join(dirname, f"{split}.tsv")
        with open(filename, "r") as f:
            lines = f.readlines()
        dataset[split] = lines
    return dataset

def split_header(lines):
    """Splits header from data for MRPC."""
    return lines[0:1], lines[1:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/original",
                      help="Path to original MRPC data")
    parser.add_argument("--output_dir", type=str, default="data/full",
                      help="Output path")

    args = parser.parse_args()

    # Setup directories
    args.output_dir = os.path.join(args.output_dir)
    task = "MRPC"
    k = "full"
    seed = 42
    np.random.seed(seed)

    print(f"Processing full MRPC dataset")
    dataset = load_mrpc_data(args.data_dir)

    # Process train data
    train_header, train_lines = split_header(dataset["train"])
    np.random.shuffle(train_lines)

    # Split into labels and exclude dev examples first
    label_list = {}
    for line in train_lines:
        label = get_label(task, line)
        if label not in label_list:
            label_list[label] = []
        label_list[label].append(line)

    # Create dev set (10% of each class, excluded from train)
    dev_lines = []
    for label in label_list:
        lines = label_list[label]
        dev_size = int(len(lines) * 0.1)
        np.random.shuffle(lines)
        dev_lines.extend(lines[:dev_size])
        label_list[label] = lines[dev_size:]  # Remove dev examples

    # Prepare output dir
    setting_dir = os.path.join(args.output_dir, task)
    os.makedirs(setting_dir, exist_ok=True)

    # Save test set (original dev set)
    test_header, test_lines = split_header(dataset["dev"])
    if len(test_lines) > 1000:
        np.random.shuffle(test_lines)
        test_lines = test_lines[:1000]
    with open(os.path.join(setting_dir, "test.tsv"), "w") as f:
        f.writelines(test_header + test_lines)

    # Save original train set (dev excluded)
    train_lines = []
    for label in label_list:
        train_lines.extend(label_list[label])
    np.random.shuffle(train_lines)
    with open(os.path.join(setting_dir, "train.tsv"), "w") as f:
        f.writelines(train_header + train_lines)

    # Save dev set
    np.random.shuffle(dev_lines)    
    with open(os.path.join(setting_dir, "dev.tsv"), "w") as f:
        f.writelines(train_header + dev_lines)

    print(f"Data saved to {setting_dir}")
    print(f"- train.tsv: {len(train_lines)} examples (original distribution)")
    print(f"- dev.tsv: {len(dev_lines)} examples")
    print(f"- test.tsv: {len(test_lines)} examples")

if __name__ == "__main__":
    main()