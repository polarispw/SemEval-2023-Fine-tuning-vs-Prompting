"""
This file is used to collate the data files into a single file.
"""
import json

from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from tqdm import tqdm


def merge_argue_and_label(arg_path, label_path, save_path):
    """
    Merge the argue and label data files.
    """
    arg_dataset = load_dataset("csv", data_files=arg_path, delimiter="\t")
    label_dataset = load_dataset("csv", data_files=label_path, delimiter="\t")

    assert arg_dataset.num_rows == label_dataset.num_rows, "The number of rows in the two datasets are not equal."

    miss_match = []
    if isinstance(arg_dataset, DatasetDict):
        if len(arg_dataset) > 1:
            raise ValueError("The dataset is a DatasetDict with  more than one dataset.")
        else:
            key = list(arg_dataset.keys())[0]
            arg_dataset = arg_dataset[key]

    if isinstance(label_dataset, DatasetDict):
        if len(label_dataset) > 1:
            raise ValueError("The dataset is a DatasetDict with  more than one dataset.")
        else:
            key = list(label_dataset.keys())[0]
            label_dataset = label_dataset[key]

    for row in range(arg_dataset.num_rows):
        if arg_dataset[row]["Argument ID"] != label_dataset[row]["Argument ID"]:
            miss_match.append(row)

    if len(miss_match) > 0:
        raise ValueError(f"The following rows are not matched: {miss_match}")

    label_dataset = label_dataset.remove_columns("Argument ID")
    for col in label_dataset.column_names:
        arg_dataset = arg_dataset.add_column(col, label_dataset[col])

    if 'csv' in save_path:
        arg_dataset.to_csv(save_path)
    elif 'json' in save_path:
        arg_dataset.to_json(save_path)

    return arg_dataset


def flat_labels(dataset, col_names, save_path=None):
    """
    Flatten the labels in the dataset.
    """
    if isinstance(dataset, DatasetDict):
        if len(dataset) > 1:
            raise ValueError("The dataset is a DatasetDict with  more than one dataset.")
        else:
            key = list(dataset.keys())[0]
            dataset = dataset[key]

    # add the label category column
    dataset = dataset.add_column("label_category", [None] * dataset.num_rows)
    dataset = dataset.add_column("label", [None] * dataset.num_rows)

    # flatten the labels to the rows
    new_dataset = {"Argument ID": [],
                   "Premise": [],
                   "Stance": [],
                   "Conclusion": [],
                   "label_category": [],
                   "label": []}
    for i in tqdm(range(dataset.num_rows)):
        item = dataset[i]
        for label_name in col_names:
            new_dataset["Argument ID"].append(item["Argument ID"])
            new_dataset["Premise"].append(item["Premise"])
            new_dataset["Stance"].append(item["Stance"])
            new_dataset["Conclusion"].append(item["Conclusion"])
            new_dataset["label_category"].append(label_name)
            new_dataset["label"].append(item[label_name])

    dataset = Dataset.from_dict(new_dataset)

    if 'csv' in save_path:
        dataset.to_csv(save_path)
    elif 'json' in save_path:
        dataset.to_json(save_path)

    return dataset


def load_value_categories(path):
    """
    Load the value categories from json.
    """
    with open(path, "r") as f:
        category_dict = json.load(f)

    return category_dict


if __name__ == "__main__":
    arg_path = "../data_lib/arguments-training.tsv"
    label_path = "../data_lib/labels-training.tsv"
    save_path = "../data_lib/flatten_train_set.json"

    m_dataset = merge_argue_and_label(arg_path, label_path, save_path)
    # extract first 10 rows
    f_dataset = flat_labels(m_dataset, ["Self-direction: thought",
                                        "Self-direction: action",
                                        "Stimulation",
                                        "Hedonism",
                                        "Achievement",
                                        "Power: dominance",
                                        "Power: resources",
                                        "Security: personal",
                                        "Security: societal",
                                        "Tradition",
                                        "Conformity: rules",
                                        "Conformity: interpersonal",
                                        "Humility",
                                        "Face",
                                        "Benevolence: caring",
                                        "Benevolence: dependability",
                                        "Universalism: concern",
                                        "Universalism: nature",
                                        "Universalism: tolerance",
                                        "Universalism: objectivity",], save_path=save_path)
    
    print(load_dataset("json", data_files=save_path))
    print(load_value_categories("../data_lib/value-categories.json"))
