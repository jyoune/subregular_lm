import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset, DatasetDict
import evaluate
import pathlib
from datasets import Dataset, DatasetDict

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
# establish a mapping between indices and files in each language directory
# this is consistent assuming for each language i create a directory that stores all 6 relevant
# data files and do not rename anything
file_order = ["dev", "test_la", "test_lr", "test_sa", "test_sr", "train"]


def process_data(directory, use_spaces: bool = False) -> dict[str, pd.DataFrame]:
    """
    Processes files in a directory into pandas dataframes and returns them collected as a dictionary
    labeled according to the predefined order that they appear in
    :param directory: name for directory containing a given regular language's train, dev, and test sets
    :param use_spaces: bool for whether to split each
    :return: dict of str -> dataframe
    """
    file_dict = {}
    path = pathlib.Path(directory)
    for file, name in zip(path.iterdir(), file_order):
        data = pd.read_csv(file, sep='\t', names=["string", "label"])
        if use_spaces:
            data["string"] = data["string"].apply(lambda x: ' '.join(x))
        # replace TRUE/FALSE with 1/0
        data["label"] = data["label"].apply(lambda x: 1 if x == True else 0)
        file_dict[name] = data
    return file_dict


def load_data(directory, use_spaces: bool = False):
    """
    Loads data into a DatasetDict
    :param directory:
    :param use_spaces:
    :return:
    """
    file_dict = process_data(directory, use_spaces)
    dataset = DatasetDict()
    for name in file_dict:
        dataset[name] = Dataset.from_pandas(file_dict[name])
    return dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy_dict = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_dict = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    new_dict = {key: accuracy_dict[key] for key in accuracy_dict}
    new_dict["f1"] = f1_dict
    return new_dict
