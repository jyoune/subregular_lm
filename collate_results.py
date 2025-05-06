import pandas as pd
import jsonlines
import pathlib

def process_language(lang_file):
    pass


def make_model_df(model_dir):
    """Iterates through a model's results directory containing one jsonl file per language tested.
    Each json contains accuracy and f1 scores for one out of four test sets."""
    column_names = ["Language Name", "Short Random Accuracy", "Short Adversarial Accuracy", "Long Random Accuracy",
                    "Long Adversarial Accuracy", "Short Random F1", "Short Adversarial F1", "Long Random F1", "Long Adversarial F1"]
    all_results = []
    for lang_file in pathlib.Path(model_dir).glob('*.jsonl'):
        with jsonlines.open(lang_file, 'r') as f:
            results_dict = {"Language Name":pathlib.Path(lang_file).stem}
            for json in f:
                test_set = json["test_set"]
                match test_set:
                    case "test_sr":
                        results_dict["Short Random Accuracy"] = json["accuracy"]
                        # oops they're nested
                        results_dict["Short Random F1"] = json["f1"]["f1"]
                    case "test_sa":
                        results_dict["Short Adversarial Accuracy"] = json["accuracy"]
                        results_dict["Short Adversarial F1"] = json["f1"]["f1"]
                    case "test_lr":
                        results_dict["Long Random Accuracy"] = json["accuracy"]
                        results_dict["Long Random F1"] = json["f1"]["f1"]
                    case "test_la":
                        results_dict["Long Adversarial Accuracy"] = json["accuracy"]
                        results_dict["Long Adversarial F1"] = json["f1"]["f1"]
        all_results.append(results_dict)
    results_df = pd.DataFrame(all_results, columns=column_names)
    return results_df


def average_across_langs(model_dirs: dict[str:str]):
    avg_list = []
    column_names = ["Model Name", "Short Random Accuracy", "Short Adversarial Accuracy", "Long Random Accuracy",
                    "Long Adversarial Accuracy", "Short Random F1", "Short Adversarial F1", "Long Random F1", "Long Adversarial F1"]
    for model_name in model_dirs:
        model_dict = {"Model Name": model_name}
        curr_avg = make_model_df(model_dirs[model_name]).mean(axis=0, numeric_only=True)
        model_dict.update(curr_avg)
        avg_list.append(model_dict)
    return pd.DataFrame(avg_list, columns=column_names)


def average_across_tests(model_dirs: dict[str:str]):
    avg_dict = {}
    column_names = ["Language Name"]
    # make columns
    for model_name in model_dirs:
        column_names.extend([model_name + " Accuracy", model_name + " F1"])
    for model_name in model_dirs:
        # add lang names first
        curr_df = make_model_df(model_dirs[model_name])
        if "Language Name" not in avg_dict:
            avg_dict["Language Name"] = curr_df["Language Name"]
        curr_avg_acc = curr_df.loc[:, ["Short Random Accuracy", "Short Adversarial Accuracy", "Long Random Accuracy",
                                       "Long Adversarial Accuracy"]].mean(axis=1, numeric_only=True)
        avg_dict[model_name + " Accuracy"] = curr_avg_acc
        curr_avg_f1 = curr_df.loc[:, ["Short Random F1", "Short Adversarial F1", "Long Random F1", "Long Adversarial F1"]].mean(axis=1, numeric_only=True)
        avg_dict[model_name + " F1"] = curr_avg_f1
    return pd.DataFrame(avg_dict, columns=column_names)




if __name__ == "__main__":
    make_model_df("./bert_results_2")


