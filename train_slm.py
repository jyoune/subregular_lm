from transformers import CanineTokenizer, CanineForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import evaluate
import torch
from preprocess import load_data, compute_metrics
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-model_name")
parser.add_argument("-lang_dir")
parser.add_argument("-eval_only")
parser.add_argument("-use_lora")
parser.add_argument("-use_space")
parser.add_argument("-out_dir")
parser.add_argument("-results_file")


# constant for iterating through all test sets
TEST_EVAL_ORDER = ["test_sr", "test_sa", "test_lr", "test_la"]


def tokenize_data(data):
    return tokenizer(data["string"], padding="max_length", truncation=True)


def train_eval_bert(model_name: str, eval_only: bool, use_lora: bool, lora_rank: int, use_rs: bool, data, output_dir: str, out_file:str = "results.txt"):
    device = "cuda"
    model = CanineForSequenceClassification.from_pretrained(model_name).to(device)
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="lora_only",
        use_rslora=use_rs,
        target_modules=["query", "value"]
    )
    lora_model = get_peft_model(model, config).to(device)
    training_args = TrainingArguments(output_dir=output_dir,
                                      # logging_steps=100,
                                      eval_strategy="epoch",
                                      # eval_steps=5,
                                      label_names=["labels"],
                                      report_to="none",
                                      per_device_train_batch_size=16)
    if use_lora:
        trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=data["train"],
            eval_dataset=data["dev"],
            compute_metrics=compute_metrics
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data["train"],
            eval_dataset=data["dev"],
            compute_metrics=compute_metrics
        )
    # training and eval only modes
    if not eval_only:
        trainer.train()
    # iterate through all test sets in the constant
    for test_set in TEST_EVAL_ORDER:
        evaluated = trainer.evaluate(data[test_set])
        print(evaluated)
        with open(out_file, "a") as f:
            f.write(f"language: {directory}, test set: {test_set}, rank: {lora_rank}, use_rs: {use_rs} \n {evaluated} \n")


if __name__ == "__main__":
    # get args
    args = parser.parse_args()
    model_name = args.model_name
    eval_only = args.eval_only
    directory = args.lang_dir
    if_lora = args.use_lora
    out_dir = args.out_dir
    results_file = args.results_file
    use_space = args.use_space
    tokenizer = CanineTokenizer.from_pretrained(model_name)
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    dataset = load_data(directory, use_spaces=use_space)
    tokenized_dataset = dataset.map(tokenize_data, batched=True)
    train_eval_bert(model_name=model_name, eval_only=eval_only, use_lora=if_lora, lora_rank=32, use_rs=False, data=tokenized_dataset, output_dir="rank32_ZP", out_file=results_file)