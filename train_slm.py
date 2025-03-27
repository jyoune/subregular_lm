from transformers import CanineTokenizer, CanineForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import evaluate
import torch
from preprocess import load_data, compute_metrics

# constant for iterating through all test sets
TEST_EVAL_ORDER = ["test_sr", "test_sa", "test_lr", "test_la"]


def tokenize_data(data):
    return tokenizer(data["string"], padding="max_length", truncation=True)


def train_eval_bert(lora_rank: int, use_rs: bool, data, output:str):
    device = "cuda"
    model = CanineForSequenceClassification.from_pretrained("google/canine-c").to(device)
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="lora_only",
        use_rslora=use_rs,
        target_modules=["query", "value"]
    )
    lora_model = get_peft_model(model, config).to(device)
    training_args = TrainingArguments(output_dir=output,
                                      # logging_steps=100,
                                      eval_strategy="epoch",
                                      # eval_steps=5,
                                      label_names=["labels"],
                                      report_to="none",
                                      per_device_train_batch_size=16)
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        compute_metrics=compute_metrics
    )
    trainer.train()
    # iterate through all test sets in the constant
    for test_set in TEST_EVAL_ORDER:
        evaluated = trainer.evaluate(data[test_set])
        print(evaluated)
        with open("results.txt", "a") as f:
            f.write(f"language: {directory}, test set: {test_set}, rank: {lora_rank}, use_rs: {use_rs} \n {evaluated} \n")


if __name__ == "__main__":
    directory = "data/coSP413"
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    dataset = load_data(directory, use_spaces=True)
    tokenized_dataset = dataset.map(tokenize_data, batched=True)
    train_eval_bert(lora_rank=32, use_rs=False, data=tokenized_dataset, output="rank32")