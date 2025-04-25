import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer
from preprocess import load_data
import evaluate
import jsonlines
import os
from tqdm import tqdm

TEST_EVAL_ORDER = ["test_sr", "test_sa", "test_lr", "test_la"]


def prompt_example(example):
    prompt = (
        f"Question: I am thinking of a regular expression. Here is a string of characters. Is it in the language? Answer 1 for yes and 0 for no."
        f"string: {example['string']}"
        f"answer:"
    )
    return prompt


def parse_predicted_label(generated_text):
    if "answer:" in generated_text:
        answer_part = generated_text.split("answer:")[-1].strip()
        if answer_part.startswith("0"):
            return 0
        else:
            return 1
    return 0 # fallback if the structure is unexpected


def eval_per_set(test_set, model, tokenizer):
    pred_labels = []
    gold_labels = []
    for example in tqdm(test_set):
        gold_labels.append(example["label"])
        prompt = prompt_example(example)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            # set max_new_tokens low to avoid OOM
            outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_label = parse_predicted_label(generated_text)
        pred_labels.append(pred_label)
    result = compute_metrics(gold_labels, pred_labels)
    return result


def evaluate_llm(model, tokenizer, data, out_file):
    # iterate through all test sets in the constant
    for test_set in TEST_EVAL_ORDER:
        evaluated = eval_per_set(data[test_set], model, tokenizer)
        print(evaluated)
        with jsonlines.open(out_file, "a") as f:
            f.write({"language": directory, "test_set": test_set, "accuracy": evaluated["accuracy"],
                     "f1": evaluated["f1"]})


def compute_metrics(labels, predictions):
    accuracy_dict = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_dict = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    new_dict = {key: accuracy_dict[key] for key in accuracy_dict}
    new_dict["f1"] = f1_dict
    return new_dict


if __name__ == "__main__":
    device = "cuda"
    # device = "cpu"
    directory = "data/ZP313"
    out_file = "llm/llm_ZP313.jsonl"
    base_model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    output_dir = "llama"
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_data(directory=directory, use_spaces=True)
    pretrained_model = "./llama/llama_model_ZP"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, pretrained_model)
    evaluate_llm(model, tokenizer, dataset, out_file)

