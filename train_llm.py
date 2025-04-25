import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from preprocess import load_data, compute_metrics
import evaluate
import jsonlines
import os
from huggingface_hub import login
login("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")


#TODO: fix prompt function - include representation of regular language and ask about string membership?
# or just prompt it to learn it implicitly (e.g there's a specific regular language i'm thinking of.
# is this string in it? 1 or 0.)
#TODO: make sure labels are inserted properly. (included in prompt for eval?)
#TODO: write new eval function?
#TODO: fix saving eval results




def prompt_example(example):
    prompt = (
        f"Question: I am thinking of a regular expression. Here is a string of characters. Is it in the language? Answer 1 for yes and 0 for no."
        f"string: {example['string']}"
        f"answer: {str(example['label'])}"
    )
    return prompt


def tokenize_data(data):
    return tokenizer(data["string"], padding="max_length", truncation=True)


def train_llm(model_name: str, data, output_dir: str, out_file:str = "results.txt"):
    #quantization_config = BitsAndBytesConfig(
    #    load_in_4_bit=True,
    #    bnb_4bit_quant_type="nf4"
    #)

    peft_config = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="lora_only",
        task_type="CAUSAL_LM",
        use_rslora=True,
        target_modules=["q_proj", "v_proj"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        #quantization_config=quantization_config
    )
    sft_args = SFTConfig(
        output_dir="llm_outputs",
        max_seq_length=256,
        packing=True,
        bf16=True,
        label_names=["labels"],
        learning_rate=1e-4,
        gradient_checkpointing=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        do_eval=False,
        #eval_strategy="epoch"
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        peft_config=peft_config,
        formatting_func=prompt_example,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model("./llama/llama_model")
    # # iterate through all test sets in the constant
    # for test_set in TEST_EVAL_ORDER:
    #     evaluated = trainer.evaluate(data[test_set])
    #     print(evaluated)
    #     with jsonlines.open(out_file, "a") as f:
    #         f.write({"language": directory, "test_set": test_set, "accuracy": evaluated["eval_accuracy"],
    #                  "f1": evaluated["eval_f1"]})



if __name__ == "__main__":
    os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"
    device = "cuda"
    # device = "cpu"
    directory = "data/SL413"
    out_file = "llm_SL413.jsonl"
    model_name = 'meta-llama/Llama-3.2-3B'
    output_dir = "llama"
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_data(directory=directory, use_spaces=True)
    # tokenized_dataset = dataset.map(tokenize_data, batched=True)
    train_llm(model_name=model_name, data=dataset, output_dir=output_dir, out_file=out_file)
    




