import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from preprocess import load_data
import evaluate
import jsonlines
import os
from tqdm import tqdm
from train_llm import train_llm
from eval_llm import evaluate_llm

os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"
device = "cuda"
# device = "cpu"
base_model_name = 'meta-llama/Llama-3.2-3B'
output_dir = "llama"
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
# first training run
directory = "data/ZP313"
out_file = "llm/llm_ZP313_nospace.jsonl"
dataset = load_data(directory=directory, use_spaces=False)
train_llm(model_name=base_model_name, data=dataset, model_output="./llama/llama_model_ZP_nospace")
# eval
pretrained_model = "./llama/llama_model_ZP_nospace"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, pretrained_model)
evaluate_llm(model, tokenizer, dataset, out_file)

# second training run

directory = "data/SL413"
out_file = "llm/llm_SL413_nospace.jsonl"
dataset = load_data(directory=directory, use_spaces=False)
train_llm(model_name=base_model_name, data=dataset, model_output="./llama/llama_model_SL_nospace")
# eval
pretrained_model = "./llama/llama_model_SL_nospace"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, pretrained_model)
evaluate_llm(model, tokenizer, dataset, out_file)
