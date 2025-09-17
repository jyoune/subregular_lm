# Subregular Language Learning
This is a project that extends the work done by Van der Poel et. al in [MLRegTest](https://arxiv.org/abs/2304.07687) to a set of larger transformer models including Google's [CANINE-c](https://huggingface.co/google/canine-c) (tokenization-free transformer encoder trained on autoregressive character loss), [CANINE-s](https://huggingface.co/google/canine-s) (same thing but trained on subword loss instead), and [BERT](https://huggingface.co/google-bert/bert-base-uncased).

# Files

`data` - contains the train, eval, and test sets (in txt form, can be parsed as two-column dataframes or similar) for each of the 32 languages used in this project, organized in subdirectories.

`results` - contains the results for each language for BERT and the two CANINE models, organized in subdirectories according to model and whether spaces are inserted between characters in the data.

`preprocess.py`- contains methods to process and load data from a given directory as well as computing metrics. 

`train_slm.py`- script to run for training and evaluating a (small) language model (e.g BERT, CANINE) using command line arguments to specify models, datasets, and various training parameters.

`train_llm.py` - contains methods to train a large language model. can be run as a script to obtain a fine-tuned LLM for separate evaluation.

`eval_llm.py`- contains methods to evaluate large language models. can be run as a script to evaluate an extant fine-tuned model.

`train_eval_llm_script.py` - short script that invokes both train_llm and eval_llm to train and evaluate Llama 3.2-3B on two of the selected languages.

`collate_results.py` - contains methods to collate the evaluation results of language models. includes averages across languages, test types, and models.

`stream.py`- streamlit script that displays collated results for BERT, CANINE-s, and CANINE-c.
