from utils import utils
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from typing import List, Dict
from datasets import Dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import transformers

"""
This module is for preprocessing the evaluation data in the GAR model training pipeline, 
it will takein the raw eavl data and process into the data loader that suits the training.

The main function for this module is to tokenize the eval data and create the dataloader.
"""

DeepSeekProver_HEADER_CoT = \
"""Complete the following Lean 4 code with explanatory comments preceding each line of code:

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat"""



def preprocess_eval_data(
    eval_dataset: List[Dict]
) -> List[Dict]:
    """
    This function preprocess the evaluation into input-output pairs for the model. We assume the pair is only in code-filling format.
    :param eval_dataset: List of dict of the data, each inner dicts should contains the following keys:
        - "Name": the name of the problem
        - "Statement": The Lean4 statement of the problem
        - "Informal_statement": The natural language statement of the problem
        - "Commented_proof": The commented Lean4 proof of the problem, we assume it is started with the statement
    
    :return: List of dicts, each dict contains the input-output pair for the model.
        Each dict should contains the following keys:
        - "Name": the name of the problem
        - "idx": the index of the problem in the original dataset
        - "input": the input text for the model, which is the statement and the header
        - "output": the output text for the model, which is the proof without the statement
    """

    processed_data = []

    for idx, record in enumerate(eval_dataset):
        curr_statement = record["Statement"]
        curr_proof = record["Commented_proof"]

        # If the proof does not start with the statement, we skip this record
        if not curr_proof.startswith(curr_statement):
            print(f"Warning: The proof does not start with the statement for problem {record['Name']}. Skipping this record.")
            continue

        # We assume the proof is in code-filling format, so we create the input-output pair accordingly
        input_text = f"""{DeepSeekProver_HEADER_CoT}

/-- {record["Informal_statement"]} -/
{curr_statement}
"""
        output_text = curr_proof[len(curr_statement):] + "\n```"

        processed_data.append({
            "Name": record["Name"],
            "idx": idx,
            "input": input_text,
            "output": output_text
        })

    return processed_data


def tokenize_data_ls(
        data_ls: List[Dict], 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048
) -> Dataset:
    """
    This is the helper function to tokenize the data list into a dataset.
    :param data_ls: List of dict, each dict should contains the following keys:
        - "idx": the index of the problem in the original dataset
        - "input": the input text for the model, which is the statement and the header
        - "output": the output text for the model, which is the proof without the statement
    :param tokenizer: The tokenizer to use for tokenizing the data
    :param max_length: The maximum length of the tokenized input-output pair

    :return: A Dataset object containing the tokenized data, the dataset will have the following columns:
        - "input_ids": The input ids of the tokenized input text
        - "labels": The labels of the tokenized output text
    """

    dataset_dict = {}
    for curr_key in list(data_ls[0].keys()):
        dataset_dict[curr_key] = [curr_record[curr_key] for curr_record in data_ls]
    dataset = Dataset.from_dict(dataset_dict)

    input_ids = [[] for _ in dataset["input"]]
    labels = [[] for _ in dataset["input"]]

    raw_input_list = dataset["input"]
    raw_input_list = [str(x) for x in raw_input_list]
    tokenized_input = tokenizer(raw_input_list, max_length=max_length, truncation=True)["input_ids"]
    # tokenized_input = tokenizer(dataset["input"],
    #                             max_length=max_length,
    #                             truncation=True,
    #                             add_special_tokens=True)

    
    print("processing input...")
    for i, curr_input in enumerate(tqdm(tokenized_input)):
        if curr_input[-1] == tokenizer.eos_token_id:
            curr_input = curr_input[:-1]
        input_ids[i] += curr_input
        labels[i] += [-100] * len(curr_input)

    raw_output_list = dataset["output"]
    raw_output_list = [str(x) for x in raw_output_list]
    tokenized_output = tokenizer(raw_output_list, max_length=max_length, truncation=True)["input_ids"]

    # tokenized_output = tokenizer(dataset["output"],
    #                              max_length=max_length,
    #                              truncation=True,
    #                              add_special_tokens=False)["input_ids"]

    print("processing output...")
    for i, curr_output in enumerate(tqdm(tokenized_output)):
        # If the output is not ending, add the eos token
        if len(curr_output) == 0 or curr_output[-1] != tokenizer.eos_token_id:
            curr_output += [tokenizer.eos_token_id]
        input_ids[i] += curr_output
        labels[i] += curr_output

        if len(input_ids[i]) > max_length:
            input_ids[i] = input_ids[i][:max_length]
            labels[i] = labels[i][:max_length]
    
    dic_to_return = {
        "input_ids": input_ids,
        "labels": labels
    }

    return Dataset.from_dict(dic_to_return)


def obtain_eval_dataloader(
    eval_dataset: List[Dict], 
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048, 
    num_workers: int = 32, 
    micro_batch_size: int = 1
) -> DataLoader:
    """
    This function preprocess the evaluation data into a DataLoader for the model. It will tokenize the data and create the DataLoader.
    :param eval_dataset: List of dict, each dict should contains the following keys:
        - "Name": the name of the problem
        - "Statement": The Lean4 statement of the problem
        - "Informal_statement": The natural language statement of the problem
        - "Commented_proof": The commented Lean4 proof of the problem, we assume it is started with the statement
    :param tokenizer: The tokenizer to use for tokenizing the data
    :param max_length: The maximum length of the tokenized input-output pair
    :param num_workers: The number of workers to use for the DataLoader
    :param micro_batch_size: The micro batch size for training, it will be used to create the DataLoader as batch size
    :return: A DataLoader object containing the tokenized data, the DataLoader will have the following columns:
        - "input_ids": The input ids of the tokenized input text
        - "labels": The labels of the tokenized output text
    """

    preprocessed_data = preprocess_eval_data(eval_dataset)
    tokenized_dataset = tokenize_data_ls(preprocessed_data, tokenizer, max_length)
    data_loader = DataLoader(
        tokenized_dataset, 
        collate_fn=transformers.DataCollatorForSeq2Seq(
            tokenizer, 
            padding=True, 
            pad_to_multiple_of=2,
            return_tensors="pt"
        ), 
        batch_size=micro_batch_size,
        num_workers=num_workers,
    )
    return data_loader
