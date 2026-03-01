# RL_trainner.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict

# Standard library
import os
import sys
import time
import random
import shutil
import math
import signal
import hashlib
import copy
import datetime
import gc
import pickle
import multiprocessing
from functools import partial
from typing import List, Dict, Any, Optional, Tuple

# Third-party libraries
import wandb
import torch
from datasets import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm as original_tqdm
from concurrent.futures import ThreadPoolExecutor

# Hugging Face Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

# Hugging Face Datasets
from datasets import load_dataset

# Accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger

# self-defined libraries
from utils import utils


# ============================================ Pre-computation of Log Probabilities =============================================


# Helper functions for log probability computation
def _stream_compute_log_probs(
        old_model, 
        ref_model, 
        tokenized_content: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes log probabilities on two models concurrently using CUDA streams.
    """
    ids_cpu = tokenized_content["input_ids"]          # CPU tensor
    ids_old = ids_cpu.to(old_model.device, non_blocking=False)
    ids_ref = ids_cpu.to(ref_model.device, non_blocking=False)

    s_old = torch.cuda.Stream(device=old_model.device)
    s_ref = torch.cuda.Stream(device=ref_model.device)

    with torch.cuda.stream(s_old):
        with torch.no_grad():
            old_lp = _obtain_log_probs_on_device(old_model, ids_old).cpu()
    with torch.cuda.stream(s_ref):
        with torch.no_grad():
            ref_lp = _obtain_log_probs_on_device(ref_model, ids_ref).cpu()

    s_old.synchronize()
    s_ref.synchronize()
    return old_lp, ref_lp



# precompute for NL fuser
def summarize_precompute_results_NL_fuser(
        ckpt_root_path: str = "./NL_fuser_log_prob_ckpt_path",
        save_path: Optional[str] = None
):
    """
    This function gathers all the .pkl data from each worker's 'NL_fuser_log_prob_ckpt_path' subfolder.
    Each .pkl is a dict, and all dicts across all workers are merged into a big list.
    """

    all_results = []
    for worker_name in os.listdir(ckpt_root_path):
        worker_dir = os.path.join(ckpt_root_path, worker_name)
        ckpt_dir = os.path.join(worker_dir, "NL_fuser_log_prob_ckpt_path")

        if not os.path.isdir(ckpt_dir):
            continue
        
        print(f"[Info] Processing {ckpt_dir}")
        for file in os.listdir(ckpt_dir):
            if file.endswith(".pkl"):
                file_path = os.path.join(ckpt_dir, file)
                try:
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                        if isinstance(data, dict):
                            all_results.append(data)
                        else:
                            print(f"[Warning] {file_path} is not a dict. Skipped.")
                except Exception as e:
                    print(f"[Error] Failed to load {file_path}: {e}")
    print(f"[Done] Total number of dicts collected: {len(all_results)}")

    # Save as a single list of dicts
    if save_path is None:
        save_path = os.path.join(ckpt_root_path, "merged_precomputed_results.pkl")
    else:
        if not save_path.endswith(".pkl"):
            save_path += ".pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"[Saved] Merged results saved to {save_path}")

@torch.no_grad()
def pre_compute_log_probs_NL_fuser(
        old_model, 
        ref_model,
        tokenizer,
        dataset_to_compute: List[Dict],
        batch_size: int = 4,
        ckpt_path: str = "./NL_fuser_log_prob_ckpt_path",
        max_length: int = 16384,
        parallel_compute: bool = True
):
    """
    This is the function that pre-comnputes the log probabilities of the model outputs for a given dataset, which is used for the NL fuser.
    It will save the pre-computed log probabilities for the NL fuser into the following data structure:
        - f"{ckpt_path}/{batch_name}.pkl":
            - "name": The list of name of the batch, which is used to uniquely identify the batch.
            - "input": The list of text inputs for the model, which is the prompt together with any necessary formatting string like the generation header.
            - "output": The list of text outputs of the model, which is the generated content.
            - "tokenized_content": The tokenized of the input and output, which is used to calculate the probabilities.
            - "loss_mask": The mask for the merged text of input and output, which is used to calculate the loss. It is a torch.Tensor with the same shape as the tokenized content, where 1 indicates the token should be considered in the loss calculation, and 0 indicates the token should be ignored.
            - "old_model_log_prob": The log probs for the old model in the batch in torch.Tensor format.
            - "ref_model_log_prob": The log probs for the reference model in the batch in torch.Tensor format.
    :param old_model: The old model from the last step of gradient update, which is used to calculate the log probs for the old model.
    :param ref_model: The reference model, which is the base model for training, used to calculate the log probs for the reference model.
    :param tokenizer: The tokenizer of the LLM, which is used to tokenize the dataset.
    :param dataset_to_compute: The dataset to compute the log probabilities, it should be generated through RL_data_preprocessor.RL_DataPreprocessor.preprocess_NL_fuser_training_data.
        It is a list of dict, where the inner dict has the following keys:
            - "name": The name of the batch, which is used to uniquely identify the batch.
            - "input": The input text for the model, which is the prompt together with any necessary formatting string like the generation header.
            - "output": The output text of the model, which is the generated content.
            - "correctness": A 0~1 float variable indicating (1 - pass_rate) * (1 - statement_modification_rate) 
                and it will be 0 if the pass_rate is 0 (the statement maybe unsolvable). This is used as reward of the NL fuser model. 
    :param batch_size: The batch size for computation and training, which should be identical to the actual RL training batch size
    :param max_length: The maximum length of the tokenized content, which is used to truncate the input and output text. It should be identical to the actual RL training max_length.
    :param ckpt_path: The path to save the pre-computed log probabilities, it should be a folder that exists before the function is called.
        If the folder does not exist, it will be created.
    :param parallel_compute: Whether to use parallel computation to speed up the log probabilities calculation. If set to True, it will use two CUDA streams to compute the log probabilities concurrently.
    """

    utils.check_folder_exit(ckpt_path)
    old_model.eval()
    ref_model.eval()

    batched_training_data = utils.split_batch(
        dataset_to_compute,
        batch_size=batch_size
    )


    for curr_batch in original_tqdm(batched_training_data):

        # Test for without statement modification penalty reward
        rewards = torch.tensor([
            curr_record["correctness"] for curr_record in curr_batch
        ])

        # prepare the input and output text
        input_texts = [curr_record["input"] for curr_record in curr_batch]
        output_texts = [curr_record["output"] for curr_record in curr_batch]

        assert len(input_texts) == len(output_texts), "Input and output texts must have the same length."

        merged_texts = [input_text + output_text for input_text, output_text in zip(input_texts, output_texts)]

        tokenized_content = tokenizer(
            merged_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(torch.device("cpu"))

        prompt_lens = torch.Tensor([
            len(tokenizer(curr_input_text, add_special_tokens=False)["input_ids"])
            for curr_input_text in input_texts
        ])


        valid_lens = tokenized_content["attention_mask"].sum(dim=-1)

        B, L = tokenized_content["input_ids"].shape
        arange = torch.arange(L).unsqueeze(0).expand(B, L)

        loss_mask = (arange >= prompt_lens.unsqueeze(1)) & (arange < valid_lens.unsqueeze(1))
        loss_mask = loss_mask.to(tokenized_content["input_ids"].dtype)   

        loss_mask = loss_mask * tokenized_content["attention_mask"]

        # obtain log probabilities for the old model and the reference model
        with torch.no_grad():
            try:
                print(f"current input ids shape: {tokenized_content['input_ids'].shape}\ncurrent reward is {rewards}")
                if parallel_compute:
                    old_model_log_prob, ref_model_log_prob = _stream_compute_log_probs(
                        old_model=old_model,
                        ref_model=ref_model,
                        tokenized_content=tokenized_content
                    )
                else:
                    old_model_log_prob = _obtain_log_probs(old_model, tokenized_content["input_ids"].to(old_model.device)).to(torch.device("cpu"))
                    ref_model_log_prob = _obtain_log_probs(ref_model, tokenized_content["input_ids"].to(ref_model.device)).to(torch.device("cpu"))
            except Exception as e:
                if "CUDA out of memory" in str(e):
                    print("⚠️ CUDA OOM detected, fallback to single-sample mode")

                    old_model_log_prob_list = []
                    ref_model_log_prob_list = []

                    for j in range(tokenized_content["input_ids"].size(0)):
                        try:
                            curr_input_ids = tokenized_content["input_ids"][j:j+1]
                            if parallel_compute:
                                curr_old_model_log_prob, curr_ref_model_log_prob = _stream_compute_log_probs(
                                    old_model=old_model, 
                                    ref_model=ref_model, 
                                    tokenized_content={
                                        "input_ids": curr_input_ids,
                                        "attention_mask": tokenized_content["attention_mask"][j:j+1]
                                    }
                                )
                            else:
                                curr_old_model_log_prob = _obtain_log_probs(
                                    old_model, 
                                    curr_input_ids.to(old_model.device)
                                ).to(torch.device("cpu"))
                                curr_ref_model_log_prob = _obtain_log_probs(
                                    ref_model, 
                                    curr_input_ids.to(ref_model.device)
                                ).to(torch.device("cpu"))
                            old_model_log_prob_list.append(curr_old_model_log_prob)
                            ref_model_log_prob_list.append(curr_ref_model_log_prob)
                            del curr_input_ids, curr_old_model_log_prob, curr_ref_model_log_prob
                            torch.cuda.empty_cache()
                            gc.collect()
                        except RuntimeError as inner_e:
                            print(f"❌ OOM on single sample {j} of batch")
                            raise inner_e
                    old_model_log_prob = torch.cat(old_model_log_prob_list, dim=0)
                    ref_model_log_prob = torch.cat(ref_model_log_prob_list, dim=0)
                else:
                    raise e
        # append the current batch data to the list
        curr_batch_data_to_save = {
            "name": curr_batch[0]["name"],  # We are using the first record's name as the batch name
            "input": input_texts,
            "output": output_texts,
            "tokenized_content": tokenized_content,
            "loss_mask": loss_mask,
            "old_model_log_prob": old_model_log_prob,
            "ref_model_log_prob": ref_model_log_prob,
            "rewards": rewards.to(torch.device("cpu"))  # move rewards to CPU for saving
        }
        with open(f"{ckpt_path}/{curr_batch_data_to_save['name']}_training_data.pkl", "wb") as f:
            pickle.dump(curr_batch_data_to_save, f)
        del tokenized_content, loss_mask, ref_model_log_prob, old_model_log_prob
        torch.cuda.empty_cache()
        gc.collect()

def _precompute_log_probs_NL_fuser_worker(
    gpu_idx_pair: Tuple[int, int],
    dataset_path: str,
    start_idx: int,
    end_idx: int,
    rank: int,
    ckpt_root: str,
    old_model_path: str,
    ref_model_path: str,
    batch_size: int = 4,
    max_length: int = 16384,
    parallel_compute: bool = True,
):
    import os, sys, json, time, gc
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_idx_pair[0]},{gpu_idx_pair[1]}"

    worker_root = os.path.join(ckpt_root, f"worker_{rank}")
    os.makedirs(worker_root, exist_ok=True)
    log_fp = open(os.path.join(worker_root, f"worker_{rank}.log"), "w", buffering=1, encoding="utf-8")
    err_fp = open(os.path.join(worker_root, f"worker_{rank}.err"), "w", buffering=1, encoding="utf-8")
    sys.stdout = log_fp
    sys.stderr = err_fp

    print(f"[Worker-{rank}] GPUs (physical) = {gpu_idx_pair}, slice= [{start_idx}:{end_idx})")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch, torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    dev_old = torch.device(f"cuda:{gpu_idx_pair[0]}")
    dev_ref = torch.device(f"cuda:{gpu_idx_pair[1]}")

    full = utils.read_from_json(dataset_path)
    dataset_chunk = full[start_idx:end_idx]
    print(f"[Worker-{rank}] chunk size = {len(dataset_chunk)}")

    print(f"[Worker-{rank}] loading models...")
    tokenizer = AutoTokenizer.from_pretrained(old_model_path)
    old_model = AutoModelForCausalLM.from_pretrained(old_model_path, torch_dtype=torch.bfloat16).to(dev_old)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, torch_dtype=torch.bfloat16).to(dev_ref)
    old_model.eval(); ref_model.eval()

    hb_path = os.path.join(worker_root, "heartbeat.json")
    worker_ckpt_path = os.path.join(worker_root, "NL_fuser_log_prob_ckpt_path")
    utils.check_folder_exit(worker_ckpt_path)

    try:
        pre_compute_log_probs_NL_fuser(
            old_model=old_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset_to_compute=dataset_chunk,
            batch_size=batch_size,
            ckpt_path=worker_ckpt_path,
            max_length=max_length,
            parallel_compute=parallel_compute
        )
        with open(hb_path, "w") as f:
            json.dump({"done": True, "ts": time.time(), "slice": [start_idx, end_idx]}, f)
        print(f"[Worker-{rank}] finished ✅")
    finally:
        del old_model, ref_model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        log_fp.close()
        err_fp.close()


def node_worker_precompute_log_probs_NL_fuser(
    gpu_idx_pair_ls: List[Tuple[int, int]],
    worker_idx: List[int],
    dataset_path: str,
    ckpt_root_path: str,
    old_model_path: str,
    ref_model_path: str,
    batch_size: int = 4,
    max_length: int = 16384,
    shuffle_data: bool = True,
    parallel_compute: bool = True,
):
    """
    This function is the main function for pre-computing the log probabilities for proofs in a node (typically 4 GPUs).
    It will split the dataset into chunks and assign each chunk to a worker thread.
    :param gpu_idx_pair_ls: A list of two tuples, each tuple contains two integers indicating the GPU indices to use for the old model and the reference model respectively.
    :param dataset_path: The path to the dataset to compute the log probabilities for, it should be a file path that exists before the function is called.
        The dataset should be a list of dicts, where each dict contains the following:
            - "name": The name of the theorem, which is used to uniquely identify the theorem.
            - "training_data": The training data for the model, which is obtained by generating the formal proof using the LLM for multiple times. It should be a list of dict, where the
                - "generation_idx": The index of the generation, which is used to identify the generation.
                - "input": The input text for the model, which is the prompt together with any necessary formatting string like the generation header.
                - "output": The output text of the model, which is the generated content.
                - "correctness": A bool variable indicating whether the generated content passes the verification.
    :param ckpt_root_path: The root path to save the pre-computed log probabilities, it should be a folder that exists before the function is called.
        If the folder does not exist, it will be created.
    :param old_model_path: The path to the old model from the last step of gradient update, which is used to calculate the log probs for the old model.
    :param ref_model_path: The path to the reference model, which is the base model for training, used to calculate the log probs for the reference model.
    :param batch_size: The batch size for computation and training, which should be identical to the actual RL training batch size.
    :param max_length: The maximum length of the tokenized content, which is used to truncate the input and output text. It should be identical to the actual RL training max_length.
    :param num_workers: The number of worker threads to use for computation, it should be the same as length of gpu_idx_pair_ls.
    :param shuffle_data: Whether to shuffle the dataset before splitting it into chunks, it can balance the computation load for each worker. If set to True, it will shuffle the dataset before splitting.
    :param parallel_compute: Whether to use parallel computation to speed up the log probabilities calculation. If set to True, it will use two CUDA streams to compute the log probabilities concurrently.

    """
    assert len(gpu_idx_pair_ls) == len(worker_idx), "gpu_idx_pair_ls 长度必须等于 worker_idx"

    full = utils.read_from_json(dataset_path)
    if shuffle_data:
        random.shuffle(full)
        tmp_dataset_path = os.path.join(ckpt_root_path, "_shuffled_dataset.json")
        os.makedirs(ckpt_root_path, exist_ok=True)
        utils.write_to_json(tmp_dataset_path, full)
        dataset_path_use = tmp_dataset_path
    else:
        dataset_path_use = dataset_path

    num_samples = len(full)
    n_workers = len(worker_idx)
    print(f"Total samples: {num_samples}, split into {n_workers} workers.")

    chunk_size = math.ceil(max(1, num_samples) / max(1, n_workers))

    ctx = multiprocessing.get_context("spawn")
    procs = []

    def _signal_handler(sig, frame):
        print(f"Received signal {sig}, terminating all workers...")
        for p in procs:
            if p.is_alive():
                p.terminate()
                print(f"Worker {p.pid} terminated.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    for i, (rank, gpu_pair) in enumerate(zip(worker_idx, gpu_idx_pair_ls)):
        start = i * chunk_size
        end   = min((i + 1) * chunk_size, num_samples)
        p = ctx.Process(
            target=_precompute_log_probs_NL_fuser_worker,
            args=(gpu_pair, dataset_path_use, start, end, rank, ckpt_root_path,
                  old_model_path, ref_model_path, batch_size, max_length, parallel_compute),
            name=f"GRPO-Precompute-NL-Fuser-Worker-{rank}",
        )
        p.daemon = False
        p.start()
        procs.append(p)
        print(f"Started worker {rank} on GPUs {gpu_pair} with PID {p.pid}, slice=[{start}:{end})")
        time.sleep(0.5)

    for p in procs:
        p.join()

    print("All workers finished. Pre-computation of log probabilities for NL fuser completed.")


# precompute for prover
def summarize_precompute_results_prover(
        ckpt_root_path: str = "./proofs_log_prob_ckpt_path", 
        save_path: Optional[str] = None
):
    """
    This function gathers all the .pkl data from each worker's 'proofs_log_prob_ckpt_path' subfolder.
    Each .pkl is a dict, and all dicts across all workers are merged into a big list.
    """
    all_results = []

    for worker_name in os.listdir(ckpt_root_path):
        worker_dir = os.path.join(ckpt_root_path, worker_name)
        ckpt_dir = os.path.join(worker_dir, "proofs_log_prob_ckpt_path")

        if not os.path.isdir(ckpt_dir):
            continue

        print(f"[Info] Processing {ckpt_dir}")
        for file in os.listdir(ckpt_dir):
            if file.endswith(".pkl"):
                file_path = os.path.join(ckpt_dir, file)
                try:
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                        if isinstance(data, dict):
                            all_results.append(data)
                        else:
                            print(f"[Warning] {file_path} is not a dict. Skipped.")
                except Exception as e:
                    print(f"[Error] Failed to load {file_path}: {e}")

    print(f"[Done] Total number of dicts collected: {len(all_results)}")

    # Save as a single list of dicts
    if save_path is None:
        save_path = os.path.join(ckpt_root_path, "merged_precomputed_results.pkl")
    else:
        if not save_path.endswith(".pkl"):
            save_path += ".pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)

    print(f"[Saved] Merged results saved to {save_path}")

@torch.no_grad()
def pre_compute_log_probs_proofs(
        old_model,
        ref_model, 
        tokenizer, 
        dataset_to_compute: List[Dict], 
        batch_size: int = 4, 
        ckpt_path: str = "./proofs_log_prob_ckpt_path", 
        max_length: int = 16384, 
        parallel_compute: bool = True
):
    """
    This is the function that pre-computes the log probabilities of the model outputs for a given dataset, which is used for the proofs. 
    It will save the pre-computed log probabilities for the proofs into the following data structure:
        - f"{ckpt_path}/{thm_name}.pkl":
            - "name": The name of the theorem, which is used to uniquely identify the theorem.
            - "training_batch_data": A list of dict of the training data for the model with batched proofs, all the batched proofs in the same batch will be concatenated together and padded into the same length. 
                Each dict contains the following keys:
                - "input": The list of text inputs for the model, which is the prompt together with any necessary formatting string like the generation header.
                - "output": The list of text outputs of the model, which is the generated content.
                - "tokenized_content": The tokenized of the input and output, which is used to calculate the probabilities.
                - "loss_mask": The mask for the merged text of input and output, which is used to calculate the loss. It is a torch.Tensor with the same shape as the tokenized content, where 1 indicates the token should be considered in the loss calculation, and 0 indicates the token should be ignored.
                - "old_model_log_prob": The log probs for the old model in the batch in torch.Tensor format.
                - "ref_model_log_prob": The log probs for the reference model in the batch in torch.Tensor format.
    :param old_model: The old model from the last step of gradient update, which is used to calculate the log probs for the old model.
    :param ref_model: The reference model, which is the base model for training, used to calculate the log probs for the reference model.
    :param tokenizer: The tokenizer of the LLM, which is used to tokenize the dataset.
    :param dataset_to_compute: The dataset to compute the log probabilities, it should be generated through RL_data_preprocessor.RL_DataPreprocessor.preprocess_Prover_training_data. 
        It is a list of dict, where the inner dict has the following keys:
            - "name": The name of the theorem, , which is used to uniquely identify the theorem.
            - "training_data": The training data for the model, which is obtained by generating the formal proof using the LLM for multiple times. It should be a list of dict, where the inner dict contains the following keys:
                - "generation_idx": The index of the generation, which is used to identify the generation.
                - "input": The input text for the model, which is the prompt together with any necessary formatting string like the generation header.
                - "output": The output text of the model, which is the generated content. 
                - "correctness": A bool variable indicating whether the generated content passes the verification.
                - "statement_modified": A bool variable indicating whether the statement is modified during the proof generation, this is useful in punishing the model for modifying the statement.
    :param batch_size: The batch size for computation and training, which should be identical to the actual RL training batch size
    :param max_length: The maximum length of the tokenized content, which is used to truncate the input and output text. It should be identical to the actual RL training max_length.
    :param ckpt_path: The path to save the pre-computed log probabilities, it should be a folder that exists before the function is called. 
        If the folder does not exist, it will be created.
    :param parallel_compute: Whether to use parallel computation to speed up the log probabilities calculation. If set to True, it will use two CUDA streams to compute the log probabilities concurrently.
    """

    utils.check_folder_exit(ckpt_path)
    old_model.eval()
    ref_model.eval()

    for curr_data_record in original_tqdm(dataset_to_compute):
        curr_batched_training_data = utils.split_batch(
            curr_data_record["training_data"], 
            batch_size=batch_size
        )
        thm_name = curr_data_record["name"]
        curr_training_batched_data = []

        for i, curr_batch in enumerate(curr_batched_training_data):
            reward_list = []

            for curr_record in curr_batch:
                if curr_record["correctness"]:

                    # Statement modification penalty reward design
                    correctness = True
                    if curr_record["statement_modified"]:
                        reward_list.append(0.5)
                    else:
                        reward_list.append(1.0)

                    # reward_list.append(1.0)
                else:
                    # correctness = False
                    reward_list.append(0.0)
            
            rewards = torch.tensor(reward_list)

            # prepare the input and output text
            input_texts = [curr_record["input"] for curr_record in curr_batch]
            output_texts = [curr_record["output"] for curr_record in curr_batch]

            assert len(input_texts) == len(output_texts), "Input and output texts must have the same length."

            merged_texts = [input_text + output_text for input_text, output_text in zip(input_texts, output_texts)]

            # tokenize the input and output texts
            tokenized_content = tokenizer(
                merged_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
                add_special_tokens=False,
            ).to(torch.device("cpu"))

            prompt_lens = torch.Tensor([
                len(tokenizer(curr_input_text, add_special_tokens=False)["input_ids"])
                for curr_input_text in input_texts
            ])
            
            valid_lens = tokenized_content["attention_mask"].sum(dim=-1)

            B, L = tokenized_content["input_ids"].shape
            arange = torch.arange(L).unsqueeze(0).expand(B, L)

            loss_mask = (arange >= prompt_lens.unsqueeze(1)) & (arange < valid_lens.unsqueeze(1))
            loss_mask = loss_mask.to(tokenized_content["input_ids"].dtype)   

            loss_mask = loss_mask * tokenized_content["attention_mask"]

            # obtain log probabilities for the old model and the reference model
            with torch.no_grad():
                try:
                    print(f"begin calculating log probabilities for {thm_name} batch {i + 1}/{len(curr_batched_training_data)}")
                    print(f"current input ids shape: {tokenized_content['input_ids'].shape}")
                    if parallel_compute:
                        old_model_log_prob, ref_model_log_prob = _stream_compute_log_probs(
                            old_model=old_model, 
                            ref_model=ref_model, 
                            tokenized_content=tokenized_content
                        )
                    else:
                        old_model_log_prob = _obtain_log_probs(old_model, tokenized_content["input_ids"].to(old_model.device)).to(torch.device("cpu"))
                        ref_model_log_prob = _obtain_log_probs(ref_model, tokenized_content["input_ids"].to(ref_model.device)).to(torch.device("cpu"))

                except Exception as e:
                    if "CUDA out of memory" in str(e):
                        print(f"⚠️ CUDA OOM detected on batch {i+1}, fallback to single-sample mode")

                        old_model_log_prob_list = []
                        ref_model_log_prob_list = []

                        for j in range(tokenized_content["input_ids"].size(0)):
                            try:
                                curr_input_ids = tokenized_content["input_ids"][j:j+1]
                                if parallel_compute:
                                    curr_old_model_log_prob, curr_ref_model_log_prob = _stream_compute_log_probs(
                                        old_model=old_model, 
                                        ref_model=ref_model, 
                                        tokenized_content={
                                            "input_ids": curr_input_ids,
                                            "attention_mask": tokenized_content["attention_mask"][j:j+1]
                                        }
                                    )
                                else:
                                    curr_old_model_log_prob = _obtain_log_probs(
                                        old_model, 
                                        curr_input_ids.to(old_model.device)
                                    ).to(torch.device("cpu"))
                                    curr_ref_model_log_prob = _obtain_log_probs(
                                        ref_model, 
                                        curr_input_ids.to(ref_model.device)
                                    ).to(torch.device("cpu"))
                                old_model_log_prob_list.append(curr_old_model_log_prob)
                                ref_model_log_prob_list.append(curr_ref_model_log_prob)
                                del curr_input_ids, curr_old_model_log_prob, curr_ref_model_log_prob
                                torch.cuda.empty_cache()
                                gc.collect()
                            except RuntimeError as inner_e:
                                print(f"❌ OOM on single sample {j} of batch {i+1}")
                                raise inner_e  
                        old_model_log_prob = torch.cat(old_model_log_prob_list, dim=0)
                        ref_model_log_prob = torch.cat(ref_model_log_prob_list, dim=0)
                    else:
                        raise e  

            # append the current batch data to the list
            curr_training_batched_data.append(copy.deepcopy({
                "input": input_texts,
                "output": output_texts,
                "tokenized_content": tokenized_content,
                "loss_mask": loss_mask,
                "old_model_log_prob": old_model_log_prob,
                "ref_model_log_prob": ref_model_log_prob, 
                "rewards": rewards.to(torch.device("cpu"))  # move rewards to CPU for saving
            }))
            del tokenized_content, loss_mask, ref_model_log_prob, old_model_log_prob
            torch.cuda.empty_cache()
            gc.collect()
        # save the pre-computed log probabilities to the ckpt file
        curr_data_record_to_save = {
            "name": thm_name,
            "training_data": curr_training_batched_data
        }

        with open(f"{ckpt_path}/{thm_name}_training_data.pkl", "wb") as f:
            pickle.dump(curr_data_record_to_save, f)

def _precompute_log_probs_proofs_worker(
    gpu_idx_pair: Tuple[int, int],
    dataset_path: str,
    start_idx: int,
    end_idx: int,
    rank: int,
    ckpt_root: str,
    old_model_path: str,
    ref_model_path: str,
    batch_size: int = 4,
    max_length: int = 16384,
    parallel_compute: bool = True,
):
    import os, sys, json, time, gc
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

    # 日志重定向（尽量早）
    worker_root = os.path.join(ckpt_root, f"worker_{rank}")
    os.makedirs(worker_root, exist_ok=True)
    log_fp = open(os.path.join(worker_root, f"worker_{rank}.log"), "w", buffering=1, encoding="utf-8")
    err_fp = open(os.path.join(worker_root, f"worker_{rank}.err"), "w", buffering=1, encoding="utf-8")
    sys.stdout = log_fp
    sys.stderr = err_fp

    print(f"[Worker-{rank}] GPUs(physical)={gpu_idx_pair}, slice=[{start_idx}:{end_idx})")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch, torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    dev_old = torch.device(f"cuda:{gpu_idx_pair[0]}")
    dev_ref = torch.device(f"cuda:{gpu_idx_pair[1]}")

    full = utils.read_from_json(dataset_path)
    dataset_chunk = full[start_idx:end_idx]
    print(f"[Worker-{rank}] chunk size = {len(dataset_chunk)}")

    print(f"[Worker-{rank}] loading models...")
    tokenizer = AutoTokenizer.from_pretrained(old_model_path)
    old_model = AutoModelForCausalLM.from_pretrained(old_model_path, torch_dtype=torch.bfloat16).to(dev_old)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, torch_dtype=torch.bfloat16).to(dev_ref)
    old_model.eval(); ref_model.eval()

    hb_path = os.path.join(worker_root, "heartbeat.json")
    worker_ckpt_path = os.path.join(worker_root, "proofs_log_prob_ckpt_path")
    utils.check_folder_exit(worker_ckpt_path)

    try:
        pre_compute_log_probs_proofs(
            old_model=old_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset_to_compute=dataset_chunk,
            batch_size=batch_size,
            ckpt_path=worker_ckpt_path,
            max_length=max_length,
            parallel_compute=parallel_compute,
        )
        with open(hb_path, "w") as f:
            json.dump({"done": True, "ts": time.time(), "slice": [start_idx, end_idx]}, f)
        print(f"[Worker-{rank}] finished ✅")
    finally:
        del old_model, ref_model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        log_fp.close()
        err_fp.close()


def node_worker_precompute_log_probs_proofs(
    gpu_idx_pair_ls: List[Tuple[int, int]],
    worker_idx: List[int],
    dataset_path: str,
    ckpt_root_path: str,
    old_model_path: str,
    ref_model_path: str,
    batch_size: int = 4,
    max_length: int = 16384,
    shuffle_data: bool = True,
    parallel_compute: bool = True,  
):
    assert len(gpu_idx_pair_ls) == len(worker_idx), "The length of gpu_idx_pair_ls must equal worker_idx."

    full = utils.read_from_json(dataset_path)
    if shuffle_data:
        random.shuffle(full)
        os.makedirs(ckpt_root_path, exist_ok=True)
        shuffled_path = os.path.join(ckpt_root_path, "_shuffled_prover_dataset.json")
        utils.write_to_json(shuffled_path, full)
        dataset_use = shuffled_path
    else:
        dataset_use = dataset_path

    num_samples = len(full)
    n_workers = len(worker_idx)
    print(f"Total number of training data: {num_samples}, split into {n_workers} workers.")

    chunk_size = math.ceil(max(1, num_samples) / max(1, n_workers))

    ctx = multiprocessing.get_context("spawn")
    processes = []

    def _signal_handler(sig, frame):
        print(f"Received signal {sig}, terminating all workers...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                print(f"Worker {p.pid} terminated.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    for i, (rank, gpu_pair) in enumerate(zip(worker_idx, gpu_idx_pair_ls)):
        start = i * chunk_size
        end   = min((i + 1) * chunk_size, num_samples)
        p = ctx.Process(
            target=_precompute_log_probs_proofs_worker,
            args=(gpu_pair, dataset_use, start, end, rank, ckpt_root_path,
                  old_model_path, ref_model_path, batch_size, max_length, parallel_compute),
            name=f"GRPO-Precompute-Prover-Worker-{rank}",
        )
        # 关键：不要 daemon
        p.daemon = False
        p.start()
        processes.append(p)
        print(f"Started worker {rank} on GPUs {gpu_pair} with PID {p.pid}, slice=[{start}:{end})")
        time.sleep(0.5)

    for p in processes:
        p.join()

    print("All workers finished. Pre-computation of log probabilities for proofs completed.")



# ============================================= Optimizer and Evaluation =============================================

def get_optimizer(model_paras,
                  optimizer_name="adamw",
                  lr=1E-5,
                  weight_decay=1E-6):
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model_paras, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adagrad':
        return torch.optim.Adagrad(model_paras, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model_paras, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(model_paras, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optiizer {optimizer_name} is not supported")


@torch.no_grad()
def evaluate(model,
             accelerator: Accelerator,
             data_loader: DataLoader):
    """
    Evaluate the model performance on dataset `data_loader`
    During the GRPO stage, we will use the SFT dataset to perform direct code filling based on the text, it is the same function as the SFT evaluation.
    :param model: The model to evaluate, it should be a PyTorch model.
    :param accelerator: The accelerator to use for evaluation, it should be an instance of `Accelerator`.
    :param data_loader: The data loader to use for evaluation, it should be a PyTorch DataLoader instance. Each batch should be a dictionary with the following keys:
        - "input_ids": The input ids of the text, which is the tokenized input
        - "labels": The labels of the text, which is the tokenized output
        - "attention_mask": The attention mask of the text, which is used to mask the padding tokens in the input.
    :return: The average loss of the model on the dataset `data_loader`.
    """
    torch_rng_state = torch.get_rng_state()

    model.eval()
    sum_loss = 0.0

    for batch in data_loader:
        batch = batch.to(accelerator.device)
        x_batch = batch["input_ids"]
        y_batch = batch["labels"]
        attention_mask = batch["attention_mask"]
        loss = model(input_ids=x_batch,
                     labels=y_batch,
                     attention_mask=attention_mask,
                     return_dict=True).loss
        sum_loss += accelerator.gather(loss).detach().cpu().mean()
    num_batch = len(data_loader)
    eval_loss = sum_loss / float(num_batch)

    torch.set_rng_state(torch_rng_state)
    return eval_loss
    
def make_tqdm(accelerator, list_data):
    tqdm = partial(original_tqdm, disable=not accelerator.is_local_main_process, position=0)
    return tqdm(list_data)

def logging_stat_dict(stat_dict: Dict,
                      prefix="",
                      suffix="",
                      use_wandb=False,
                      accelerator=None):
    logger = get_logger("accelerator")
    stat_str_list = [f'{prefix}']
    for k, v in stat_dict.items():
        stat_str_list += [f" {k} = {v},"]
    stat_str_list += [f"{suffix}"]

    stat_str = "".join(stat_str_list)
    logger.info(stat_str)

    if use_wandb and accelerator.is_main_process:
        wandb.log(stat_dict, step=stat_dict.get("step"))


def _calculate_advantage(
        rewards, 
        epslion=1E-9, 
        mean_r=None, 
        std_r=None
    ):

    if mean_r is not None and std_r is not None:
        return (rewards - mean_r) / (std_r + epslion)
    else:
        print("use zero for mean and std for rewards")
        mean_r = rewards.mean()
        std_r = rewards.std(unbiased=False)
        return (rewards - mean_r) / (std_r + epslion)

def infer_device(model):
    if hasattr(model, "module"):
        return infer_device(model.module)
    try:
        p = next(p for p in model.parameters() if p.device.type != "meta")
        return p.device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _obtain_log_probs_on_device(
    model: torch.nn.Module,
    generated_outputs: torch.Tensor,
    dim_to_chunk: int = 0,         
    min_chunk: int = 1,            
) -> torch.Tensor:
    device = infer_device(model)
    generated_outputs = generated_outputs.to(device, non_blocking=True)

    def _forward(ids: torch.Tensor) -> torch.Tensor:
        out = model(ids)
        logits = out.logits[:, :-1]               # [B, L-1, V]
        log_probs = F.log_softmax(logits, dim=-1) # same device as logits
        gather_idx = ids[:, 1:].unsqueeze(-1).to(log_probs.device)
        return log_probs.gather(-1, gather_idx).squeeze(-1)

    chunk_size = generated_outputs.size(dim_to_chunk)
    while True:
        try:
            if chunk_size == generated_outputs.size(dim_to_chunk):       
                return _forward(generated_outputs)
            else:                                                       
                pieces, results = [], []
                for sl in torch.split(generated_outputs, chunk_size, dim=dim_to_chunk):
                    results.append(_forward(sl))
                    torch.cuda.empty_cache()            
                return torch.cat(results, dim=dim_to_chunk)
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise                                   
            torch.cuda.empty_cache()                    
            if chunk_size <= min_chunk:
                raise                                   
            chunk_size = 1
            if infer_device(model).type == "cuda":
                print(f"[OOM] fallback: chunk_size={chunk_size}")

def _log_probs_cross_entropy(model, input_ids):
    logits = model(input_ids).logits[:, :-1]      # (B, T-1, V)
    targets = input_ids[:, 1:]                    # (B, T-1)

    try:
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none"
        )  # (B * (T-1))
        return -nll.view_as(targets)              # (B, T-1)
    
    except Exception as e:
        if "out of memory" in str(e):
            print("[Warning] CUDA OOM in cross_entropy. Fallback to per-sample computation.")
            torch.cuda.empty_cache()

            # 只 fallback cross_entropy
            results = []
            for i in original_tqdm(range(logits.size(0))):
                single_logits = logits[i]         # (T-1, V)
                single_targets = targets[i]       # (T-1,)
                nll = F.cross_entropy(
                    single_logits,
                    single_targets,
                    reduction="none"
                ).to(torch.device("cpu"))  # (T-1,)
                results.append(-copy.deepcopy(nll))
                del single_logits, single_targets, nll
                torch.cuda.empty_cache()
                gc.collect()

            return torch.stack(results, dim=0)     # (B, T-1)
        else:
            raise e


def _obtain_log_probs(model, generated_outputs):
    output_logits = model(generated_outputs).logits[:, :-1].to(torch.device("cpu"))
    generated_outputs = generated_outputs.to(torch.device("cpu"))
    log_probs = F.log_softmax(output_logits, dim=-1)
    expanded_indices = generated_outputs[:, 1:].unsqueeze(-1)
    selected_log_probs = log_probs.gather(dim=-1, index=expanded_indices).squeeze(-1)
    return selected_log_probs

def _calculate_KL_divergence(log_pi_theta, log_pi_ref):
    """
    This function calculates the KL divergence between the reference model and the current model results. The inputs are expected 
    to be the log probabilities.

    Args:
        log_pi_theta (torch.Tensor): Log probabilities from the current model.
        log_pi_ref (torch.Tensor): Log probabilities from the reference model.
    
    Returns:
        torch.Tensor: KL divergence values.
    """    
    diff = log_pi_theta - log_pi_ref
    return torch.exp(diff) - diff - 1 

def _masked_mean(tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)
    else:
        return (tensor * mask).sum(dim=dim) / mask.sum(axis=dim).clamp(min=1e-9)
    
def broadcast_advantage(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[0] == b.shape[0], "First dimension must match."
    expand_shape = (a.shape[0],) + (1,) * (b.dim() - 1)
    return a.view(*expand_shape)


def GRPO_loss(
        log_pi_theta: torch.Tensor, 
        log_pi_ref: torch.Tensor,
        log_pi_old: torch.Tensor,
        loss_mask: torch.Tensor,
        rewards: torch.Tensor,
        eps_clip: float = 0.2, 
        beta_kl: float = 0.01, 
        mean_r: Optional[float] = None,
        std_r: Optional[float] = None
) -> torch.Tensor:
    """
    This is the function to calculate the GRPO loss based on the implementation of https://huggingface.co/docs/trl/main/en/grpo_trainer
    This function is only for loss calculation, and does not include the backward pass or optimizer step.
    :param log_pi_theta: Log probabilities from the current model (normally we use cached result).
    :param log_pi_ref: Log probabilities from the reference model, which is the base model for training (normally we use cached result).
    :param log_pi_old: Log probabilities from the model of the last step, used for doing clipping to make sure the loss is stable.
    :param loss_mask: A mask to indicate which position of data should be considered in the loss calculation. Typically, we will ignore the prompt tokens as well as the padding tokens in the mask. Their mask will be set to 0.
    :param rewards: The real-world reward values, which is obtained from the correctness of the generated outputs. It will be used to calculate the advantage for GRPO training.
    :param eps_clip: The clipping value for the KL divergence, which is used to make sure the loss is stable.
    :param beta_kl: The weight for the KL divergence term in the loss function.

    :return: The loss value for the GRPO training, it should be a torch tensor with only one value.
    """
    # Make sure all the log probabilities are in the same shape
    assert log_pi_theta.shape == log_pi_ref.shape == log_pi_old.shape, "All log probabilities must have the same shape."
    assert rewards.shape[0] == log_pi_theta.shape[0], "Rewards must have the same first dimension as log probabilities."

    advantage = _calculate_advantage(rewards, mean_r=mean_r, std_r=std_r)

    # DEBUG: Try the reverse KL to see if there is better result
    kl_div = _calculate_KL_divergence(log_pi_ref, log_pi_theta)
    # kl_div = _calculate_KL_divergence(log_pi_theta, log_pi_ref)
    ratio = (log_pi_theta - log_pi_old).exp()
    surr1 = ratio * broadcast_advantage(advantage, ratio)
    surr2 = ratio.clamp(1 - eps_clip, 1 + eps_clip) * broadcast_advantage(advantage, ratio)

    loss = -torch.min(surr1, surr2) + beta_kl * kl_div
    loss = _masked_mean(loss, mask=loss_mask, dim=-1)

    return loss.mean()


def _data_sorting(dataset: List[Dict[str, str]], key: str = "output"):
    """
    This function sorts the dataset based on the length of the specified key in increasing order.
    """
    return sorted(dataset, key=lambda x: len(x[key]), reverse=False)


def save_model(accelerator: Accelerator,
               accelerate_model,
               tokenizer,
               save_dir):
    accelerator.wait_for_everyone()
    accelerator.print(f"saving model at {save_dir}")
    unwrapped_model = accelerator.unwrap_model(accelerate_model)
    unwrapped_model.save_pretrained(save_dir,
                                    is_main_process=accelerator.is_main_process,
                                    save_function=accelerator.save,
                                    state_dict=accelerator.get_state_dict(accelerate_model),
                                    max_shard_size="8GB")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_dir)

def evaluate_checkpointing_and_logging(
        model, 
        tokenizer: PreTrainedTokenizer,
        global_step: int,
        start_time, 
        accelerator: Accelerator,
        eval_loader: DataLoader, 
        eval_freq: int = 128, 
        use_wandb: bool = True,
        ckpt_freq: int = 4096, 
        ckpt_path: str = "./ckpt",
        model_name: str = "GRPO_trained_model"
):
    """
    This function evaluates the model and logs the evaluation results. It also saves the model checkpoint if the global step is a multiple of `ckpt_freq`.
    :param model: The model to evaluate, it should be a PyTorch model.
    :param tokenizer: The tokenizer to use for the model, it should be a PreTrainedTokenizer instance.
    :param global_step: The current global step of the training process. In GRPO, the global step is the number of total training steps for current model, including previous GRPO generation steps.
    :param start_time: The start time of the training process, used for logging the elapsed time.
    :param accelerator: The accelerator to use for evaluation, which typically used for multi-gpu training, it should be an instance of `Accelerator`.
    :param eval_loader: The data loader to use for evaluation, it should be a PyTorch DataLoader instance. Each batch should be a dictionary with the following keys:
        - "input_ids": The input ids of the text, which is the tokenized input
        - "labels": The labels of the text, which is the tokenized output
        - "attention_mask": The attention mask of the text, which is used to mask the padding tokens in the input.
    :param eval_freq: The frequency of evaluation, if the global step is a multiple of this value, the model will be evaluated.
    :param use_wandb: Whether to use wandb for logging the evaluation results, if set to True, the evaluation results will be logged to wandb.
    :param ckpt_freq: The frequency of checkpointing, if the global step is a multiple of this value, the model will be saved to the checkpoint path.
    :param ckpt_path: The path to save the model checkpoint, it should be a folder that exists before the function is called.
        If the folder does not exist, it will be created.
    :param model_name: The name of the model, which will be used to save the model checkpoint.
    """

    if global_step < 0 or (global_step % eval_freq == 0):
        eval_loss = evaluate(model,
                             accelerator,
                             eval_loader)
        prefix = f"At the beginning of i = {global_step}:"
        suffix = " (evaluation mode of model)"
        stat_dict = {'validation loss': eval_loss,
                     "step": global_step,
                     "time": time.time() - start_time}
        logging_stat_dict(stat_dict,
                          prefix,
                          suffix,
                          use_wandb=use_wandb,
                          accelerator=accelerator)
    if global_step != 0 and (global_step % ckpt_freq == 0):
        curr_ckpt_dir = f"{ckpt_path}/{model_name}-step-{global_step}"
        model.eval()
        if accelerator.is_main_process:
            accelerator.print(f"Saving checkpoint at step {global_step} to {curr_ckpt_dir}")
            save_model(accelerator, model, tokenizer, curr_ckpt_dir)


class ListDictDataset(TorchDataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

def preparing_training_dataloader_NL_fuser(
        train_dataset: List[Dict], 
        micro_batch_size_per_process: int = 1,
        num_workers: int = 0,
        use_data_sorting: bool = False,
        begin_idx: int = 0
):
    """
    This function preprocess the training dataset of NL fuser model to make the batch size fits the local batch size for each machine, we do not do padding here.
    :param train_dataset: The dataset for training, it should be a list of dict that has already been batched to fit the GRPO training.
        We will use the customized data loader for the training, the inner dict contains the following keys:
        - "name": The name of the theorem, which is a sha256 code used to uniquely identify the theorem.
        - "input": The input text for the model, which is the prompt together with any necessary formatting string like the generation header.
        - "output": The output text of the model, which is the generated content.
        - "tokenized_content": The tokenized merge text of both inputs and outputs, which is used for training, it is a dict with the following keys:
            - "input_ids": The padded input ids for the model, which is the tokenized merge text of both inputs and outputs, we assume it is padded to the same length for all samples in the batch.
            - "attention_mask": The attention mask for the padded input ids, which is used to mask the padding tokens in the input.
        - "loss_mask": The loss mask for the merged text, it is used to mask all the padding tokens as well as input tokens for the training process.
            Note that the shape of the loss mask should be the same as `input_ids` and `attention_mask`
        - "old_model_log_prob": The old model's log prob for GRPO training, which is the log prob from the last round of training, in step0, it should be the same as reference model.
        - "ref_model_log_prob": The ref model's log prob, which is the log prob from the base model, it is used to calculate the KL divergence for GRPO training.
            Note that both `old_model_log_prob` and `ref_model_log_prob` should have one token difference compared to `loss_mask` beacuse we need to ignore the first input token to perform causality masking.
        - "rewards": The rewards for the training data, which is a 1D torch tensor with the same length as the batch size, it is used to calculate the advantage for GRPO training.
    """
    subbatches = []
    # precompute mean and std of rewards for each theorem
    rewards = []
    for curr_theorem_record in train_dataset:
        rewards.append(curr_theorem_record["rewards"])
    rewards = torch.cat(rewards)
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False)

    for curr_theorem_record in train_dataset:
        input_ids = curr_theorem_record["tokenized_content"]["input_ids"]
        attention_mask = curr_theorem_record["tokenized_content"]["attention_mask"]
        loss_mask = curr_theorem_record["loss_mask"]
        log_old = curr_theorem_record["old_model_log_prob"]
        log_ref = curr_theorem_record["ref_model_log_prob"]
        rewards = curr_theorem_record["rewards"]

        B = input_ids.size(0)
        assert B % micro_batch_size_per_process == 0, \
            f"B={B} 不能被 micro_batch_size_per_process={micro_batch_size_per_process} 整除"
    
        for i in range(0, B, micro_batch_size_per_process):
            j = i + micro_batch_size_per_process
            subbatches.append({
                "tokenized_content": {
                    "input_ids":      input_ids[i:j],
                    "attention_mask": attention_mask[i:j],
                },
                "labels":              input_ids[i:j],
                "loss_mask":           loss_mask[i:j],
                "old_model_log_prob":  log_old[i:j],
                "ref_model_log_prob":  log_ref[i:j],
                "rewards":             rewards[i:j],
                "input":  curr_theorem_record.get("input",  None),
                "output": curr_theorem_record.get("output", None),
                "mean_reward": mean_r,
                "std_reward": std_r,
            })
    if use_data_sorting:
        subbatches = sorted(subbatches, key=lambda x: x["loss_mask"].shape[1], reverse=False)
    subbatches = subbatches[begin_idx:]  # skip the first `begin_idx` items
    print(f"Total number of subbatches: {len(subbatches)}")
    def identity_collate(batch):
        return batch[0]
    dl = DataLoader(
        ListDictDataset(subbatches),
        batch_size=1, 
        shuffle=False,
        collate_fn=identity_collate,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    return dl



def preparing_training_dataloader_prover(
        train_dataset: List[Dict],
        micro_batch_size_per_process: int = 1,
        num_workers: int = 0, 
        use_data_sorting=False, 
        begin_idx = 0
):
    """
    This function preprocesses the training dataset of prover model to make the batch size fits the local batch size for each machine, we do not do padding here.
    :param train_dataset: The dataset for training, it should be a list of dict that has already been batched to fit the GRPO training.
        We will use the customized data loader for the training, the inner dict contains the following keys:
        - "name": The name of the theorem, which is a sha256 code used to uniquely identify the theorem.
        - "training_data": The precomputed training data, which is a list of dict that has already been batched to fit the GRPO training, the inner dict contains the following keys:
            - "input": The list of text inputs for the model, which is the prompt together with any necessary formatting string like the generation header. Note that in here, this input is only for reference, it is not used in the training process.
            - "output": The list of text outputs of the model, which is the generated content. Note that in here, this output is only for reference, it is not used in the training process.
            - "tokenized_content": The tokenized merge text of both inputs and outputs, which is used for training, it is a dict with the following keys:
                - "input_ids": The padded input ids for the model, which is the tokenized merge text of both inputs and outputs, we assume it is padded to the same length for all samples in the batch.
                - "attention_mask": The attention mask for the padded input ids, which is used to mask the padding tokens in the input.
            - "loss_mask": The loss mask for the merged text, it is used to mask all the padding tokens as well as input tokens for the training process.
                Note that the shape of the loss mask should be the same as `input_ids` and `attention_mask`
            - "old_model_log_prob": The old model's log prob for GRPO training, which is the log prob from the last round of training, in step0, it should be the same as reference model.
            - "ref_model_log_prob": The ref model's log prob, which is the log prob from the base model, it is used to calculate the KL divergence for GRPO training.
                Note that both `old_model_log_prob` and `ref_model_log_prob` should have one token difference compared to `loss_mask` beacuse we need to ignore the first input token to perform causality masking.
            - "rewards": The rewards for the training data, which is a 1D torch tensor with the same length as the batch size, it is used to calculate the advantage for GRPO training.
    :param local_batch_size: The local batch size for the training, it is the batch size for each machine.
    :param use_data_sorting: Whether sort the data
    :return: A pytorch dataloader that can be used for training, it will return a batch of dict with the following:
        - "input_ids": The input ids of the text, which is the tokenized input
        - "labels": The labels of the text, which is the tokenized output
        - "attention_mask": The attention mask of the text, which is used to mask the padding tokens in the input.
        - "loss_mask": The loss mask of the text, which is used to mask the padding tokens in the input.
        - "old_model_log_prob": The old model's log prob for GRPO training, which is the log prob from the last round of training, in step0, it should be the same as reference model.
        - "ref_model_log_prob": The ref model's log prob, which is the log prob from the base model, it is used to calculate the KL divergence for GRPO training.
        - "rewards": The rewards for the training data, which is a 1D torch tensor with the same length as the batch size, it is used to calculate the advantage for GRPO training.
    """
    training_theorems = [rec for rec in train_dataset]

    subbatches = []

    for curr_theorem_record in training_theorems:

        # calculate the mean and std of the rewards for this theorem
        rewards = []
        print(curr_theorem_record.keys())
        for big in curr_theorem_record["training_data"]:
            rewards.append(big["rewards"])
        rewards = torch.cat(rewards)
        mean_r = rewards.mean()
        std_r = rewards.std(unbiased=False)

        for big in curr_theorem_record["training_data"]:
            input_ids      = big["tokenized_content"]["input_ids"]
            attention_mask = big["tokenized_content"]["attention_mask"]
            loss_mask      = big["loss_mask"]
            log_old        = big["old_model_log_prob"]
            log_ref        = big["ref_model_log_prob"]
            rewards        = big["rewards"]

            B = input_ids.size(0)
            assert B % micro_batch_size_per_process == 0, \
                f"B={B} 不能被 micro_batch_size_per_process={micro_batch_size_per_process} 整除"

            for i in range(0, B, micro_batch_size_per_process):
                j = i + micro_batch_size_per_process
                subbatches.append({
                    "tokenized_content": {
                        "input_ids":      input_ids[i:j],
                        "attention_mask": attention_mask[i:j],
                    },
                    "labels":              input_ids[i:j],
                    "loss_mask":           loss_mask[i:j],
                    "old_model_log_prob":  log_old[i:j],
                    "ref_model_log_prob":  log_ref[i:j],
                    "rewards":             rewards[i:j],
                    "input":  big.get("input",  None),
                    "output": big.get("output", None),
                    "mean_reward": mean_r,
                    "std_reward": std_r,
                })

    if use_data_sorting:
        subbatches = sorted(subbatches, key=lambda x: x["loss_mask"].shape[1], reverse=False)
    subbatches = subbatches[begin_idx:]
    print("after reducing the batch")

    def identity_collate(batch):
        return batch[0]

    dl = DataLoader(
        ListDictDataset(subbatches),
        batch_size=1,                  
        shuffle=False,               
        collate_fn=identity_collate,  
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return dl


def GRPO_step_optimize(
        model, 
        tokenizer, 
        model_name: str, 
        GRPO_training_step: int, 
        GRPO_step: int,
        train_dataset: List[Dict], 
        eval_loader: DataLoader, 
        accelerator: Accelerator,
        data_type: str = "prover",
        global_batch_size = 8, 
        micro_batch_size = 2, 
        gradient_accumulation_steps = 4,
        max_steps = None, 
        epoch = 1, 
        optimizer_name = "adamw",
        lr = 2E-6, 
        weight_decay = 1E-8,
        eval_freq = 128,
        use_wandb = True, 
        ckpt_freq = 4096,
        ckpt_path = "./ckpt",
        save_path = "./GRPO_trained_model",
        eps_clip = 0.2,
        beta_kl = 0.01, 
        begin_idx = 0
):
    """
    This function is to perform GRPO training for one GRPO step after the model rollout, reward calculation and data preprocessing.
    :param model: The model to train, it should be either the base model or the model from the last step of GRPO training.
    :param tokenizer: The tokenizer to use for the model, it should be a PreTrainedTokenizer instance.
    :param model_name: The name of the model, which will be used to save the model checkpoint. We suggest this name to reflect the GRPO step information, e.g. "GRPO_step_1".
    :param GRPO_training_step: The total training steps for the entire training process, it is used to perform logging and checkpointing.
    :param GRPO_step: The step number for current GRPO training, it is used to manage the entir training process.
    :param train_dataset: The dataset for training, it should be a list of dict that has already been batched to fit the GRPO training.
        We will use the customized data loader for the training, the inner dict contains the following keys:
        - "name": The name of the theorem, which is a sha256 code used to uniquely identify the theorem.
        - "training_data": The precomputed training data, which is a list of dict that has already been batched to fit the GRPO training, the inner dict contains the following keys:
            - "input": The list of text inputs for the model, which is the prompt together with any necessary formatting string like the generation header. Note that in here, this input is only for reference, it is not used in the training process.
            - "output": The list of text outputs of the model, which is the generated content. Note that in here, this output is only for reference, it is not used in the training process.
            - "tokenized_content": The tokenized merge text of both inputs and outputs, which is used for training, it is a dict with the following keys:
                - "input_ids": The padded input ids for the model, which is the tokenized merge text of both inputs and outputs, we assume it is padded to the same length for all samples in the batch.
                - "attention_mask": The attention mask for the padded input ids, which is used to mask the padding tokens in the input.
            - "loss_mask": The loss mask for the merged text, it is used to mask all the padding tokens as well as input tokens for the training process.
                Note that the shape of the loss mask should be the same as `input_ids` and `attention_mask`
            - "old_model_log_prob": The old model's log prob for GRPO training, which is the log prob from the last round of training, in step0, it should be the same as reference model.
            - "ref_model_log_prob": The ref model's log prob, which is the log prob from the base model, it is used to calculate the KL divergence for GRPO training.
                Note that both `old_model_log_prob` and `ref_model_log_prob` should have one token difference compared to `loss_mask` beacuse we need to ignore the first input token to perform causality masking.
            - "rewards": The rewards for the training data, which is a 1D torch tensor with the same length as the batch size, it is used to calculate the advantage for GRPO training.
    :param eval_loader: The data loader for evaluation, it should be doing the code filling task, the data format should be the following:
        Each batch should be a dictionary with the following keys:
        - "input_ids": The input ids of the text, which is the tokenized input
        - "labels": The labels of the text, which is the tokenized output
        - "attention_mask": The attention mask of the text, which is used to mask the padding tokens in the input.
    :param accelerator: The accelerator to use for training, which typically used for multi-gpu training, it should be an instance of `Accelerator`.
    :param data_type: The type of the data, it is used to obtain training data loader from different type of preprocessing.
    :param global_batch_size: The global batch size for the training, it is the total batch size across all devices.
    :param micro_batch_size: The micro batch size for the training, it is the batch size for each device.
    :param gradient_accumulation_steps: The number of gradient accumulation steps, it is used to accumulate the gradients before performing the optimizer step.
    :param max_steps: The maximum number of steps for the training, if set to None, it will be set to the length of the train dataset divided by the global batch size.
    :param epoch: The number of epochs for the training, it is used to perform the training for multiple epochs.
    :param optimizer_name: The name of the optimizer to use for the training, it should be one of the following: "adamw", "sgd", "adagrad", "adam".
    :param lr: The learning rate for the optimizer, it is used to update the model parameters.
    :param weight_decay: The weight decay for the optimizer, it is used to prevent overfitting.
    :param eval_freq: The frequency of evaluation, if the global step is a multiple of this value, the model will be evaluated.
    :param use_wandb: Whether to use wandb for logging the evaluation results, if set to True, the evaluation results will be logged to wandb.
    :param ckpt_freq: The frequency of checkpointing, if the global step is a multiple of this value, the model will be saved to the checkpoint path.
    :param ckpt_path: The path to save the model checkpoint, it should be a folder that exists before the function is called.
        If the folder does not exist, it will be created.
    :param save_path: The path to save the final model, it should be a folder that exists before the function is called.
        If the folder does not exist, it will be created.
    :return: The trained model after one GRPO step, it should be a PyTorch model.
    """


    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if data_type == "prover":
        train_loader = preparing_training_dataloader_prover(
            train_dataset,
            micro_batch_size_per_process=micro_batch_size,
            num_workers=32,  
            use_data_sorting=True, 
            begin_idx=begin_idx * micro_batch_size * world_size
        )
    elif data_type == "NL_fuser":
        train_loader = preparing_training_dataloader_NL_fuser(
            train_dataset,
            micro_batch_size_per_process=micro_batch_size,
            num_workers=32,  
            use_data_sorting=True, 
            begin_idx=begin_idx * micro_batch_size * world_size
        )
    else:
        raise ValueError(f"Unsupported data type: {data_type}, only support 'prover' and 'NL_fuser'.")
    
    assert global_batch_size == (micro_batch_size * gradient_accumulation_steps * world_size), f"global_batch_size={global_batch_size} must equal micro_batch_size × gradient_accumulation_steps × world_size={micro_batch_size}×{gradient_accumulation_steps}×{world_size}"

    if accelerator.is_main_process:
        accelerator.print(f"max_steps: {max_steps}, grad_accu_steps: {gradient_accumulation_steps}")

        print(f"curr optimizer args: \n\toptimizer name: {optimizer_name}, \n\t"
            f"lr: {lr}, \n\t"
            f"weight_decay: {weight_decay}")
    optimizer = get_optimizer(model.parameters(),
                              optimizer_name,
                              lr=lr,
                              weight_decay=weight_decay)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    if accelerator.is_main_process:
        print("preparing dataloader, model, optimizer...")

    if accelerator.is_main_process:
        print("preparing dataloader, model, optimizer...")

    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )
    accelerator.print(model)
    start_time = time.time()
    global_step = GRPO_training_step
    

    for epoch_idx in range(epoch):
        if accelerator.is_main_process:
            print(f"begin training for epoch {epoch_idx}...")
        if hasattr(train_loader, "sampler") \
           and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch_idx)

        for step_idx, batch in enumerate(make_tqdm(accelerator, train_loader)):
            model.train()
            if accelerator.is_main_process:
                accelerator.print(f"GRPO step {GRPO_step}, epoch {epoch_idx}, global step {global_step}, batch idx {step_idx}")
            with accelerator.accumulate(model):
                if accelerator.is_main_process:
                    print(f"current data size: {batch['tokenized_content']['input_ids'].shape}")
                input_ids = batch["tokenized_content"]["input_ids"].to(accelerator.device)
                attention_mask = batch["tokenized_content"]["attention_mask"].to(accelerator.device)
                loss_mask = batch["loss_mask"].to(accelerator.device)
                loss_mask = loss_mask[:, 1:]
                rewards = batch["rewards"].to(accelerator.device)

                log_pi_theta = _obtain_log_probs_on_device(model, input_ids)

                log_pi_ref = batch["ref_model_log_prob"].to(accelerator.device)
                log_pi_old = batch["old_model_log_prob"].to(accelerator.device)

                assert loss_mask.shape[-1] == log_pi_theta.shape[-1] == log_pi_ref.shape[-1] == log_pi_old.shape[-1]

                loss = GRPO_loss(
                    log_pi_theta=log_pi_theta, 
                    log_pi_ref=log_pi_ref,
                    log_pi_old=log_pi_old,
                    loss_mask=loss_mask,
                    rewards=rewards,
                    eps_clip=eps_clip,
                    beta_kl=beta_kl, 
                    mean_r=batch["mean_reward"],
                    std_r=batch["std_reward"]
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                global_step += 1
                accelerator.print(f"[{global_step}] loss={loss.item():.4f}")
                if use_wandb and accelerator.is_main_process and wandb.run is not None and accelerator.sync_gradients:
                    wandb.log(
                        {
                            "train/loss": float(loss.detach().float().item()),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                        },
                        step=global_step,
                    )
                evaluate_checkpointing_and_logging(
                    model, tokenizer,
                    global_step, start_time,
                    accelerator, eval_loader,
                    eval_freq=eval_freq,
                    use_wandb=use_wandb,
                    ckpt_freq=ckpt_freq,
                    ckpt_path=ckpt_path,
                    model_name=model_name,
                )
            if max_steps and global_step >= max_steps:
                break

        if max_steps and global_step >= max_steps:
            break
    print("begin saving trained model")
    save_model(accelerator, model, tokenizer, save_path)
    accelerator.print(f"Finished GRPO-step {GRPO_step}. Total steps={global_step}")
    return model