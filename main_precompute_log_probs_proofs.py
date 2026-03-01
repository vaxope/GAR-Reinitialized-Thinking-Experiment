from RL_trainner import *
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
import argparse
from utils import utils
import torch
from torch.utils.data import DataLoader
import logging
import logging.config
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
from EvalDataPreprocess import obtain_eval_dataloader
import ast
from dataclasses import dataclass


def main():
    parser = argparse.ArgumentParser(description="Main program for precompute log probabilities for LLM proofs")

    parser.add_argument("--gpu_idx_pair_ls", type=str, default="[(0, 1), (2, 3)]", help="List of tuples indicating GPU index pairs for model and data parallelism, e.g., '[(0, 1), (2, 3)]'")
    parser.add_argument("--worker_idx_ls", type=str, defaulet="[0, 1]", help="List of worker indices to use, it should be of the same length of python list as gpu_id, e.g., '[0, 1]'")
    parser.add_argument("--dataset_path", type=str, help="The path to the json file to precompute the log probabilities")
    parser.add_argument("--old_model_path", type=str, help="The path to the old model to compute the log probabilities")
    parser.add_argument("--ref_model_path", type=str, help="The path to the reference model to compute the log probabilities")
    parser.add_argument("--ckpt_root_path", type=str, help="The root path to save the checkpoints for each worker")
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size for each GPU, default is 4")
    parser.add_argument("--max_length", type=int, default=16384, help="The maximum length of the input sequence, default is 16384")
    parser.add_argument("--shuffle_data", type=bool, default=True, help="Whether to shuffle the data to balance worker load, default is True")

    args = parser.parse_args()

    gpu_idx_pair_ls = ast.literal_eval(args.gpu_idx_pair_ls)
    worker_idx_ls = ast.literal_eval(args.worker_idx_ls)
    dataset_path = args.dataset_path
    old_model_path = args.old_model_path
    ref_model_path = args.ref_model_path
    ckpt_root_path = args.ckpt_root_path
    batch_size = args.batch_size
    max_length = args.max_length
    shuffle_data = args.shuffle_data

    node_worker_precompute_log_probs_proofs(
        gpu_idx_pair_ls=gpu_idx_pair_ls,
        worker_idx=worker_idx_ls,
        dataset_path=dataset_path,
        ckpt_root_path=ckpt_root_path,
        old_model_path=old_model_path,
        ref_model_path=ref_model_path,
        batch_size=batch_size,
        max_length=max_length,
        shuffle_data=shuffle_data
    )

    summarize_precompute_results_prover(
        ckpt_root_path=ckpt_root_path
    )


def run_precompute_log_probs_proofs(
    gpu_idx_pair_ls: List[Tuple[int, int]],
    worker_idx: List[int],
    dataset_path: str,
    ckpt_root_path: str,
    old_model_path: str,
    ref_model_path: str,
    batch_size: int = 4,
    max_length: int = 16384,
    shuffle_data: bool = True
):
    node_worker_precompute_log_probs_proofs(
        gpu_idx_pair_ls=gpu_idx_pair_ls,
        worker_idx=worker_idx,
        dataset_path=dataset_path,
        ckpt_root_path=ckpt_root_path,
        old_model_path=old_model_path,
        ref_model_path=ref_model_path,
        batch_size=batch_size,
        max_length=max_length,
        shuffle_data=shuffle_data
    )

    summarize_precompute_results_prover(
        ckpt_root_path=ckpt_root_path
    )
    
def run_precompute_log_probs_NL_fuser(
    gpu_idx_pair_ls: List[Tuple[int, int]],
    worker_idx: List[int],
    dataset_path: str,
    ckpt_root_path: str,
    old_model_path: str,
    ref_model_path: str,
    batch_size: int = 4,
    max_length: int = 16384,
    shuffle_data: bool = True
):
    node_worker_precompute_log_probs_NL_fuser(
        gpu_idx_pair_ls=gpu_idx_pair_ls,
        worker_idx=worker_idx,
        dataset_path=dataset_path,
        ckpt_root_path=ckpt_root_path,
        old_model_path=old_model_path,
        ref_model_path=ref_model_path,
        batch_size=batch_size,
        max_length=max_length,
        shuffle_data=shuffle_data
    )

    summarize_precompute_results_NL_fuser(
        ckpt_root_path=ckpt_root_path
    )


if __name__ == "__main__":
    main()
