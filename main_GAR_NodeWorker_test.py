from StatementFuser import StatementFuserMetaConfig, NLStatementFuserConfig, AutoformalizerConfig, StatementCheckerConfig, StatementFuser
from GAR_ForwardWorker import GAR_ForwardWorker, GAR_ForwardWorkerMetaConfig, ProveWriterConfig, ProofVerifierConfig, StatementFuserConfig
from GAR_NodeWorker import GAR_NodeWorker, GAR_ForwardWorkerConfig
from utils import utils
from dataclasses import dataclass
from typing import List, Dict, Any
import copy
import os
import json
import multiprocessing
import time
import signal
import sys

STEP_NUM = 1
EXPERIMENT_NAME = "GAR_NodeWorker_MagicCoder_test"
AVAIL_CUDA_DEVICES = [0,1]  
SUB_EXPERRIMENT_IDX = [0, 1]  # This is the sub-experiment index, it will be used to form the path for each experiment.
EXPERIMENT_FILE_ROOT_PATH = "./GAR_Forward_worker_test"
EXPERIMENT_IDX_LS = [f"step-{STEP_NUM}_{i}" for i in SUB_EXPERRIMENT_IDX]
# EXPERIMENT_IDX_LS = [f"step-{STEP_NUM}_4", f"step-{STEP_NUM}_5", f"step-{STEP_NUM}_6", f"step-{STEP_NUM}_7"]
GPU_UTILIZATION_RATE = 0.9
EXPERIMENT_FILE_ROOT_PATH = f"./GAR_Forward_worker_test/MagicCoder_test/Step_{STEP_NUM}/save_files"
GENERATED_STATEMENT_NUM = 32

os.environ['EXP_TYPE'] = 'MagicCoder_test'

def main():
    # Example usage
    GAR_ForwardWorker_meta_config = GAR_ForwardWorkerMetaConfig(
        experiment_name="GAR_ForwardWorker_test_run",
        save_dir_path="./GAR_ForwardWorker_test_run/test_run_0",
        base_dataset_path="../data/Base_dataset/Goedel_dataset_testing.json",
        cuda_device_id=0, 
        gpu_memory_utilization=0.95,
        generated_statement_num=GENERATED_STATEMENT_NUM,
        num_base_statement_per_fuse=2
    )

    # Config for statement fuser
    statement_fuser_meta_config = StatementFuserMetaConfig(
        base_dataset_path=GAR_ForwardWorker_meta_config.base_dataset_path,
        save_dir_path=f"{GAR_ForwardWorker_meta_config.save_dir_path}/StatementFuser_files",
        cuda_device_id=GAR_ForwardWorker_meta_config.cuda_device_id,
        generated_statement_num=GAR_ForwardWorker_meta_config.generated_statement_num,
        num_base_statement_per_fuse=GAR_ForwardWorker_meta_config.num_base_statement_per_fuse
    )

    nl_fuser_config = NLStatementFuserConfig(
        main_file_name="./NLStatementFuser.py", 
        model_id="../data/models/Qwen3-8B", 
        skip_thinking=True, 
        max_tokens=8192, 
        batch_size=16, 
        micro_batch_size=16, 
        temperature=0.8, 
        repetition_penalty=1.0, 
        top_p=0.95
    )

    autoformalizer_config = AutoformalizerConfig(
        main_file_name="./Autoformalizer.py",
        model_id="../data/models/Kimina-Autoformalizer-7B",
        max_tokens=8192,
        batch_size=16,
        temperature=0.6,
        repetition_penalty=1.0,
        top_p=0.95
    )

    statement_checker_config = StatementCheckerConfig(
        main_file_name="LeanChecker.py",
        Lean4_server_url="http://localhost:81",
        batch_size=128,
        timeout=120
    )
    statement_fuser_config = StatementFuserConfig(
        statement_fuser_meta_config=statement_fuser_meta_config,
        nl_statement_fuser_config=nl_fuser_config,
        autoformalizer_config=autoformalizer_config,
        statement_checker_config=statement_checker_config
    )

    # Prove writer config
    prove_writer_config = ProveWriterConfig(
        main_file_name="ProveWriter.py", 
        model_id="../data/models/Goedel-Prover-V2-8B",
        global_begin_idx=0, 
        global_end_idx=-1,  # -1 means all statements
        local_begin_idx=0, 
        local_end_idx=-1,  # -1 means all statements
        proof_num=16,
        temperature=0.8, 
        top_p=0.95, 
        batch_size=16, 
        max_tokens=16384
    )

    # Proof verifier config
    proof_verifier_config = ProofVerifierConfig(
        main_file_name="LeanChecker.py", 
        Lean4_server_url="http://localhost:81",
        batch_size=16,
        timeout=240
    )

    GAR_ForwardWorker_config = GAR_ForwardWorkerConfig(
        statement_fuser_config=statement_fuser_config,
        forward_worker_meta_config=GAR_ForwardWorker_meta_config,
        prove_writer_config=prove_writer_config,
        proof_verifier_config=proof_verifier_config
    )

    # Create the GAR Node Worker
    experiment_name = EXPERIMENT_NAME
    avail_cuda_devices = AVAIL_CUDA_DEVICES
    gpu_utilization_rate = GPU_UTILIZATION_RATE
    experiment_file_root_path = EXPERIMENT_FILE_ROOT_PATH
    experiment_idx_ls = EXPERIMENT_IDX_LS

    gar_node_worker = GAR_NodeWorker(
        experiment_name=experiment_name,
        avail_cuda_devices=avail_cuda_devices,
        gpu_utilization_rate=gpu_utilization_rate,
        experiment_file_root_path=experiment_file_root_path,
        experiment_idx_ls=experiment_idx_ls,
        base_config=GAR_ForwardWorker_config
    )

    # Run all workers
    gar_node_worker.run_all_workers()

if __name__ == "__main__":
    main()