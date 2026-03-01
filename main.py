from utils import utils
import sys
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from GAR_NodeWorker import GAR_ForwardWorker, GAR_ForwardWorkerConfig, GAR_NodeWorker
from StatementFuser import StatementFuserMetaConfig, NLStatementFuserConfig, AutoformalizerConfig, StatementCheckerConfig
from GAR_ForwardWorker import GAR_ForwardWorker, GAR_ForwardWorkerMetaConfig, ProveWriterConfig, ProofVerifierConfig, StatementFuserConfig, ProveWriterSLURMConfig
from RL_data_preprocessor import RL_DataPreprocessor
from transformers import AutoModelForCausalLM, AutoTokenizer
import signal
import multiprocessing
import subprocess
from RL_trainner import (
    summarize_precompute_results_prover,
    summarize_precompute_results_NL_fuser
)
from main_RL_trainner_GRPO import main_args_version
from main_precompute_log_probs_proofs import (
    run_precompute_log_probs_proofs, 
    run_precompute_log_probs_NL_fuser
)
from GAR_trainer import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="GAR Training Pipeline")
    
    # GAR_TrainerMetaConfig arguments
    parser.add_argument("--training_name", type=str, default="GAR-Prover",
                        help="Name of the training session")
    parser.add_argument("--base_data_path", type=str, 
                        default="./data/3base_data/NuminaMath_Lean_dataset.json",
                        help="Path to the base dataset")
    parser.add_argument("--save_folder", type=str, 
                        default="./GAR_save_files/GAR-Prover",
                        help="Folder to save training outputs")
    parser.add_argument("--total_steps", type=int, default=5,
                        help="Total number of GAR training steps")
    parser.add_argument("--theorems_fused_per_step", type=int, default=1024,
                        help="Number of theorems fused per step ")
    parser.add_argument("--proofs_per_theorem", type=int, default=16,
                        help="Number of proofs generated per theorem")
    parser.add_argument("--NL_FuserMaxPassRate", type=float, default=0.4,
                        help="Maximum pass rate for NL fuser training data")
    parser.add_argument("--ProverMaxPassRate", type=float, default=0.5,
                        help="Maximum pass rate for prover training data")
    
    # Lean server config arguments
    parser.add_argument("--lean_main_file_name", type=str, default="LeanChecker.py",
                        help="Main file name for Lean checker")
    parser.add_argument("--lean_server_url", type=str, default="http://localhost:80",
                        help="URL of the Lean4 server")
    parser.add_argument("--lean_batch_size", type=int, default=256,
                        help="Batch size for Lean verification")
    parser.add_argument("--lean_timeout", type=int, default=240,
                        help="Timeout for Lean verification")
    
    # GAR_TrainerForwardConfig arguments
    parser.add_argument("--gpu_utilization_rate", type=float, default=0.8,
                        help="GPU memory utilization rate")
    parser.add_argument("--prove_writer_batch_size", type=int, default=1024,
                        help="Batch size for prove writer")
    
    # CUDA devices
    parser.add_argument("--avail_cuda_devices", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="List of available CUDA device IDs")
    
    # Model paths
    parser.add_argument("--NL_fuser_model_id", type=str, 
                        default="Qwen/Qwen3-8B",
                        help="Path to NL fuser model")
    parser.add_argument("--autoformalizer_model_id", type=str, 
                        default="Goedel-LM/Goedel-Formalizer-V2-8B",
                        help="Path to autoformalizer model")
    parser.add_argument("--prove_writer_model_id", type=str, 
                        default="Goedel-LM/Goedel-Prover-V2-8B",
                        help="Path to prove writer model")
    parser.add_argument("--ref_model_NL_fuser", type=str, 
                        default="Qwen/Qwen3-8B",
                        help="Path to reference NL fuser model")
    parser.add_argument("--ref_model_prove_writer", type=str, 
                        default="Goedel-LM/Goedel-Prover-V2-8B",
                        help="Path to reference prove writer model")
    
    # Training control
    parser.add_argument("--step_idx", type=int, default=0,
                        help="Starting step index for training")
    
    return parser.parse_args()


def main_training(args):
    GAR_trainer_meta_config = GAR_TrainerMetaConfig(
        training_name=args.training_name, 
        base_data_path=args.base_data_path, 
        save_folder=args.save_folder, 
        total_steps=args.total_steps, 
        theorems_fused_per_step=args.theorems_fused_per_step / (len(args.avail_cuda_devices)),  
        proofs_per_theorem=args.proofs_per_theorem, 
        NL_FuserMaxPassRate=args.NL_FuserMaxPassRate,
        ProverMaxPassRate=args.ProverMaxPassRate
    )

    Lean_server_config = ProofVerifierConfig(
        main_file_name=args.lean_main_file_name, 
        Lean4_server_url=args.lean_server_url,
        batch_size=args.lean_batch_size,
        timeout=args.lean_timeout
    )

    GAR_trainer_forward_config = GAR_TrainerForwardConfig(
        gpu_utilization_rate=args.gpu_utilization_rate,       
        prove_writer_batch_size=args.prove_writer_batch_size, 
    )

    GAR_test_trainer = GAR_trainer(
        GAR_trainer_meta_config=GAR_trainer_meta_config,
        Lean_server_config=Lean_server_config,
        avail_cuda_devices=args.avail_cuda_devices,
        NL_fuser_model_id=args.NL_fuser_model_id,
        autoformalizer_model_id=args.autoformalizer_model_id,
        prover_writer_model_id=args.prove_writer_model_id, 
        GAR_trainer_forward_config=GAR_trainer_forward_config, 
        ref_model_NL_fuser=args.ref_model_NL_fuser,
        ref_model_prover_writer=args.ref_model_prove_writer,
    )

    GAR_test_trainer.run_GAR_training(step_idx=args.step_idx)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=False)
    args = parse_args()
    main_training(args)