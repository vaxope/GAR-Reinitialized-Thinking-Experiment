# main_RL_trainner.py
# accelerate launch --config_file fsdp_config.yaml main_RL_trainner.py > ./test.log 2> ./test.err &
# accelerate launch --config_file fsdp_config_offload_enabled.yaml main_RL_trainner.py > ./test_1.log 2> ./test_1.err &

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


def test_main_GRPO_step_training():
    """
    Testing the main function for GRPO step training.
    """

    TRAINING_IDX = "GRPO_test_training_step_0"

    train_data_path = "./test_data/precomputed_results.pkl"
    eval_data_path = "../data/eval_dataset/MiniF2F_test_partial_CoT_withProof_commented.json"


    optimizer_name = "AdamW"
    lr = 2E-6
    weight_decay = 1E-8
    max_steps = None
    epoch = 1
    global_batch_size = 32
    micro_batch_size = 1
    eval_batch_size = 1
    gradient_accumulation_steps = 8

    model_path = "../data/models/Qwen3-1.7B"
    model_name = "Qwen3-8B-testing"
    num_eval_dataloader_worker = 0

    use_bf16 = True  # whether use bf16 for training
    max_length = 8192 * 2

    # Test parameters
    eval_frequency = 20

    # Debug para
    logging_conf_file = "./conf/common.log_conf"  # The configuration file for logging
    use_wandb = True
    wandb_project = "GAR_test"
    wandb_run_name = model_name

    # Saving and ckpting
    save_path = f"./Trained_mode/{model_name}"
    ckpt_path = "./test_ckpt"
    ckpt_step = 20000000


    args_dict = {
        "train_data_path": train_data_path,
        "eval_data_path": eval_data_path,
        "num_eval_dataloader_worker": num_eval_dataloader_worker,
        "optimizer_name": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_steps": max_steps,
        "epoch": epoch,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "eval_batch_size": eval_batch_size,
        "model_path": model_path,
        "use_bf16": use_bf16,
        "max_length": max_length,
        "eval_frequency": eval_frequency,
        "logging_conf_file": logging_conf_file,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "save_path": save_path,
        "ckpt_path": ckpt_path,
        "ckpt_step": ckpt_step,
    }

    args_optimizer_dict = {
        "name": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
    }

    args = argparse.Namespace(**args_dict)

     # init logging
    logging.config.fileConfig(logging_conf_file)

    # init logging
    logging.config.fileConfig(logging_conf_file)

    # load tokenizer and model
    if use_bf16:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.bfloat16,
                                                     attn_implementation="flash_attention_2")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.float32)
        
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              add_eos_token=True,
                                              add_bos_token=True)
    # Set the Fully shared data parallel plugin
    fsdp_plugin = FullyShardedDataParallelPlugin()
    fsdp_plugin.set_mixed_precision("fp32")

    if use_bf16:
        accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=gradient_accumulation_steps,)
    else:
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps,
                                  fsdp_plugin=fsdp_plugin)
        
    train_data_ls = pickle.load(open(train_data_path, "rb"))
    eval_data_ls = utils.read_from_json(eval_data_path)

    eval_dataloader = obtain_eval_dataloader(
        eval_dataset=eval_data_ls, 
        tokenizer=tokenizer,
        max_length=4096, 
        num_workers=num_eval_dataloader_worker,
        micro_batch_size=eval_batch_size
    )

    if accelerator.is_main_process:
        print(f"current accelerator state: \n{accelerator.state.fsdp_plugin}")

    # perform optimization
    if use_wandb and accelerator.is_main_process:
        wandb.init(project=wandb_project,
                   name=wandb_run_name,
                   config=args_dict)

    # make the saving dir if not exist
    if accelerator.is_main_process:
        utils.check_folder_exit(save_path)
        utils.check_folder_exit(ckpt_path)
    
    # 禁用训练缓存（避免保存KV缓存占显存）
    model.config.use_cache = False  

    # 启用 gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 让输入要求梯度（HF要求）
    model.enable_input_require_grads()

    # 启用 FlashAttention2 或 SDPA
    model.config._attn_implementation = "flash_attention_2"
    
    try:
        if accelerator.is_main_process:
            print("begin GRPO step training...")
        GRPO_step_optimize(
            model=model, 
            model_name=model_name,
            tokenizer=tokenizer,
            GRPO_training_step=0, 
            GRPO_step=0, 
            train_dataset=train_data_ls, 
            eval_loader=eval_dataloader, 
            accelerator=accelerator,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=max_steps, 
            epoch=epoch,
            optimizer_name=optimizer_name, 
            lr=lr, 
            weight_decay=weight_decay,
            eval_freq=eval_frequency,
            use_wandb=use_wandb,
            ckpt_freq=ckpt_step,
            ckpt_path=ckpt_path,
            save_path=save_path,
            eps_clip=0.2, 
            beta_kl=0.01, 
            begin_idx=0
        )
    except Exception as e:
        if use_wandb and accelerator.is_main_process:
            wandb.finish()
        raise e

    if use_wandb and accelerator.is_main_process:
        wandb.finish()
    
    


def test_main_node_worker_precompute_log_probs_proofs():
    """
    Testing the main function for pre-computing log probabilities for proofs in a node with multiple GPUs.
    """

    gpu_idx_pair_ls = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]  # Example GPU pairs for two workers
    worker_idx = [0, 1, 2, 3, 4]  # Worker indices corresponding to the GPU pairs
    dataset_path = "./GAR_Training_data/Step_0/processed_prover_training_data.json"
    ckpt_root_path = "./proofs_log_prob_ckpt_path/Node_worker_precompute_3"
    old_model_path = "../data/models/Goedel-Prover-V2-8B"
    ref_model_path = "../data/models/Goedel-Prover-V2-8B"

    node_worker_precompute_log_probs_proofs(
        gpu_idx_pair_ls=gpu_idx_pair_ls,
        worker_idx=worker_idx,
        dataset_path=dataset_path,
        ckpt_root_path=ckpt_root_path,
        old_model_path=old_model_path,
        ref_model_path=ref_model_path,
        batch_size=4,
        max_length=16384,
        shuffle_data=True
    )

    summarize_precompute_results_prover(
        ckpt_root_path=ckpt_root_path
    )



def test_main_pre_compute_log_probs_proofs():
    """
    Testing main function for pre-computing log probabilities for proofs.
    """

    training_dataset = utils.read_from_json("./GAR_Training_data/Step_0/processed_prover_training_data.json")
    print(f"Total number of training data: {len(training_dataset)}")

    old_model = AutoModelForCausalLM.from_pretrained(
        "../data/models/Goedel-Prover-V2-8B",
        torch_dtype=torch.bfloat16
    ).to(torch.device("cuda:0"))
    ref_model = AutoModelForCausalLM.from_pretrained(
        "../data/models/Goedel-Prover-V2-8B", 
        torch_dtype=torch.bfloat16
    ).to(torch.device("cuda:1"))

    tokenizer = AutoTokenizer.from_pretrained("../data/models/Goedel-Prover-V2-8B")

    pre_compute_log_probs_proofs(
        old_model=old_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset_to_compute=training_dataset,
        batch_size=4,
        ckpt_path="./proofs_log_prob_ckpt_path/Step_0_parallel",
        max_length=16384, 
        parallel_compute=True  # Set to False for non-parallel computation test
    )


if __name__ == "__main__":
    test_main_GRPO_step_training()