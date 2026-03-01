from RL_trainner import *
import argparse
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from utils import utils
import torch
from torch.utils.data import DataLoader
import logging
import logging.config
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
from EvalDataPreprocess import obtain_eval_dataloader

def main_args_version(
        # Meta config for the training
        training_idx, 
        model_path, 
        model_name, 
        wandb_run_name,
        data_type,
        # Training and evaluation data paths
        train_data_path,
        eval_data_path,
        begin_idx,
        # Optimizer parameters
        optimizer_name,
        lr,
        weight_decay,
        max_steps,
        epoch,
        global_batch_size,
        micro_batch_size,
        eval_batch_size,
        gradient_accumulation_steps,
        use_bf16,
        max_length,
        eval_frequency,
        eps_clip,
        beta_kl,
        # Debug parameters
        logging_conf_file,
        use_wandb,
        wandb_project,
        # Saving and checkpointing parameters
        save_path,
        ckpt_path,
        ckpt_step
):
    args_dict = {
        "data_type": data_type,
        "train_data_path": train_data_path,
        "eval_data_path": eval_data_path,
        "begin_idx": begin_idx,
        "model_path": model_path,
        "model_name": model_name,
        "wandb_run_name": wandb_run_name,
        "optimizer_name": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_steps": max_steps,
        "epoch": epoch,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "use_bf16": use_bf16,
        "max_length": max_length,
        "eval_frequency": eval_frequency,
        "logging_conf_file": logging_conf_file,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "eps_clip": eps_clip,
        "beta_kl": beta_kl,
        "save_path": save_path,
        "ckpt_path": ckpt_path,
        "ckpt_step": ckpt_step
    }

    args_optimizer_dict = {
        "name": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
    }

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
        num_workers=0,
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

    model.config.use_cache = False  

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model.enable_input_require_grads()

    model.config._attn_implementation = "flash_attention_2"

    try:
        if accelerator.is_main_process:
            accelerator.print("begin GRPO step training...")
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
            eps_clip=eps_clip, 
            beta_kl=beta_kl,
            begin_idx=begin_idx, 
            data_type=data_type,
        )
    except Exception as e:
        if use_wandb and accelerator.is_main_process:
            wandb.finish()
        raise e

    if use_wandb and accelerator.is_main_process:
        wandb.finish()



def main():
    parser = argparse.ArgumentParser(description="Main program for RL training")

    # Meta config for the training
    parser.add_argument("--training_idx", type=str, help="The index for training, it will be the prefix of saving model")
    parser.add_argument("--model_path", type=str, help="The path to the model to train, it can be a local path or a HuggingFace model id")
    parser.add_argument("--model_name", type=str, help="The name of the model, it will be represented in the wandb project and the save path")
    parser.add_argument("--wandb_run_name", type=str, help="The name of the wandb run, it will be used to identify the run in the wandb project")
    parser.add_argument("--data_type", type=str, default="prover", help="The training mode, it can be prover or NL_fuser, default is prover")
    
    # Training and evaluation data paths
    parser.add_argument("--train_data_path", type=str, help="The path to the training data, details of the data format can be found in RL_trainner.py")
    parser.add_argument("--eval_data_path", type=str, help="The path to the evaluation data, details of the data format can be found in RL_trainner.py")
    parser.add_argument("--begin_idx", type=int, default=0, help="The begin index for the training data, default is 0")

    # Optimizer parameters
    parser.add_argument("--optimizer_name", type=str, default="AdamW", help="The which type of optimizer use, default is AdamW")
    parser.add_argument("--lr", type=float, default=2E-6, help="The learning rate for the optimizer, default is 2E-6")
    parser.add_argument("--weight_decay", type=float, default=1E-8, help="The weight decay for the optimizer, default is 1E-8")
    parser.add_argument("--max_steps", type=int, default=None, help="The maximum steps for training, if None, it will be set to the length of the training data")
    parser.add_argument("--epoch", type=int, default=1, help="The number of epochs for training, default is 1")
    parser.add_argument("--global_batch_size", type=int, default=32, help="The global batch size for training, default is 32")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="The micro batch size for training, default is 1")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="The evaluation batch size, default is 1")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="The number of gradient accumulation steps, default is 8")
    parser.add_argument("--use_bf16", type=bool, default=True, help="Whether to use bf16 for training, default is True")
    parser.add_argument("--max_length", type=int, default=16384, help="The maximum length of the input sequence, default is 16384")
    parser.add_argument("--eval_frequency", type=int, default=20, help="The frequency of evaluation during training, default is 20")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="The epsilon clip for the PPO algorithm, default is 0.2")
    parser.add_argument("--beta_kl", type=float, default=0.01, help="The beta for the KL divergence term in the PPO algorithm, default is 0.01")

    # Debug parameters
    parser.add_argument("--logging_conf_file", type=str, default="./conf/common.log_conf", help="The configuration file for logging, default is ./conf/common.log_conf")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Whether to use wandb for logging, default is True")
    parser.add_argument("--wandb_project", type=str, default="GAR_test", help="The name of the wandb project, default is GAR_test")

    # Saving and checkpointing parameters
    parser.add_argument("--save_path", type=str, default="./Trained_mode", help="The path to save the trained model, default is ./Trained_mode")
    parser.add_argument("--ckpt_path", type=str, default="./test_ckpt", help="The path to save the checkpoints, default is ./test_ckpt")
    parser.add_argument("--ckpt_step", type=int, default=20000000, help="The step to save the checkpoint, default is 20000000")

    args = parser.parse_args()
    

    training_idx = args.training_idx
    eval_data_path = args.eval_data_path
    model_path = args.model_path
    model_name = args.model_name
    wandb_run_name = args.wandb_run_name
    data_type = args.data_type

    train_data_path = args.train_data_path
    eval_data_path = args.eval_data_path
    begin_idx = args.begin_idx
    optimizer_name = args.optimizer_name
    lr = args.lr
    weight_decay = args.weight_decay
    max_steps = args.max_steps
    epoch = args.epoch
    global_batch_size = args.global_batch_size
    micro_batch_size = args.micro_batch_size
    eval_batch_size = args.eval_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    use_bf16 = args.use_bf16 
    max_length = args.max_length
    eval_frequency = args.eval_frequency
    eps_clip = args.eps_clip
    beta_kl = args.beta_kl

    logging_conf_file = args.logging_conf_file
    use_wandb = args.use_wandb
    wandb_project = args.wandb_project
    save_path = f"{args.save_path}/"
    ckpt_path = args.ckpt_path
    ckpt_step = args.ckpt_step

    main_args_version(
        training_idx=training_idx, 
        model_path=model_path, 
        model_name=model_name, 
        wandb_run_name=wandb_run_name,
        data_type=data_type,
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        begin_idx=begin_idx,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        max_steps=max_steps,
        epoch=epoch,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_bf16=use_bf16,
        max_length=max_length,
        eval_frequency=eval_frequency,
        eps_clip=eps_clip,
        beta_kl=beta_kl,
        logging_conf_file=logging_conf_file,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        save_path=save_path,
        ckpt_path=ckpt_path,
        ckpt_step=ckpt_step
    )

if __name__ == "__main__":
    main()


