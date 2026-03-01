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

# If use Goedel-Prover-V2 (Qwen3-based) use this config
PROVER_FSDP_FILENAME = "fsdp_config_Qwen3.yaml"
# If use DeepSeek-Prover-V2 (Llama-based) use this config
# PROVER_FSDP_FILENAME = "fsdp_config_Llama.yaml"
NL_FUSER_FSDP_FILENAME = "fsdp_config_offload_enabled.yaml"


# Global variable to hold the child process
child_process = None

def signal_handler(sig, frame):
    global child_process
    print(f"Received signal {sig}, terminating child process...")
    
    # If the child process is still running, terminate it
    if child_process and child_process.poll() is None:
        child_process.terminate()
        try:
            child_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Child process did not terminate, forcing kill...")
            child_process.kill()

    sys.exit(1)

def run_command(command):
    global child_process

    print(f"running command: \n{command}")

    # catch signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # activate subprocess
        child_process = subprocess.Popen(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=True,
            preexec_fn=os.setsid  # Put the child process into a new process group for unified management
        )

        # Wait for the child process to complete
        child_process.wait()

        # Check the child process return code
        if child_process.returncode != 0:
            raise subprocess.CalledProcessError(child_process.returncode, command)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command '{e.cmd}' failed with exit code {e.returncode}")

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


@dataclass
class GAR_TrainerMetaConfig:
    """
    The meta configuration for the GAR trainer.
    :param training_name: The name of the training session, which will be used as prefix for all modules.
    :param base_data_path: The path to the base data, it should be a json file in list of Dict format, each inner Dict should contains the following keys:
        - "idx": The unique index for this record of data.
        - "Name": The name of the data, it will be used sometime as prefix for saving data.
        - "Statement": The Lean4 statement for the data, we plan to add it to the statement fusion module.
        - "Proof": The verified Lean4 proof for the data.
        - "Natural_language_statement": The natural language statement for the data, it will be used for generating the fused statements.
    :param total_steps: The total GAR steps for the model, each GAR step contains the following processes:
        1. Statement Fusion: This process will try to fuse the statements in the base dataset to generate a new and more complex statement. It contains the following sub-steps:
            1. NL statement fusion: This process will merge the NL statements of the base dataset to generate a new NL statement that should be harder.
            2. Autoformalization: The process will try to autoformalize the NL statement to generate a Lean4 statement.
            3. Statement verification: This process will verify the autoformalized Lean4 statement to ensure it successfully compiles.
        2. Proof generation: Query the prover to generate multiple proofs for fused statements.
        3. Proof verification: Verify the generated proofs to obtain the correct rate and verification results of each generated proofs.
        4. Prover training: Train the prover model using GRPO algorithm using the generated proofs whose pass rate is within a specific range. It should be not too high (too easy) or 0 (too hard).
        5. Fuser training: Train the NL statement fuser model using the generated NL statements whose pass rate is within a specific range. It should be not too high (too easy) or 0 (too hard).
    """
    training_name: str
    base_data_path: str
    save_folder: str = "./GAR_model_train/"
    total_steps: int = 16
    theorems_fused_per_step: int = 1024
    num_base_statements_per_fuse: int = 2
    proofs_per_theorem: int = 16
    NL_FuserMaxPassRate:float = 0.4
    ProverMaxPassRate: float = 0.5
    train_fuser: bool = True

@dataclass
class GAR_TrainerForwardConfig:
    """
    This is the class for default forward configs in GAR trainer, typically, these parameters are the things we should not touch
    """
    gpu_utilization_rate: float = 0.95
    # NL statement fuser parameters
    NL_fuser_main_file_name: str = "./NLStatementFuser.py"
    NL_fuser_skip_thinking: bool = False
    NL_fuser_max_tokens: int = 16384
    NL_fuser_batch_size: int = 16
    NL_fuser_micro_batch_size: int = 16
    NL_fuser_temperature: float = 0.8
    # Autoformalizer parameters, we assume most parameters are set to default since it is not the main focus of GAR.
    autoformalizer_main_file_name: str = "./Autoformalizer.py"
    # ProofWriter parameters
    prove_writer_main_file_name: str = "./ProveWriter.py"
    prove_writer_max_length: int = 16385
    prove_writer_batch_size: int = 16
    prove_writer_temperature: float = 1.0
    prove_writer_slurm_config: ProveWriterSLURMConfig = field(default_factory=ProveWriterSLURMConfig)

@dataclass
class GRPO_Training_config:
    """
    This is the class for GRPO training config, it includes configs that may be changed frequently (maybe different for different steps).
    IMPORTANT: We assume the log_prob calculation of the old and ref model has already been done and stored in the training data, details can be found in RL_trainner.py
    """
    training_idx: str
    model_to_train_path: str
    training_model_name: str
    wandb_run_name: str
    train_data_path: str      # The data should be pkl that contains both training data and precomputed log probs of old and ref models
    eval_data_path: str
    model_save_path: str
    model_ckpt_path: str
    ckpt_freq: int = int(1E9)   # The frequency to save the checkpoint, default is 1E9 steps, which means no checkpoint will be saved
    begin_idx: int = 0     # The begin idx for the training data, default is 0, means use all data
    logging_conf_file: str = "./conf/common.log_conf"
    use_wandb: bool = True
    wandb_project: str = "GAR_test"


# Need name change
@dataclass
class GRPO_Training_hyperparams:
    """
    This class is for hyperparameters for GRPO training, it should not be changed frequently.
    """
    optimizer_name: str = "AdamW"
    lr: float = 5E-6
    weight_decay: float = 1E-8
    max_steps: int = int(1E9)
    epoch: int = 1
    global_batch_size: int = 32
    micro_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    use_bf16: bool = True  # whether use bf16 for training
    max_length: int = 16384
    eval_frequency: int = 40
    eps_clip: float = 0.2  
    beta_kl: float = 0.01  

def formulate_GAR_forward_node_worker(
    GAR_trainer_meta_config: GAR_TrainerMetaConfig,
    Lean_server_config: ProofVerifierConfig, 
    step_idx: int,
    avail_cuda_devices: List[int], 
    sub_experiment_idx: List[int],
    NL_fuser_model_id: str, 
    autoformalizer_model_id: str,
    prover_writer_model_id: str,
    gpu_utilization_rate: float = 0.95, 
    # NL statement fuser parameters
    NL_fuser_main_file_name: str = "./NLStatementFuser.py",
    NL_fuser_skip_thinking: bool = True,
    NL_fuser_max_tokens: int = 8192,
    NL_fuser_batch_size: int = 16,
    NL_fuser_micro_batch_size: int = 16,
    NL_fuser_temperature: float = 0.8,
    # Autoformalizer parameters, we assume most parameters are set to default since it is not the main focus of GAR.
    autoformalizer_main_file_name: str = "./Autoformalizer.py",
    # ProofWriter parameters
    prove_writer_main_file_name: str = "./ProveWriter.py",
    prove_writer_max_length: int = 16384,
    prove_writer_batch_size: int = 16, 
    prove_writer_temperature: float = 0.8,
    prove_writer_slurm_config: ProveWriterSLURMConfig = ProveWriterSLURMConfig()
) -> GAR_NodeWorker:
    """
    This is the helper function to formulate the parameters for the GAR forward node worker.
    :param GAR_trainer_meta_config: The meta configuration for the GAR trainer, it will be used to decide most of the folder/file names
    :param Lean_server_config: The configuration for the Lean server, we use uniform config for both statement checker and prove checker.
    :param avail_cuda_devices: The available CUDA devices for the worker, it should be a list of integers.
    :param sub_experiment_idx: The sub-experiment indices for the worker, it should be a list of integers and should be the same length as avail_cuda_devices.
    :param NL_fuser_model_id: The model id for the NL statement fuser, it should be a local path because we will train the model.
    :param autoformalizer_model_id: The model id for the autoformalizer, it should be a local path or HF model id.
    :param prover_writer_model_id: The model id for the proof writer, it should be a local path because we will train the model.
    :param gpu_utilization_rate: The GPU utilization rate for the worker, it should be a float between 0 and 1, default is 0.95.
    :return: The GAR_NodeWorker that can be used for GAR training forward process.
    """
    assert len(avail_cuda_devices) == len(sub_experiment_idx), "The length of avail_cuda_devices and sub_experiment_idx should be the same."
    assert len(avail_cuda_devices) > 0, "The length of avail_cuda_devices should be greater than 0."

    # meta config for the single worker, note that some parameters will be overwritten by the node worker
    GAR_ForwardWorker_meta_config = GAR_ForwardWorkerMetaConfig(
        experiment_name=f"{GAR_trainer_meta_config.training_name}_GAR_Forward_Worker",
        save_dir_path=f"{GAR_trainer_meta_config.save_folder}/GAR_Forward_Worker",
        base_dataset_path=GAR_trainer_meta_config.base_data_path,
        cuda_device_id=avail_cuda_devices[0], 
        gpu_memory_utilization=gpu_utilization_rate,
        generated_statement_num=GAR_trainer_meta_config.theorems_fused_per_step,
        num_base_statement_per_fuse=GAR_trainer_meta_config.num_base_statements_per_fuse
    )

    # Config for statement fuser, note that some parameters will be overwritten by the node worker
    statement_fuser_meta_config = StatementFuserMetaConfig(
        base_dataset_path=GAR_trainer_meta_config.base_data_path,
        save_dir_path=f"{GAR_ForwardWorker_meta_config.save_dir_path}/StatementFuser",
        cuda_device_id=avail_cuda_devices[0],
        generated_statement_num=GAR_trainer_meta_config.theorems_fused_per_step,
        num_base_statement_per_fuse=GAR_trainer_meta_config.num_base_statements_per_fuse,
    )

    # Config for NL statement fuser, note that some parameters will be overwritten by the node worker
    NL_fuser_config = NLStatementFuserConfig(
        main_file_name=NL_fuser_main_file_name,
        model_id=NL_fuser_model_id,
        skip_thinking=NL_fuser_skip_thinking,
        max_tokens=NL_fuser_max_tokens,
        batch_size=NL_fuser_batch_size,
        micro_batch_size=NL_fuser_micro_batch_size,
        temperature=NL_fuser_temperature,
        top_p=0.95,  # default value, can be changed later
        repetition_penalty=1.0,  # default value, can be changed later
    )

    # config for autoformalizer, note that some parameters will be overwritten by the node worker
    autoformalizer_config = AutoformalizerConfig(
        main_file_name=autoformalizer_main_file_name,
        model_id=autoformalizer_model_id,
        max_tokens=8192,  # default value, can be changed later
        batch_size=16,  # default value, can be changed later
        temperature=0.6, # default value, can be changed later
        top_p=0.95,  # default value, can be changed later
        repetition_penalty=1.0,  # default value, can be changed later
    )

    # config for statement checker
    statement_checker_config = StatementCheckerConfig(
        main_file_name=Lean_server_config.main_file_name,
        Lean4_server_url=Lean_server_config.Lean4_server_url,
        batch_size=Lean_server_config.batch_size,  # default value, can be changed later
        timeout=Lean_server_config.timeout,  # default value, can be changed later
    )

    # config for prove writer, note that some parameters will be overwritten by the node worker
    prover_writer_config = ProveWriterConfig(
        main_file_name=prove_writer_main_file_name, 
        model_id=prover_writer_model_id, 
        global_begin_idx=0,  # default value, can be changed later
        global_end_idx=-1, # default value, can be changed later
        local_begin_idx=0,  # default value, can be changed later
        local_end_idx=-1,  # default value, can be changed later
        proof_num=GAR_trainer_meta_config.proofs_per_theorem, 
        top_p=0.95,  # default value, can be changed later
        repetition_penalty=1.0,  # default value, can be changed later
        max_tokens=prove_writer_max_length, 
        batch_size=prove_writer_batch_size,
        temperature=prove_writer_temperature,
        slurm_config=prove_writer_slurm_config
    )

    # Statement fuser config
    statement_fuser_config = StatementFuserConfig(
        statement_fuser_meta_config=statement_fuser_meta_config,
        nl_statement_fuser_config=NL_fuser_config,
        autoformalizer_config=autoformalizer_config,
        statement_checker_config=statement_checker_config
    )

    # Proof verifier config
    proof_verifier_config = ProofVerifierConfig(
        main_file_name=Lean_server_config.main_file_name,
        Lean4_server_url=Lean_server_config.Lean4_server_url,
        batch_size=Lean_server_config.batch_size,
        timeout=Lean_server_config.timeout
    )

    GAR_ForwardWorker_config = GAR_ForwardWorkerConfig(
        statement_fuser_config=statement_fuser_config, 
        forward_worker_meta_config=GAR_ForwardWorker_meta_config,
        prove_writer_config=prover_writer_config,
        proof_verifier_config=proof_verifier_config
    )
    experiment_file_root_path = f"{GAR_trainer_meta_config.save_folder}/GAR_step_{step_idx}/GAR_forward/save_files"
    experiment_idx_ls = [f"step-{step_idx}_{i}" for i in sub_experiment_idx]
    # create the GAR forward node worker
    print(f"[DEBUG] current GPU utilization rate: {gpu_utilization_rate}")
    GAR_forward_node_worker = GAR_NodeWorker(
        base_config=GAR_ForwardWorker_config, 
        experiment_name=f"GAR_step_{step_idx}_forward",
        avail_cuda_devices=avail_cuda_devices,
        gpu_utilization_rate=gpu_utilization_rate,
        experiment_file_root_path=experiment_file_root_path, 
        experiment_idx_ls=experiment_idx_ls
    )

    return GAR_forward_node_worker



class GAR_trainer:
    """
    This is the main class for training the GAR framework. 
    """

    def __init__(
            self, 
            GAR_trainer_meta_config: GAR_TrainerMetaConfig, 
            Lean_server_config: ProofVerifierConfig, 
            avail_cuda_devices: List[int],
            NL_fuser_model_id: str, 
            autoformalizer_model_id: str,
            prover_writer_model_id: str,
            eval_data_path: str = "./data/eval_dataset/eval_dataset_MiniF2F.json", 
            ref_model_NL_fuser: str = None,
            ref_model_prover_writer: str = None,
            step_idx: int = 0,      # By default, we start from step 0, this parameter is for reinit from the ckpt
            GAR_trainer_forward_config: GAR_TrainerForwardConfig = GAR_TrainerForwardConfig(), 
            GRPO_training_hyperparams: GRPO_Training_hyperparams = GRPO_Training_hyperparams()
    ):
        """
        Initialize the GAR_trainer class.
        """

        # Create the save file folder if not exist
        utils.check_folder_exit(GAR_trainer_meta_config.save_folder)

        # init the training step idx
        self.step_idx = step_idx

        self.eval_data_path = eval_data_path

        # init the training models
        self.NL_fuser_model_id = NL_fuser_model_id
        self.autoformalizer_model_id = autoformalizer_model_id
        self.prover_writer_model_id = prover_writer_model_id
        
        # When reinit the GAR trainer, we may want to load the reference models for NL fuser and prover writer
        if ref_model_NL_fuser is not None:
            self.ref_model_NL_fuser = ref_model_NL_fuser
        else:
            self.ref_model_NL_fuser = NL_fuser_model_id

        if ref_model_prover_writer is not None:
            self.ref_model_prover_writer = ref_model_prover_writer
        else:
            self.ref_model_prover_writer = prover_writer_model_id

        # Obtain the GAR forward node worker, since we use subprocess to run the forward worker, we do not need to worry about the garbage collection of the worker
        # The parameters maybe changed during the training process are as follows:
        self.GAR_trainer_forward_config = GAR_trainer_forward_config
        self.GAR_trainer_meta_config = GAR_trainer_meta_config
        self.Lean_server_config = Lean_server_config
        self.avail_cuda_devices = avail_cuda_devices
        self.GAR_forward_node_worker = formulate_GAR_forward_node_worker(
            GAR_trainer_meta_config=GAR_trainer_meta_config,
            Lean_server_config=Lean_server_config,
            step_idx=step_idx,
            avail_cuda_devices=avail_cuda_devices,
            sub_experiment_idx=list(range(len(avail_cuda_devices))),
            NL_fuser_model_id=NL_fuser_model_id,
            autoformalizer_model_id=autoformalizer_model_id,
            prover_writer_model_id=prover_writer_model_id,
            gpu_utilization_rate=GAR_trainer_forward_config.gpu_utilization_rate,
            NL_fuser_main_file_name=GAR_trainer_forward_config.NL_fuser_main_file_name,
            NL_fuser_skip_thinking=GAR_trainer_forward_config.NL_fuser_skip_thinking,
            NL_fuser_max_tokens=GAR_trainer_forward_config.NL_fuser_max_tokens,
            NL_fuser_batch_size=GAR_trainer_forward_config.NL_fuser_batch_size,
            NL_fuser_micro_batch_size=GAR_trainer_forward_config.NL_fuser_micro_batch_size,
            NL_fuser_temperature=GAR_trainer_forward_config.NL_fuser_temperature,
            autoformalizer_main_file_name=GAR_trainer_forward_config.autoformalizer_main_file_name,
            prove_writer_main_file_name=GAR_trainer_forward_config.prove_writer_main_file_name,
            prove_writer_max_length=GAR_trainer_forward_config.prove_writer_max_length,
            prove_writer_batch_size=GAR_trainer_forward_config.prove_writer_batch_size,
            prove_writer_temperature=GAR_trainer_forward_config.prove_writer_temperature, 
            prove_writer_slurm_config=GAR_trainer_forward_config.prove_writer_slurm_config
        )

        # Name change needed
        self.GRPO_training_hyperparams = GRPO_training_hyperparams
        
        # Init the ckpt dict
        self.ckpt_dict = {
            "meta_config": asdict(GAR_trainer_meta_config),
            "step_idx": step_idx,
            "NL_fuser_model_id": NL_fuser_model_id,
            "autoformalizer_model_id": autoformalizer_model_id,
            "prover_writer_model_id": prover_writer_model_id,
            "ref_model_NL_fuser": ref_model_NL_fuser,
            "ref_model_prover_writer": ref_model_prover_writer,
            "GRPO_training_hyperparams": asdict(GRPO_training_hyperparams),
            "forward_config": asdict(GAR_trainer_forward_config), 
            "GRPO_training_steps": 0,  # The number of GRPO training, it will accounts for how much steps of training globally
            # The training
            "training_log": []
        }

        self.training_ckpt_path = f"{self.GAR_trainer_meta_config.save_folder}/GAR_trainer_ckpt.json"

        utils.write_to_json(
            self.training_ckpt_path, 
            self.ckpt_dict
        )

    def _init_step_training_dict(self, step: int):
        """
        This is the helper function to init the training dict for each GAR step.
        """
        step_training_dict = {
            "step": step,
            "forward_process_finished": False,
            "data_state": None, 
            "NL_fuser_GRPO_precompute_log": {
                "finished": False
            },
            "Prover_GRPO_precompute_status": {
                "finished": False
            },
            "NL_fuser_GRPO_training_status": {
                "finished": False,
                "model_path": None
            },
            "Prover_GRPO_training_status": {
                "finished": False,
                "model_path": None
            }
        }
        return step_training_dict



    def run_GAR_training(
            self, 
            step_idx: int = 0
    ):
        """
        This is the main function to run the GAR training process, 
        we will manage the training steps including the forward and backward process in this function. It also supports resuming from the checkpoint by providing the step_idx.
        We assume the loading of ckpt by step and we do not support loading from arbitrary intermediate state.

        :param step_idx: The step index to start training from (by default is 0), it will be used to load the checkpoint and resume the training.
        """

        # Check if the step_idx is valid
        if step_idx < 0 or step_idx >= self.GAR_trainer_meta_config.total_steps:
            raise ValueError(f"Invalid step_idx: {step_idx}, it should be in range [0, {self.GAR_trainer_meta_config.total_steps})")

        if step_idx != 0:
            print(f"Resuming from ckpt: {self.GAR_trainer_meta_config.save_folder}/GAR_trainer_ckpt.json")
            self.ckpt_dict = utils.read_from_json(self.training_ckpt_path)

        # Set the current step index
        self.step_idx = step_idx

        max_steps = self.GRPO_training_hyperparams.max_steps

        for step in range(self.step_idx, max_steps):

            # Init the training dict for this step
            step_training_dict = self._init_step_training_dict(step)
            self.ckpt_dict["training_log"].append(step_training_dict)
            self.ckpt_dict["step_idx"] = step
            utils.write_to_json(
                self.training_ckpt_path, 
                self.ckpt_dict
            )

            print(f"Running forward process for GAR step {step}...")
            NL_fuser_data_state_dict, prover_data_state_dict = self._run_GAR_forward(step)
            step_training_dict["forward_process_finished"] = True
            step_training_dict["data_state"] = {
                "NL_fuser_data_state": NL_fuser_data_state_dict,
                "prover_data_state": prover_data_state_dict
            }
            utils.write_to_json(
                self.training_ckpt_path, 
                self.ckpt_dict
            )


            # Precompute the log probabilities for the prover and NL fuser models
            print("Precomputing log probabilities for prover model")
            
            # split the available cuda devices into list of two devices tuple
            cuda_device_pairs = [(self.avail_cuda_devices[i], self.avail_cuda_devices[i+1]) for i in range(0, len(self.avail_cuda_devices)-1, 2)]
            print(f"Using CUDA device pairs: {cuda_device_pairs}")
            prover_precompute_ckpt_root_path = f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/Precomputed_log_probs_proofs/prover_log_probs"
            prover_precompute_dataset_path = f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_training_data/processed_prover_training_data.json"
            prover_precomputed_results_path = f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_training_data/prover_precomputed_log_probs.pkl"
            
            # We use the same set of available cuda devices for the NL fuser model
            NL_fuser_precompute_ckpt_root_path = f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/Precomputed_log_probs_proofs/NL_fuser_log_probs"
            NL_fuser_precompute_dataset_path = f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_training_data/processed_NL_fuser_training_data.json"
            NL_fuser_precomputed_results_path = f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_training_data/NL_fuser_precomputed_log_probs.pkl"


            self._run_GAR_data_precompute(
                step=step,
                step_training_dict=step_training_dict,
                cuda_device_pairs=cuda_device_pairs,
                NL_fuser_precompute_dataset_path=NL_fuser_precompute_dataset_path,
                prover_precompute_dataset_path=prover_precompute_dataset_path,
                NL_fuser_precompute_ckpt_root_path=NL_fuser_precompute_ckpt_root_path,
                prover_precompute_ckpt_root_path=prover_precompute_ckpt_root_path,
                NL_fuser_precomputed_results_path=NL_fuser_precomputed_results_path,
                prover_precomputed_results_path=prover_precomputed_results_path, 
            )

            # Run the GRPO training step for the NL fuser model
            print("Running GRPO training for NL Fuser model")

            GRPO_training_config_NL_fuser = GRPO_Training_config(
                training_idx=f"{self.GAR_trainer_meta_config.training_name}_step_{step}_NL_fuser_GRPO_training",
                model_to_train_path=self.NL_fuser_model_id,
                training_model_name=f"NL_fuser_step_{step}",
                wandb_run_name=f"{self.GAR_trainer_meta_config.training_name}_step_{step}_NL_fuser_GRPO_training",
                train_data_path=NL_fuser_precomputed_results_path, 
                eval_data_path=self.eval_data_path,
                model_save_path=f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/trained_models/NL_fuser_model_step_{step}", 
                model_ckpt_path=f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/trained_models/NL_fuser_model_step_{step}_ckpt",
                ckpt_freq=int(1E9),  # No checkpoint will be saved by default
                begin_idx=0,  # Use all data by default
                logging_conf_file="./conf/common.log_conf", # We use the default logging config file
                use_wandb=True,
                wandb_project="GAR_NL_fuser_GRPO_training"
            )

            self._run_GAR_training_step(
                step=step,
                data_type="NL_fuser",
                GRPO_trainingConfig=GRPO_training_config_NL_fuser,
                GRPO_trainingHyperparams=self.GRPO_training_hyperparams, 
                fsdp_config_path=NL_FUSER_FSDP_FILENAME
            )

            step_training_dict["NL_fuser_GRPO_training_status"]["finished"] = True
            step_training_dict["NL_fuser_GRPO_training_status"]["model_path"] = GRPO_training_config_NL_fuser.model_save_path
            if self.GAR_trainer_meta_config.train_fuser:
                self.ckpt_dict["NL_fuser_model_id"] = GRPO_training_config_NL_fuser.model_save_path
                self.NL_fuser_model_id = GRPO_training_config_NL_fuser.model_save_path
            utils.write_to_json(self.training_ckpt_path, self.ckpt_dict)
            # Run the GRPO training step for the prover model
            print("Running GRPO training for Prover model")

            GRPO_training_config_prover = GRPO_Training_config(
                training_idx=f"{self.GAR_trainer_meta_config.training_name}_step_{step}_Prover_GRPO_training",
                model_to_train_path=self.prover_writer_model_id,
                training_model_name=f"Prover_step_{step}",
                wandb_run_name=f"{self.GAR_trainer_meta_config.training_name}_step_{step}_Prover_GRPO_training",
                train_data_path=prover_precomputed_results_path,
                eval_data_path=self.eval_data_path,
                model_save_path=f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/trained_models/Prover_model_step_{step}",
                model_ckpt_path=f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/trained_models/Prover_model_step_{step}_ckpt",
                ckpt_freq=int(1E9),  # No checkpoint will be saved by default
                begin_idx=0,  # Use all data by default
                logging_conf_file="./conf/common.log_conf",  # We use the default logging config file
                use_wandb=True,
                wandb_project="GAR_Prover_GRPO_training"
            )

            self._run_GAR_training_step(
                step=step, 
                data_type="prover",
                GRPO_trainingConfig=GRPO_training_config_prover, 
                GRPO_trainingHyperparams=self.GRPO_training_hyperparams, 
                fsdp_config_path=PROVER_FSDP_FILENAME
            )

            step_training_dict["Prover_GRPO_training_status"]["finished"] = True
            step_training_dict["Prover_GRPO_training_status"]["model_path"] = GRPO_training_config_prover.model_save_path
            self.ckpt_dict["prover_writer_model_id"] = GRPO_training_config_prover.model_save_path
            self.prover_writer_model_id = GRPO_training_config_prover.model_save_path
            utils.write_to_json(self.training_ckpt_path, self.ckpt_dict)

            print("Finished GAR step {step}, model saved to {self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/trained_models/")
            print("-------------------------------------------------------")
            print(f"Current step training dict: {step_training_dict}")
            # Update the step idx in the ckpt dict
            self.ckpt_dict["step_idx"] += 1
            utils.write_to_json(self.training_ckpt_path, self.ckpt_dict)

    def _run_GAR_training_step(
        self, 
        step: int,
        data_type: str,
        GRPO_trainingConfig: GRPO_Training_config, 
        GRPO_trainingHyperparams: GRPO_Training_hyperparams = GRPO_Training_hyperparams(), 
        fsdp_config_path: Optional[str] = "fsdp_config_offload_enabled.yaml",
        log_doc_header: Optional[str] = "GAR_Training_step_log"
    ):
        """
        The function to run GRPO training step, we use subprocess to manage the training process to avoid memory leak and ensure the training process is isolated.
        :param step: The current step index for the training process.
        :param data_type: The type of data to be used for training, it should be either "NL_fuser" or "prover". It will be used to determine the training data path and model id.
        :param GRPO_trainingConfig: The configuration for the GRPO training process, it should contain the following keys:
            - training_idx: The index of the training process, it will be used to save the model and log.
            - model_to_train_path: The path to the model to be trained.
            - training_model_name: The name of the model to be trained.
            - wandb_run_name: The name of the wandb run.
            - train_data_path: The path to the training data.
            - eval_data_path: The path to the evaluation data.
            - model_save_path: The path to save the trained model.
            - model_ckpt_path: The path to save the model checkpoint.
        :param GRPO_trainingHyperparams: The hyperpara for the GRPO training process, it should not be changed frequently.
        """

        # prepare the training parameters
        # meta config for training
        training_idx = GRPO_trainingConfig.training_idx
        model_to_train_path = GRPO_trainingConfig.model_to_train_path
        training_model_name = GRPO_trainingConfig.training_model_name
        wandb_run_name = GRPO_trainingConfig.wandb_run_name
        
        # training and evaluation data paths
        train_data_path = GRPO_trainingConfig.train_data_path
        eval_data_path = GRPO_trainingConfig.eval_data_path
        begin_idx = GRPO_trainingConfig.begin_idx

        # Optimizer parameters
        optimizer_name = GRPO_trainingHyperparams.optimizer_name
        lr = GRPO_trainingHyperparams.lr
        weight_decay = GRPO_trainingHyperparams.weight_decay
        max_steps = GRPO_trainingHyperparams.max_steps
        epoch = GRPO_trainingHyperparams.epoch
        global_batch_size = GRPO_trainingHyperparams.global_batch_size
        micro_batch_size = GRPO_trainingHyperparams.micro_batch_size 
        eval_batch_size = GRPO_trainingHyperparams.eval_batch_size
        gradient_accumulation_steps = GRPO_trainingHyperparams.gradient_accumulation_steps
        use_bf16 = GRPO_trainingHyperparams.use_bf16
        max_length = GRPO_trainingHyperparams.max_length
        eval_frequency = GRPO_trainingHyperparams.eval_frequency
        eps_clip = GRPO_trainingHyperparams.eps_clip
        beta_kl = GRPO_trainingHyperparams.beta_kl

        # debug parameters
        logging_conf_file = GRPO_trainingConfig.logging_conf_file
        use_wandb = GRPO_trainingConfig.use_wandb
        wandb_project = GRPO_trainingConfig.wandb_project

        # saving and checkpointing parameters
        save_path = GRPO_trainingConfig.model_save_path
        ckpt_path = GRPO_trainingConfig.model_ckpt_path
        ckpt_freq = GRPO_trainingConfig.ckpt_freq
        
        utils.check_folder_exit(f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GRPO_training_log/")
        # run the training process
        print(f"Running GRPO training for {data_type} model at step {step}...")
        command = f"accelerate launch --config_file {fsdp_config_path} main_RL_trainner_GRPO.py " \
            f"--training_idx {training_idx} " \
            f"--model_path {model_to_train_path} " \
            f"--model_name {training_model_name} " \
            f"--wandb_run_name {wandb_run_name} " \
            f"--data_type {data_type} " \
            f"--train_data_path {train_data_path} " \
            f"--eval_data_path {eval_data_path} " \
            f"--begin_idx {begin_idx} " \
            f"--optimizer_name {optimizer_name} " \
            f"--lr {lr} " \
            f"--weight_decay {weight_decay} " \
            f"--max_steps {max_steps} " \
            f"--epoch {epoch} " \
            f"--global_batch_size {global_batch_size} " \
            f"--micro_batch_size {micro_batch_size} " \
            f"--eval_batch_size {eval_batch_size} " \
            f"--gradient_accumulation_steps {gradient_accumulation_steps} " \
            f"--use_bf16 {use_bf16} " \
            f"--max_length {max_length} " \
            f"--eval_frequency {eval_frequency} " \
            f"--eps_clip {eps_clip} " \
            f"--beta_kl {beta_kl} " \
            f"--logging_conf_file {logging_conf_file} " \
            f"--use_wandb {use_wandb} " \
            f"--wandb_project {wandb_project} " \
            f"--save_path {save_path} " \
            f"--ckpt_path {ckpt_path} " \
            f"--ckpt_step {int(ckpt_freq)} " 
        out_path = f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GRPO_training_log/{log_doc_header}_{data_type}.log"
        err_path = f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GRPO_training_log/{log_doc_header}_{data_type}.err"
        # Key: Use a subshell to write PID to stderr, then exec the actual command
        command = (
            f"(echo 'PID: $$' >&2; exec {command})"
            f" > {out_path} 2> {err_path}"
        )
        run_command(command)

        print(f"Running GRPO training for {data_type} model at step {step} finished. Model saved to {save_path}.")



    def _run_GAR_data_precompute(
            self, 
            step: int, 
            step_training_dict: Dict[str, Any],
            cuda_device_pairs: List[tuple],
            NL_fuser_precompute_dataset_path: str,
            prover_precompute_dataset_path: str,
            NL_fuser_precompute_ckpt_root_path: str,
            prover_precompute_ckpt_root_path: str,
            NL_fuser_precomputed_results_path: str,
            prover_precomputed_results_path: str
    ):
        """
        This function helps to run the GAR data precompute process, it will be used in the run_GAR_training function.
        """
        # Run the precompute log probabilities for the prover model
        print("Precomputing log probabilities for prover model")
        run_precompute_log_probs_proofs(
            gpu_idx_pair_ls=cuda_device_pairs,
            worker_idx=list(range(len(cuda_device_pairs))),
            dataset_path=prover_precompute_dataset_path,
            ckpt_root_path=prover_precompute_ckpt_root_path,
            old_model_path=self.prover_writer_model_id,
            ref_model_path=self.ref_model_prover_writer,
            batch_size=self.GRPO_training_hyperparams.micro_batch_size * len(cuda_device_pairs) * 2,  # The batch size should be the micro batch size times the number of available devices
            max_length=self.GRPO_training_hyperparams.max_length,
            shuffle_data=True  # We shuffle the data to ensure the training data is diverse
        )

        step_training_dict["Prover_GRPO_precompute_status"]["finished"] = True
        step_training_dict["Prover_GRPO_precompute_status"]["dataset_path"] = prover_precompute_dataset_path
        step_training_dict["Prover_GRPO_precompute_status"]["ckpt_root_path"] = prover_precompute_ckpt_root_path
        step_training_dict["Prover_GRPO_precompute_status"]["precomputed_results_path"] = prover_precomputed_results_path
        utils.write_to_json(
            self.training_ckpt_path, 
            self.ckpt_dict
        )

        summarize_precompute_results_prover(
            ckpt_root_path=prover_precompute_ckpt_root_path, 
            save_path=prover_precomputed_results_path
        )

        # Run the precompute log probabilities for the NL fuser model
        print("Precomputing log probabilities for NL fuser model")
        run_precompute_log_probs_NL_fuser(
            gpu_idx_pair_ls=cuda_device_pairs,
            worker_idx=list(range(len(cuda_device_pairs))),
            dataset_path=NL_fuser_precompute_dataset_path,
            ckpt_root_path=NL_fuser_precompute_ckpt_root_path,
            old_model_path=self.NL_fuser_model_id,
            ref_model_path=self.ref_model_NL_fuser,
            batch_size=self.GRPO_training_hyperparams.micro_batch_size * len(cuda_device_pairs) * 2,  # The batch size should be the micro batch size times the number of available devices
            max_length=self.GRPO_training_hyperparams.max_length,
            shuffle_data=True  # We shuffle the data to ensure the training data is diverse
        )

        summarize_precompute_results_NL_fuser(
            ckpt_root_path=NL_fuser_precompute_ckpt_root_path, 
            save_path=NL_fuser_precomputed_results_path
        )

        step_training_dict["NL_fuser_GRPO_precompute_log"]["finished"] = True
        step_training_dict["NL_fuser_GRPO_precompute_log"]["dataset_path"] = NL_fuser_precompute_dataset_path
        step_training_dict["NL_fuser_GRPO_precompute_log"]["ckpt_root_path"] = NL_fuser_precompute_ckpt_root_path
        step_training_dict["NL_fuser_GRPO_precompute_log"]["precomputed_results_path"] = NL_fuser_precomputed_results_path
        utils.write_to_json(
            self.training_ckpt_path, 
            self.ckpt_dict
        )

        return step_training_dict

    def _run_GAR_forward(self, step: int):
        # Formulate current GAR forward node worker for this step
            self.GAR_forward_node_worker = formulate_GAR_forward_node_worker(
                GAR_trainer_meta_config=self.GAR_trainer_meta_config,
                Lean_server_config=self.Lean_server_config,
                step_idx=step,
                avail_cuda_devices=self.avail_cuda_devices,
                sub_experiment_idx=list(range(len(self.avail_cuda_devices))),
                NL_fuser_model_id=self.NL_fuser_model_id,
                autoformalizer_model_id=self.autoformalizer_model_id,
                prover_writer_model_id=self.prover_writer_model_id,
                gpu_utilization_rate=self.GAR_trainer_forward_config.gpu_utilization_rate,
                NL_fuser_main_file_name=self.GAR_trainer_forward_config.NL_fuser_main_file_name,
                NL_fuser_skip_thinking=self.GAR_trainer_forward_config.NL_fuser_skip_thinking,
                NL_fuser_max_tokens=self.GAR_trainer_forward_config.NL_fuser_max_tokens,
                NL_fuser_batch_size=self.GAR_trainer_forward_config.NL_fuser_batch_size,
                NL_fuser_micro_batch_size=self.GAR_trainer_forward_config.NL_fuser_micro_batch_size,
                NL_fuser_temperature=self.GAR_trainer_forward_config.NL_fuser_temperature,
                autoformalizer_main_file_name=self.GAR_trainer_forward_config.autoformalizer_main_file_name,
                prove_writer_main_file_name=self.GAR_trainer_forward_config.prove_writer_main_file_name,
                prove_writer_max_length=self.GAR_trainer_forward_config.prove_writer_max_length,
                prove_writer_batch_size=self.GAR_trainer_forward_config.prove_writer_batch_size,
                prove_writer_temperature=self.GAR_trainer_forward_config.prove_writer_temperature, 
                prove_writer_slurm_config=self.GAR_trainer_forward_config.prove_writer_slurm_config
            )
            
            print(f"Starting GAR step {step} forward process...")

            self.GAR_forward_node_worker.run_all_workers(
                statement_fuser_finished=False, 
                prove_writer_finished=False,
                proof_verifier_finished=False
            )
            print(f"Starting preprocess RL data for GAR step {step}...")

            # Preprocess the RL data for the current step
            statementFuserTokenizer = AutoTokenizer.from_pretrained(self.NL_fuser_model_id)
            statementFuser_RL_dataPreprocessor = RL_DataPreprocessor(
                statementFuserTokenizer,
                f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_forward",
                recorded_input=True
            )

            NL_fuser_data_state_dict = statementFuser_RL_dataPreprocessor.preprocess_NL_fuser_training_data(
                f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_training_data",
                max_pass_rate=self.GAR_trainer_meta_config.NL_FuserMaxPassRate,
                min_pass_rate=0.0,  # We assume the min pass rate is 0.0, which means we will use all data
                LLM_type="Qwen3"    # The default type is Qwen3, normally we do not need it
            )

            # Preprocess the RL data for the prove writer
            proveWriterTokenizer = AutoTokenizer.from_pretrained(self.prover_writer_model_id)
            proveWriter_RL_dataPreprocessor = RL_DataPreprocessor(
                proveWriterTokenizer,
                f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_forward",
                recorded_input=True
            )
            prover_data_state_dict = proveWriter_RL_dataPreprocessor.preprocess_Prover_training_data(
                f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_training_data",
                max_pass_rate=self.GAR_trainer_meta_config.ProverMaxPassRate,
                min_pass_rate=0.0,  # We assume the min pass rate is 0.0, which means we will use all data
                LLM_type="Qwen3"    # The default type is Qwen3, normally we do not need it
            )

            return NL_fuser_data_state_dict, prover_data_state_dict
    
    def save_ckpt(self):
        if not hasattr(self, "training_ckpt_path"):
            self.training_ckpt_path = f"{self.GAR_trainer_meta_config.save_folder}/GAR_trainer_ckpt.json"
        utils.write_to_json(
            self.training_ckpt_path, 
            self.ckpt_dict
        )

    @staticmethod
    def _dic_to_dc(
        dict_class, 
        dict_to_write: Dict, 
        defaults: Dict = None 
    ):
        if dict_to_write == None:
            dict_to_write = {}
        if defaults:
            base = dict(defaults)
            base.update(dict_to_write)
            dict_to_write = base
        
        keys = {f.name for f in dict_class.__dataclass_fields__.values()}
        filtered = {k: v for k, v in dict_to_write.items() if k in keys}
        return dict_class(**filtered)

    @classmethod
    def from_ckpt(
        cls, 
        ckpt_path: str, 
        Lean_server_config: ProofVerifierConfig, 
        avail_cuda_devices: List[int],
        eval_data_path: str, 
        ref_model_NL_fuser_id: str = None, 
        ref_model_prover_id: str = None
    ) -> "GAR_trainer":
        """
        This function resume the GAR_trainer from the ckpt file, does not reinit the training_log and the progress
        :param ckpt_path: The path to the checkpoint file.
        :param Lean_server_config: The configuration for the Lean server for both statement and proof verification.
        :param avail_cuda_devices: The list of available CUDA devices for training, it should be a list of integers representing the device ids.
        :param eval_data_path: The path to the evaluation data, it will be used for evaluation during the training process.
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")

        ckpt = utils.read_from_json(ckpt_path)
        meta_config: GAR_TrainerMetaConfig = cls._dic_to_dc(
            GAR_TrainerMetaConfig, 
            ckpt.get("meta_config", {})
        )
        forward_config: GAR_TrainerForwardConfig = cls._dic_to_dc(
            GAR_TrainerForwardConfig, 
            ckpt.get("forward_config", {}),
        )
        GRPO_hyperparams: GRPO_Training_hyperparams = cls._dic_to_dc(
            GRPO_Training_hyperparams, 
            ckpt.get("GRPO_training_hyperparams", {}),
        )

        prove_writer_slurm_config: ProveWriterSLURMConfig = cls._dic_to_dc(
            ProveWriterSLURMConfig, 
            ckpt["forward_config"]["prove_writer_slurm_config"]
        )

        # Reinit the instance without __init__
        self = cls.__new__(cls)

        # Basic attributes
        self.GAR_trainer_meta_config = meta_config
        self.GAR_trainer_forward_config = forward_config
        self.GAR_trainer_forward_config.prove_writer_slurm_config = prove_writer_slurm_config
        self.GRPO_training_hyperparams = GRPO_hyperparams
        self.Lean_server_config = Lean_server_config
        self.avail_cuda_devices = avail_cuda_devices
        self.eval_data_path = eval_data_path

        # current model and reference model ids
        self.NL_fuser_model_id = ckpt.get("NL_fuser_model_id")
        self.autoformalizer_model_id = ckpt.get("autoformalizer_model_id")
        self.prover_writer_model_id = ckpt.get("prover_writer_model_id")

        if ref_model_NL_fuser_id == None:
            self.ref_model_NL_fuser = ckpt.get("ref_model_NL_fuser", self.NL_fuser_model_id)
        else:
            self.ref_model_NL_fuser = ref_model_NL_fuser_id
            ckpt["ref_model_NL_fuser"] = ref_model_NL_fuser_id  # Update the ckpt dict to avoid confusion
        
        if ref_model_prover_id == None:
            self.ref_model_prover_writer = ckpt.get("ref_model_prover_writer", self.prover_writer_model_id)
            ckpt["ref_model_prover_writer"] = self.ref_model_prover_writer  # Update the ckpt dict to avoid confusion
        else:
            self.ref_model_prover_writer = ref_model_prover_id
            ckpt["ref_model_prover_writer"] = self.ref_model_prover_writer  # Update the ckpt dict to avoid confusio

        # step and log
        self.ckpt_dict = ckpt
        self.step_idx = int(ckpt.get("step_idx", 0))
        self.training_ckpt_path = ckpt_path

        # Reinit forward work according to the current step
        self.GAR_forward_node_worker = formulate_GAR_forward_node_worker(
            GAR_trainer_meta_config=self.GAR_trainer_meta_config, 
            Lean_server_config=self.Lean_server_config, 
            step_idx=self.step_idx, 
            avail_cuda_devices=self.avail_cuda_devices,
            sub_experiment_idx=list(range(len(self.avail_cuda_devices))),
            NL_fuser_model_id=self.NL_fuser_model_id, 
            autoformalizer_model_id=self.autoformalizer_model_id, 
            prover_writer_model_id=self.prover_writer_model_id,
            gpu_utilization_rate=self.GAR_trainer_forward_config.gpu_utilization_rate, 
            NL_fuser_main_file_name=self.GAR_trainer_forward_config.NL_fuser_main_file_name,
            NL_fuser_skip_thinking=self.GAR_trainer_forward_config.NL_fuser_skip_thinking,
            NL_fuser_max_tokens=self.GAR_trainer_forward_config.NL_fuser_max_tokens,
            NL_fuser_batch_size=self.GAR_trainer_forward_config.NL_fuser_batch_size,
            NL_fuser_micro_batch_size=self.GAR_trainer_forward_config.NL_fuser_micro_batch_size,
            NL_fuser_temperature=self.GAR_trainer_forward_config.NL_fuser_temperature,
            autoformalizer_main_file_name=self.GAR_trainer_forward_config.autoformalizer_main_file_name,
            prove_writer_main_file_name=self.GAR_trainer_forward_config.prove_writer_main_file_name,
            prove_writer_max_length=self.GAR_trainer_forward_config.prove_writer_max_length,
            prove_writer_batch_size=self.GAR_trainer_forward_config.prove_writer_batch_size,
            prove_writer_temperature=self.GAR_trainer_forward_config.prove_writer_temperature, 
            prove_writer_slurm_config=self.GAR_trainer_forward_config.prove_writer_slurm_config
        )

        return self
    
    def _ensure_step_entry(self, step: int):
        """
        Ensuring the training_log has the step record, if not, we will create a new one
        """
        logs = self.ckpt_dict.setdefault("training_log", [])
        for ent in logs:
            if ent.get("step") == step:
                return ent
        ent = self._init_step_training_dict(step)
        logs.append(ent)
        self.ckpt_dict["step_idx"] = step
        self.save_ckpt()
        return ent

    def __infer_resume_step(self) -> int:
        """
        Scanning the training log and return the first unfinshed steo, if all steps are finished and step_idx + 1 < total_steps, return next step
        If not, return current step_idx.
        """
        total = self.GAR_trainer_meta_config.total_steps
        logs = self.ckpt_dict.get("training_log", [])
        logs_sorted = sorted(logs, key=lambda x: x.get("step", 0))
        for ent in logs_sorted:
            st = ent.get("step", 0)
            nl_p = ent.get("NL_fuser_GRPO_precompute_log", {}).get("finished", False)
            pv_p = ent.get("Prover_GRPO_precompute_status", {}).get("finished", False)
            nl_t = ent.get("NL_fuser_GRPO_training_status", {}).get("finished", False)
            pv_t = ent.get("Prover_GRPO_training_status", {}).get("finished", False)
            fwd = ent.get("forward_process_finished", False)
            if not (fwd and nl_p and pv_p and nl_t and pv_t):
                return st
        cur = int(self.ckpt_dict.get("step_idx", 0))
        nxt = cur + 1
        if nxt < total:
            return nxt
        return cur  

    def resume_from_ckpt(self):
        """
        The major entrance for reinit running, execute the ramining process from the last unfinished step.
        """

        step = self.__infer_resume_step()
        print(f"Resuming from step {step}...")

        total_steps = self.GAR_trainer_meta_config.total_steps

        step_ent = self._ensure_step_entry(step)
        self.step_idx = step
        self.ckpt_dict["step_idx"] = step
        self.save_ckpt()

        while step < total_steps:
            step_ent = self._ensure_step_entry(step)

            # Forward process
            if not step_ent.get("forward_process_finished", False):
                print(f"[Resume] step {step} forward process not finished, running...")
                NL_state, PV_state = self._run_GAR_forward(step)
                step_ent["forward_process_finished"] = True
                step_ent["data_state"] = {
                    "NL_fuser_data_state": NL_state,
                    "prover_data_state": PV_state,
                }
                self.save_ckpt()
            else:
                if step_ent.get("data_state", None) == None:
                    print("Some workers has results, keep training while ignorning failed worker")
                    # Preprocess the RL data for the current step
                    statementFuserTokenizer = AutoTokenizer.from_pretrained(self.NL_fuser_model_id)
                    statementFuser_RL_dataPreprocessor = RL_DataPreprocessor(
                        statementFuserTokenizer,
                        f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_forward",
                        recorded_input=True
                    )

                    NL_state = statementFuser_RL_dataPreprocessor.preprocess_NL_fuser_training_data(
                        f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_training_data",
                        max_pass_rate=self.GAR_trainer_meta_config.NL_FuserMaxPassRate,
                        min_pass_rate=0.0,  # We assume the min pass rate is 0.0, which means we will use all data
                        LLM_type="Qwen3"    # The default type is Qwen3, normally we do not need it
                    )

                    # Preprocess the RL data for the prove writer
                    proveWriterTokenizer = AutoTokenizer.from_pretrained(self.prover_writer_model_id)
                    proveWriter_RL_dataPreprocessor = RL_DataPreprocessor(
                        proveWriterTokenizer,
                        f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_forward",
                        recorded_input=True
                    )
                    PV_state = proveWriter_RL_dataPreprocessor.preprocess_Prover_training_data(
                        f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_training_data",
                        max_pass_rate=self.GAR_trainer_meta_config.ProverMaxPassRate,
                        min_pass_rate=0.0,  # We assume the min pass rate is 0.0, which means we will use all data
                        LLM_type="Qwen3"    # The default type is Qwen3, normally we do not need it
                    )

                    step_ent["data_state"] = {
                        "NL_fuser_data_state": NL_state,
                        "prover_data_state": PV_state,
                    }
                    self.save_ckpt()
                        
            
            if not (step_ent.get("Prover_GRPO_precompute_status", {}).get("finished", False) and
                    step_ent.get("NL_fuser_GRPO_precompute_log", {}).get("finished", False)):

                print(f"[RESUME] Step {step}: precompute log-probs not finished, reinit the process with preprocessing...")
                # Preprocess the RL data for the current step
                statementFuserTokenizer = AutoTokenizer.from_pretrained(self.NL_fuser_model_id)
                statementFuser_RL_dataPreprocessor = RL_DataPreprocessor(
                    statementFuserTokenizer,
                    f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_forward",
                    recorded_input=True
                )

                NL_fuser_data_state_dict = statementFuser_RL_dataPreprocessor.preprocess_NL_fuser_training_data(
                    f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_training_data",
                    max_pass_rate=self.GAR_trainer_meta_config.NL_FuserMaxPassRate,
                    min_pass_rate=0.0,  # We assume the min pass rate is 0.0, which means we will use all data
                    LLM_type="Qwen3"    # The default type is Qwen3, normally we do not need it
                )

                # Preprocess the RL data for the prove writer
                proveWriterTokenizer = AutoTokenizer.from_pretrained(self.prover_writer_model_id)
                proveWriter_RL_dataPreprocessor = RL_DataPreprocessor(
                    proveWriterTokenizer,
                    f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_forward",
                    recorded_input=True
                )
                prover_data_state_dict = proveWriter_RL_dataPreprocessor.preprocess_Prover_training_data(
                    f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_training_data",
                    max_pass_rate=self.GAR_trainer_meta_config.ProverMaxPassRate,
                    min_pass_rate=0.0,  # We assume the min pass rate is 0.0, which means we will use all data
                    LLM_type="Qwen3"    # The default type is Qwen3, normally we do not need it
                )


                print(f"[RESUME] Step {step}: precompute log-probs...")
                # 准备 device pair（沿用你在 run_GAR_training 中的逻辑）
                cuda_device_pairs = [(self.avail_cuda_devices[i], self.avail_cuda_devices[i+1])
                                     for i in range(0, len(self.avail_cuda_devices)-1, 2)]
                pv_ckpt_root = f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/Precomputed_log_probs_proofs/prover_log_probs"
                pv_dataset = f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_training_data/processed_prover_training_data.json"
                pv_result = f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_training_data/prover_precomputed_log_probs.pkl"

                nl_ckpt_root = f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/Precomputed_log_probs_proofs/NL_fuser_log_probs"
                nl_dataset = f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_training_data/processed_NL_fuser_training_data.json"
                nl_result = f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/GAR_training_data/NL_fuser_precomputed_log_probs.pkl"

                self._run_GAR_data_precompute(
                    step=step,
                    step_training_dict=step_ent,
                    cuda_device_pairs=cuda_device_pairs,
                    NL_fuser_precompute_dataset_path=nl_dataset,
                    prover_precompute_dataset_path=pv_dataset,
                    NL_fuser_precompute_ckpt_root_path=nl_ckpt_root,
                    prover_precompute_ckpt_root_path=pv_ckpt_root,
                    NL_fuser_precomputed_results_path=nl_result,
                    prover_precomputed_results_path=pv_result,
                )
                self.save_ckpt()

            # C. NL fuser GRPO
            if not step_ent.get("NL_fuser_GRPO_training_status", {}).get("finished", False):
                print(f"[RESUME] Step {step}: GRPO training (NL fuser)...")

                nl_result = step_ent["NL_fuser_GRPO_precompute_log"]["precomputed_results_path"]
                GRPO_training_config_NL_fuser = GRPO_Training_config(
                    training_idx=f"{self.GAR_trainer_meta_config.training_name}_step_{step}_NL_fuser_GRPO_training",
                    model_to_train_path=self.NL_fuser_model_id,
                    training_model_name=f"NL_fuser_step_{step}",
                    wandb_run_name=f"{self.GAR_trainer_meta_config.training_name}_step_{step}_NL_fuser_GRPO_training",
                    train_data_path=nl_result,
                    eval_data_path=self.eval_data_path,
                    model_save_path=f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/trained_models/NL_fuser_model_step_{step}",
                    model_ckpt_path=f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/trained_models/NL_fuser_model_step_{step}_ckpt",
                    ckpt_freq=1E9,
                    begin_idx=0,
                    logging_conf_file="./conf/common.log_conf",
                    use_wandb=True,
                    wandb_project="GAR_NL_fuser_GRPO_traƒining",
                )
                self._run_GAR_training_step(
                    step=step,
                    data_type="NL_fuser",
                    GRPO_trainingConfig=GRPO_training_config_NL_fuser,
                    GRPO_trainingHyperparams=self.GRPO_training_hyperparams, 
                    fsdp_config_path=NL_FUSER_FSDP_FILENAME
                )
                step_ent["NL_fuser_GRPO_training_status"]["finished"] = True
                step_ent["NL_fuser_GRPO_training_status"]["model_path"] = GRPO_training_config_NL_fuser.model_save_path

                # Update the current NL fuser model id to the newly trained model, which will be used in the next step forward process and training if needed
                if self.GAR_trainer_meta_config.train_fuser:
                    self.ckpt_dict["NL_fuser_model_id"] = GRPO_training_config_NL_fuser.model_save_path
                    self.NL_fuser_model_id = GRPO_training_config_NL_fuser.model_save_path
                self.save_ckpt()
            
            # D. Prover GRPO
            if not step_ent.get("Prover_GRPO_training_status", {}).get("finished", False):
                print(f"[RESUME] Step {step}: GRPO training (Prover)...")

                pv_result = step_ent["Prover_GRPO_precompute_status"]["precomputed_results_path"]
                GRPO_training_config_prover = GRPO_Training_config(
                    training_idx=f"{self.GAR_trainer_meta_config.training_name}_step_{step}_Prover_GRPO_training",
                    model_to_train_path=self.prover_writer_model_id,
                    training_model_name=f"Prover_step_{step}",
                    wandb_run_name=f"{self.GAR_trainer_meta_config.training_name}_step_{step}_Prover_GRPO_training",
                    train_data_path=pv_result,
                    eval_data_path=self.eval_data_path,
                    model_save_path=f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/trained_models/Prover_model_step_{step}",
                    model_ckpt_path=f"{self.GAR_trainer_meta_config.save_folder}/GAR_step_{step}/trained_models/Prover_model_step_{step}_ckpt",
                    ckpt_freq=int(1E9),
                    begin_idx=0,
                    logging_conf_file="./conf/common.log_conf",
                    use_wandb=True,
                    wandb_project="GAR_Prover_GRPO_training",
                )
                self._run_GAR_training_step(
                    step=step,
                    data_type="prover",
                    GRPO_trainingConfig=GRPO_training_config_prover,
                    GRPO_trainingHyperparams=self.GRPO_training_hyperparams, 
                    # Harded coded for DeepSeek-Prover, change when using Qwen3 based models!!!!
                    fsdp_config_path=PROVER_FSDP_FILENAME
                )
                step_ent["Prover_GRPO_training_status"]["finished"] = True
                step_ent["Prover_GRPO_training_status"]["model_path"] = GRPO_training_config_prover.model_save_path
                self.ckpt_dict["prover_writer_model_id"] = GRPO_training_config_prover.model_save_path
                self.prover_writer_model_id = GRPO_training_config_prover.model_save_path
                self.save_ckpt()
                
            print(f"[RESUME] Step {step} finished.")
            step += 1
            self.ckpt_dict["step_idx"] = step
            self.save_ckpt()

            if step >= total_steps:
                print("[RESUME] All steps finished.")
                break

            # initialize the forward worker for the next step
            self.GAR_forward_node_worker = formulate_GAR_forward_node_worker(
                GAR_trainer_meta_config=self.GAR_trainer_meta_config,
                Lean_server_config=self.Lean_server_config,
                step_idx=step,
                avail_cuda_devices=self.avail_cuda_devices,
                sub_experiment_idx=list(range(len(self.avail_cuda_devices))),
                NL_fuser_model_id=self.NL_fuser_model_id,
                autoformalizer_model_id=self.autoformalizer_model_id,
                prover_writer_model_id=self.prover_writer_model_id,
                gpu_utilization_rate=self.GAR_trainer_forward_config.gpu_utilization_rate,
                NL_fuser_main_file_name=self.GAR_trainer_forward_config.NL_fuser_main_file_name,
                NL_fuser_skip_thinking=self.GAR_trainer_forward_config.NL_fuser_skip_thinking,
                NL_fuser_max_tokens=self.GAR_trainer_forward_config.NL_fuser_max_tokens,
                NL_fuser_batch_size=self.GAR_trainer_forward_config.NL_fuser_batch_size,
                NL_fuser_micro_batch_size=self.GAR_trainer_forward_config.NL_fuser_micro_batch_size,
                NL_fuser_temperature=self.GAR_trainer_forward_config.NL_fuser_temperature,
                autoformalizer_main_file_name=self.GAR_trainer_forward_config.autoformalizer_main_file_name,
                prove_writer_main_file_name=self.GAR_trainer_forward_config.prove_writer_main_file_name,
                prove_writer_max_length=self.GAR_trainer_forward_config.prove_writer_max_length,
                prove_writer_batch_size=self.GAR_trainer_forward_config.prove_writer_batch_size,
                prove_writer_temperature=self.GAR_trainer_forward_config.prove_writer_temperature, 
                prove_writer_slurm_config=self.GAR_trainer_forward_config.prove_writer_slurm_config
            )