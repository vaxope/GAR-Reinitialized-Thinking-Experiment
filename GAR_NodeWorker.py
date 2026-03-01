from StatementFuser import StatementFuserMetaConfig, NLStatementFuserConfig, AutoformalizerConfig, StatementCheckerConfig, StatementFuser
from GAR_ForwardWorker import GAR_ForwardWorker, GAR_ForwardWorkerMetaConfig, ProveWriterConfig, ProofVerifierConfig, StatementFuserConfig, ProveWriterSLURMConfig
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



@dataclass
class GAR_ForwardWorkerConfig:
    """
    The wrap-up for configuration for the GAR Forward Worker.
    """
    statement_fuser_config: StatementFuserConfig
    forward_worker_meta_config: GAR_ForwardWorkerMetaConfig
    prove_writer_config: ProveWriterConfig
    proof_verifier_config: ProofVerifierConfig

class GAR_NodeWorker:
    def __init__(self, 
                 experiment_name: str,
                 avail_cuda_devices: List[int], 
                 gpu_utilization_rate: float,
                 experiment_file_root_path: str, 
                 experiment_idx_ls: List[str], 
                 base_config: GAR_ForwardWorkerConfig 
        ):
        """
        This class is the worker for managing the whole node of the GAR forward inference
        :param experiment_name: The name of the experiment, it will be used to form the path for the experiment.
        :param avail_cuda_devices: List of available CUDA devices that can be used for computation.
        :param gpu_utilization_rate: The portion of each single GPU can be used for computation.
        :param experiment_file_root_path: The root path for this stage of experiment, it will be concated with experiment_idx_lst to form the full path.
        :param experiment_idx_ls: List of experiment indices, normally, it is composed of `step-x_y`, where x means the step number and y means the portion. Each index will be used to form a unique path for the experiment.
        :param base_config: The base configuration of the GAR ForwardWorker, the most important function of this is to provide the config for inference and which model to use for specific stage. The following parts will be overwitten:
            - GAR_ForwardWorkerConfig.forward_worker_meta_config.experiment_name: experiment_name
            - GAR_ForwardWorkerConfig.forward_worker_meta_config.save_dir_path: f"{experiment_file_root_path}_{experiment_idx_ls[i]}"
            - GAR_ForwardWorkerConfig.statement_fuser_config.statement_fuser_meta_config: entire config will be overwritten, because it is bascially depend on the forward worker config
        """

        assert len(avail_cuda_devices) > 0, "No available CUDA devices found."
        assert len(avail_cuda_devices) == len(experiment_idx_ls), "The number of available CUDA devices must match the number of experiment indices."
        assert gpu_utilization_rate > 0 and gpu_utilization_rate <= 1, "GPU utilization rate must be between 0 and 1."

        base_config.forward_worker_meta_config.experiment_name = experiment_name

        config_list = []
        for i in range(len(experiment_idx_ls)):
            curr_config = copy.deepcopy(base_config)
            curr_config.forward_worker_meta_config.save_dir_path = f"{experiment_file_root_path}_{experiment_idx_ls[i]}"
            curr_config.forward_worker_meta_config.cuda_device_id = avail_cuda_devices[i]
            curr_config.forward_worker_meta_config.gpu_memory_utilization = gpu_utilization_rate

            curr_config.statement_fuser_config.statement_fuser_meta_config = StatementFuserMetaConfig(
                base_dataset_path=curr_config.forward_worker_meta_config.base_dataset_path,
                save_dir_path=f"{curr_config.forward_worker_meta_config.save_dir_path}/StatementFuser_files",
                cuda_device_id=curr_config.forward_worker_meta_config.cuda_device_id,
                generated_statement_num=curr_config.forward_worker_meta_config.generated_statement_num,
                num_base_statement_per_fuse=curr_config.forward_worker_meta_config.num_base_statement_per_fuse
            )
            config_list.append(curr_config)

        forward_worker_ls: List[GAR_ForwardWorker] = []
        for curr_config in config_list:
            curr_forward_worker = GAR_ForwardWorker(
                statement_fuser_config=curr_config.statement_fuser_config,
                GAR_ForwardWorker_meta_config=curr_config.forward_worker_meta_config,
                prove_writer_config=curr_config.prove_writer_config,
                proof_verifier_config=curr_config.proof_verifier_config
            )

            forward_worker_ls.append(curr_forward_worker)
        self.config_ls: List[GAR_ForwardWorkerConfig] = config_list
        self.forward_worker_ls = forward_worker_ls
        self.experiment_name = experiment_name

    def _run_worker_subprocess(
            self, 
            i: int, 
            statement_fuser_finished=False, 
            prove_writer_finished=False,
            proof_verifier_finished=False
        ):
        """
        This function will do the following things:
        1. Binding the GPU for given worker index i.
        2. Open two log files for subprocesses and add the buffer for each lines.
        3. Redirect the sys.stdout and sys.stderr to the log files.
        4. Run the worker by finding the worker and run it.

        :param i: The index of the worker to run, indicating which position of list of worker should we use. It is different from the index of the config list.
        """

        import random, time, os
        random.seed(time.time_ns() + os.getpid())

        config = self.config_ls[i]

        # Binding the GPU for given worker index i.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.forward_worker_meta_config.cuda_device_id)

        # ready the log path
        log_dir = os.path.join(config.forward_worker_meta_config.save_dir_path, f"worker_{i}.log")
        err_dir = os.path.join(config.forward_worker_meta_config.save_dir_path, f"worker_{i}.err")

        # open the file line and redirect
        file_out = open(log_dir, "w", buffering=1, encoding="utf-8")
        file_err = open(err_dir, "w", buffering=1, encoding="utf-8")
        sys.stdout = file_out
        sys.stderr = file_err

        # Run the worker
        # self.forward_worker_ls[i].run_proof_verifier()
        self.forward_worker_ls[i].run(
            skip_statement_fusion=False, 
            statement_fuser_finished=statement_fuser_finished,
            prove_writer_finished=prove_writer_finished,
            proof_verifier_finished=proof_verifier_finished
        )
        
        print(f"Worker {i} finished running.")

    def run_all_workers(
            self, 
            statement_fuser_finished=False, 
            prove_writer_finished=False,
            proof_verifier_finished=False
        ):
        """
        Start a new subprocess for each config, every subprocess will redirect their out and err into their own log files.
        """

        process_ls = []

        # Register all singal handlers
        def _singal_handler(sig, frame):
            print(f"Received signal {sig}, terminating all workers...")
            for p in process_ls:
                if p.is_alive():
                    p.terminate()
                    print(f"Worker {p.pid} terminated.")
            sys.exit(0)
        signal.signal(signal.SIGINT, _singal_handler)
        signal.signal(signal.SIGTERM, _singal_handler)

        for i in range(len(self.config_ls)):
            p = multiprocessing.Process(target=self._run_worker_subprocess, args=(
                i, 
                statement_fuser_finished,
                prove_writer_finished,
                proof_verifier_finished
            ))
            p.daemon = True 
            p.start()
            process_ls.append(p)
            print(f"Worker {i} started with PID: {p.pid}")
            time.sleep(1)  # Give some time for the worker to start
        
        # wait for all subprocesses to finish
        for p in process_ls:
            p.join()
        print("All workers have finished running.")

STEP_NUM = 0
EXPERIMENT_NAME = "stat_fusion_by_type_test"
AVAIL_CUDA_DEVICES = [0,1,2,3,4,5,6,7]  
SUB_EXPERRIMENT_IDX =[0,1,2,3,4,5,6,7]  # This is the sub-experiment index, it will be used to form the path for each experiment.
EXPERIMENT_IDX_LS = [f"step-{STEP_NUM}_{i}" for i in SUB_EXPERRIMENT_IDX]
GPU_UTILIZATION_RATE = 0.95
EXPERIMENT_FILE_ROOT_PATH = f"./GAR_Forward_worker_test/stat_fusion_by_type_test/Step_{STEP_NUM}/save_files"
BASE_STATEMENT_PATH = "../data/Base_dataset/NuminaMath_Lean_dataset.json"

# testing main function
def main():
    # Example usage
    GAR_ForwardWorker_meta_config = GAR_ForwardWorkerMetaConfig(
        experiment_name="stat_fusion_by_type_test",
        save_dir_path="./GAR_ForwardWorker_test/stat_fusion_by_type_test",
        base_dataset_path=BASE_STATEMENT_PATH,
        cuda_device_id=0, 
        gpu_memory_utilization=0.95,
        generated_statement_num=128,
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
        max_tokens=16384, 
        batch_size=16, 
        micro_batch_size=16, 
        temperature=0.8, 
        repetition_penalty=1.0, 
        top_p=0.95
    )

    autoformalizer_config = AutoformalizerConfig(
        main_file_name="./Autoformalizer.py",
        model_id="../data/models/Goedel-Formalizer-V2-8B",
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
        batch_size=1024, 
        max_tokens=16384, 
        slurm_config=ProveWriterSLURMConfig(
            useMoreSlurmGpu=False  # 确保使用本地 GPU
        )
    )

    # Proof verifier config
    proof_verifier_config = ProofVerifierConfig(
        main_file_name="LeanChecker.py", 
        Lean4_server_url="http://localhost:80",
        batch_size=1024,
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
    gar_node_worker.run_all_workers(
        statement_fuser_finished=True, 
        prove_writer_finished=True,
        proof_verifier_finished=False
    )

if __name__ == "__main__":
    main()