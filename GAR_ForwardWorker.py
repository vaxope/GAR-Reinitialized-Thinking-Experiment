from StatementFuser import StatementFuserMetaConfig, NLStatementFuserConfig, AutoformalizerConfig, StatementCheckerConfig, StatementFuser
from LeanChecker import LeanChecker
from ProveWriter import ProveWriter
from utils import utils
from dataclasses import dataclass, field
import subprocess
import signal
import sys
from typing import Dict
import os
from pathlib import Path

child_process = None

def signal_handler(sig, frame):
    global child_process
    print(f"Received signal {sig}, terminating child process...")
    
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

    # 捕获信号
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        child_process = subprocess.Popen(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=True,
            preexec_fn=os.setsid  
        )

        child_process.wait()

        if child_process.returncode != 0:
            raise subprocess.CalledProcessError(child_process.returncode, command)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command '{e.cmd}' failed with exit code {e.returncode}")

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
        
@dataclass
class StatementFuserConfig:
    statement_fuser_meta_config: StatementFuserMetaConfig = field(default_factory=StatementFuserMetaConfig)
    nl_statement_fuser_config: NLStatementFuserConfig = field(default_factory=NLStatementFuserConfig)
    autoformalizer_config: AutoformalizerConfig = field(default_factory=AutoformalizerConfig)
    statement_checker_config: StatementCheckerConfig = field(default_factory=StatementCheckerConfig)
    

@dataclass
class GAR_ForwardWorkerMetaConfig:
    experiment_name: str = "GAR_ForwardWorker"
    save_dir_path: str = "./GAR_ForwardWorker_output"
    base_dataset_path: str = "./Goedel_dataset_testing.json"
    cuda_device_id: int = 0
    gpu_memory_utilization: float = 0.95
    generated_statement_num: int = 128
    num_base_statement_per_fuse: int = 2

@dataclass
class ProveWriterSLURMConfig:
    useMoreSlurmGpu: bool = False
    numSlurmJobs: int = 3
    slurmPartition: str = "ghx4"
    slurmTime: str = "4:00:00"
    slurmMem: str = "64g"
    slurmCpus: int = 64
    slurmGPUsPerJob: int = 1
    slurmJobNamePrefix: str = "ProveWriter"

@dataclass
class ProveWriterConfig:
    main_file_name: str = "ProveWriter.py"
    model_id: str = "Goedel-LM/Goedel-Prover-V2-8B"
    global_begin_idx: int = 0
    global_end_idx: int = -1
    local_begin_idx: int = 0
    local_end_idx: int = -1
    proof_num: int = 32
    temperature: float = 0.8
    top_p: float = 0.95
    batch_size: int = 4
    max_tokens: int = 16384
    repetition_penalty: float = 1.0
    slurm_config: ProveWriterSLURMConfig = field(default_factory=ProveWriterSLURMConfig)

@dataclass
class ProofVerifierConfig:
    main_file_name: str = "LeanChecker.py"
    Lean4_server_url: str = "http://localhost:80"
    batch_size: int = 16
    timeout: int = 240


class GAR_ForwardWorker:
    def __init__(self, 
                 statement_fuser_config: StatementFuserConfig = StatementFuserConfig(), 
                 GAR_ForwardWorker_meta_config: GAR_ForwardWorkerMetaConfig = GAR_ForwardWorkerMetaConfig(), 
                 prove_writer_config: ProveWriterConfig = ProveWriterConfig(), 
                 proof_verifier_config: ProofVerifierConfig = ProofVerifierConfig()):
        """Initialize the GAR_ForwardWorker with the stated configurations. 
        Note that we are going to overriding the StatementFuserMetaConfig since this has too much similarities 
        with the GAR_ForwardWorkerMetaConfig. Additionally, the global begin idx and end idx should not be used in this setting since 
        this is a generated dataset and not concerning to eval a big dataset in multiple parts"""
        self.statement_fuser_config = statement_fuser_config
        self.GAR_ForwardWorker_meta_config = GAR_ForwardWorker_meta_config
        self.prove_writer_config = prove_writer_config
        self.proof_verifier_config = proof_verifier_config
        self.statement_fuser_config.statement_fuser_meta_config = StatementFuserMetaConfig(
            base_dataset_path=self.GAR_ForwardWorker_meta_config.base_dataset_path, 
            save_dir_path=f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/StatementFuser_files",
            cuda_device_id=self.GAR_ForwardWorker_meta_config.cuda_device_id,
            generated_statement_num=self.GAR_ForwardWorker_meta_config.generated_statement_num,
            num_base_statement_per_fuse=self.GAR_ForwardWorker_meta_config.num_base_statement_per_fuse
        )
        self.statement_fuser_config.statement_fuser_meta_config.cuda_device_id = self.GAR_ForwardWorker_meta_config.cuda_device_id
        self.statement_fuser_config.statement_fuser_meta_config.generated_statement_num = self.GAR_ForwardWorker_meta_config.generated_statement_num
        self.statement_fuser_config.statement_fuser_meta_config.num_base_statement_per_fuse = self.GAR_ForwardWorker_meta_config.num_base_statement_per_fuse

        # Initialize the logging and checkpoint system
        utils.check_folder_exit(self.GAR_ForwardWorker_meta_config.save_dir_path)
        utils.check_folder_exit(f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/ProofWriter_files")
        utils.check_folder_exit(f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/ProofWriter_files/ckpt")
        utils.check_folder_exit(f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/ProofVerifier_files")
        utils.check_folder_exit(f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/ProofVerifier_files/ckpt")
        utils.check_folder_exit(f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/StatementFuser_files/")
        utils.check_folder_exit(f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/logs")

        # Initialize the StatementFuser
        self.statement_fuser = StatementFuser(
            meta_config=self.statement_fuser_config.statement_fuser_meta_config,
            nl_fuser_config=self.statement_fuser_config.nl_statement_fuser_config,
            autoformalizer_config=self.statement_fuser_config.autoformalizer_config,
            statement_checker_config=self.statement_fuser_config.statement_checker_config
        )

    def _formulate_bash_command_prove_writer(self, log_file_name: str = "proof_writing"):
        """Formulate the bash command for the ProveWriter. We will organize all the fused, 
        and passed verification statements into a single file named `fused_statements.json` in the
        `GAR_ForwardWorker_meta_config` directory. The ProveWriter will then read this file and generate proofs for each statement."""
        
        def _transform_to_abs_path(path: str) -> str:
            return str(Path(path).resolve(strict=False))
        
        # Ensure the save path exists
        if self.prove_writer_config.slurm_config.useMoreSlurmGpu:
            print("Using SLURM to manage multiple GPUs for ProveWriter...")
            # log dir will be in absolute path format
            log_dir = f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/logs"
            log_dir = str(Path(log_dir).resolve())
            err_path = f"{log_dir}/{log_file_name}.err"
            out_path = f"{log_dir}/{log_file_name}.log"
            slurm_log_dir = f"{log_dir}/slurm_logs"
            utils.check_folder_exit(slurm_log_dir)

            curr_dataset_path = _transform_to_abs_path(f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/fused_statements.json")
            curr_model_id = _transform_to_abs_path(self.prove_writer_config.model_id)
            curr_save_dir_path = _transform_to_abs_path(self.GAR_ForwardWorker_meta_config.save_dir_path)

            command = f"python -u {self.prove_writer_config.main_file_name} " \
                    f"--dataset_path {curr_dataset_path} " \
                    f"--cuda_device_id {self.GAR_ForwardWorker_meta_config.cuda_device_id} " \
                    f"--gpu_memory_utilization {self.GAR_ForwardWorker_meta_config.gpu_memory_utilization} " \
                    f"--model_id {curr_model_id} " \
                    f"--save_path {curr_save_dir_path}/ProofWriter_files/ " \
                    f"--ckpt_path {curr_save_dir_path}/ProofWriter_files/ckpt " \
                    f"--global_begin_idx {self.prove_writer_config.global_begin_idx} " \
                    f"--global_end_idx {self.prove_writer_config.global_end_idx} " \
                    f"--proof_num {self.prove_writer_config.proof_num} " \
                    f"--temperature {self.prove_writer_config.temperature} " \
                    f"--top_p {self.prove_writer_config.top_p} " \
                    f"--batch_size {self.prove_writer_config.batch_size} " \
                    f"--max_tokens {self.prove_writer_config.max_tokens} " \
                    f"--repetition_penalty {self.prove_writer_config.repetition_penalty} " \
                    f"--use_more_slurm_gpu " \
                    f"--num_slurm_jobs {self.prove_writer_config.slurm_config.numSlurmJobs} " \
                    f"--slurm_partition {self.prove_writer_config.slurm_config.slurmPartition} " \
                    f"--slurm_time {self.prove_writer_config.slurm_config.slurmTime} " \
                    f"--slurm_mem {self.prove_writer_config.slurm_config.slurmMem} " \
                    f"--slurm_cpus {self.prove_writer_config.slurm_config.slurmCpus} " \
                    f"--slurm_gpus_per_job {self.prove_writer_config.slurm_config.slurmGPUsPerJob} " \
                    f"--slurm_job_name_prefix {self.prove_writer_config.slurm_config.slurmJobNamePrefix} " \
                    f"--slurm_log_dir {slurm_log_dir} " \
                    f"> {out_path} " \
                    f"2> {err_path}"
            command = (
                f"(echo 'PID: $$' >&2; exec {command})"
                f" > {out_path} 2> {err_path}"
            )
        else:
            print("Using single GPU for ProveWriter...")
            log_dir = f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/logs"
            err_path = f"{log_dir}/{log_file_name}.err"
            out_path = f"{log_dir}/{log_file_name}.log"
            # May add in_verification here
            command = f"python -u {self.prove_writer_config.main_file_name} " \
                    f"--dataset_path {self.GAR_ForwardWorker_meta_config.save_dir_path}/fused_statements.json "\
                    f"--cuda_device_id {self.GAR_ForwardWorker_meta_config.cuda_device_id} " \
                    f"--gpu_memory_utilization {self.GAR_ForwardWorker_meta_config.gpu_memory_utilization} " \
                    f"--model_id {self.prove_writer_config.model_id} " \
                    f"--save_path {self.GAR_ForwardWorker_meta_config.save_dir_path}/ProofWriter_files/ " \
                    f"--ckpt_path {self.GAR_ForwardWorker_meta_config.save_dir_path}/ProofWriter_files/ckpt " \
                    f"--global_begin_idx {self.prove_writer_config.global_begin_idx} " \
                    f"--global_end_idx {self.prove_writer_config.global_end_idx} " \
                    f"--proof_num {self.prove_writer_config.proof_num} " \
                    f"--temperature {self.prove_writer_config.temperature} " \
                    f"--top_p {self.prove_writer_config.top_p} " \
                    f"--batch_size {self.prove_writer_config.batch_size} " \
                    f"--max_tokens {self.prove_writer_config.max_tokens} " \
                    f"--repetition_penalty {self.prove_writer_config.repetition_penalty} " \
                    f"> {out_path} " \
                    f"2> {err_path}"
            # 关键：用子 shell，将 PID 写进 stderr，然后 exec 真正的命令
            command = (
                f"(echo 'PID: $$' >&2; exec {command})"
                f" > {out_path} 2> {err_path}"
            )

        return command
    
    def _formulate_bash_command_proof_verifier(self, log_file_name: str = "proof_verification"):
        """Formulate the bash command for the proof verification, we are going to put all the verification results into a 
        file named `verified_proofs.json` in the `GAR_ForwardWorker_meta_config` directory."""

        log_dir = f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/logs"
        err_path = f"{log_dir}/{log_file_name}.err"
        out_path = f"{log_dir}/{log_file_name}.log"

        command = f"python -u {self.proof_verifier_config.main_file_name} " \
                  f"--verify_type proof " \
                  f"--dataset_path {self.GAR_ForwardWorker_meta_config.save_dir_path}/ProofWriter_files/ckpt " \
                  f"--Lean4_server_url {self.proof_verifier_config.Lean4_server_url} " \
                  f"--save_path {self.GAR_ForwardWorker_meta_config.save_dir_path}/ProofVerifier_files/verified_proofs.json " \
                  f"--ckpt_path {self.GAR_ForwardWorker_meta_config.save_dir_path}/ProofVerifier_files/ckpt " \
                  f"--batch_size {self.proof_verifier_config.batch_size} " \
                  f"--timeout {self.proof_verifier_config.timeout} " \
                  f"> {out_path} " \
                  f"2> {err_path}"

        # 关键：用子 shell，将 PID 写进 stderr，然后 exec 真正的命令
        command = (
            f"(echo 'PID: $$' >&2; exec {command})"
            f" > {out_path} 2> {err_path}"
        )
        return command
    
    def run_statement_fuser(self, skip_statement_fusion=False):
        """Run the statement fuser to fuse statements and save the results."""
        print("Running Statement Fuser...")
        if not skip_statement_fusion:
            self.statement_fuser.run()
            print("Statement Fuser run completed.")
        else:
            print("Skipping statement fusion, make sure you have fused statements in " + f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/fused_statements.json")
        
        pass_verified_statements = utils.read_json_in_folder(f"{self.statement_fuser.meta_config.save_dir_path}/verified_results_files/ckpt/pass_statements/")
        utils.write_to_json(f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/fused_statements.json", pass_verified_statements)
        print("Fused statements saved to:", f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/fused_statements.json")

    def run_prove_writer(self):
        """Run the Prove Writer to generate proofs for the fused statements."""
        print("Running Prove Writer...")
        prove_writer_command = self._formulate_bash_command_prove_writer()
        print("Executing command:", prove_writer_command)
        run_command(prove_writer_command)
        print("Proof generation completed. Proofs saved to:", f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/ProofWriter_files/ckpt")

    def run_proof_verifier(self):
        """Run the Proof Verifier to verify the generated proofs."""
        print("Running Proof Verifier...")
        proof_verifier_command = self._formulate_bash_command_proof_verifier()
        print("Executing command:", proof_verifier_command)
        run_command(proof_verifier_command)
        print("Proof verification completed. Verified proofs saved to:", f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/ProofVerifier_files/verified_proofs.json")
        
        verified_proofs = utils.read_from_json(f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/ProofVerifier_files/verified_proofs.json")
        utils.write_to_json(f"./{self.GAR_ForwardWorker_meta_config.save_dir_path}/verified_proofs.json", verified_proofs)

    def run(
            self, 
            skip_statement_fusion=False, 
            save_running_ckpt=True, 
            statement_fuser_finished=False, 
            prove_writer_finished=False,
            proof_verifier_finished=False
        ):
        """Run the entire process of GAR LLM inference forward worker for one round of statement fusion, proof generation, and proof verification."""

        running_ckpt = {
            "statement_fuser_finished": statement_fuser_finished, 
            "prove_writer_finished": prove_writer_finished, 
            "proof_verifier_finished": proof_verifier_finished,
        }

        print("Starting GAR Forward Worker...")


        if not statement_fuser_finished:
            # Step 1: Fuse statements
            print("Fusing statements...")
            if not skip_statement_fusion:
                self.statement_fuser.run()
                running_ckpt["statement_fuser_finished"] = True
            else:
                running_ckpt["statement_fuser_finished"] = True
                UserWarning("Skipping the statement fusion stage, make sure you have fused statement in " + f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/fused_statements.json")

            pass_verified_statements = utils.read_json_in_folder(f"{self.statement_fuser.meta_config.save_dir_path}/verified_results_files/ckpt/pass_statements/")
            utils.write_to_json(f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/fused_statements.json", pass_verified_statements)
            print("Statement fusion completed. Fused statements saved to:", f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/fused_statements.json")
            if save_running_ckpt:
                utils.write_to_json(f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/running_ckpt.json", running_ckpt)

        if not prove_writer_finished:
            # Step 2: Generate proofs for the fused statements
            print("Generating proofs for the fused statements...")
            prove_writer_command = self._formulate_bash_command_prove_writer()
            print("Executing command:", prove_writer_command)
            run_command(prove_writer_command)
            print("Proof generation completed.")
            running_ckpt["prove_writer_finished"] = True
            if save_running_ckpt:
                utils.write_to_json(f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/running_ckpt.json", running_ckpt)

        if not proof_verifier_finished:
            # Step 3: Verify the generated proofs
            print("Verifying the generated proofs...")
            proof_verifier_command = self._formulate_bash_command_proof_verifier()
            print("Executing command:", proof_verifier_command)
            run_command(proof_verifier_command)
            print("Proof verification completed. Verified proofs saved to:", f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/ProofVerifier_files/verified_proofs.json")
            running_ckpt["proof_verifier_finished"] = True
            if save_running_ckpt:
                utils.write_to_json(f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/running_ckpt.json", running_ckpt)

        print("Extracting proof results")
        verified_proofs = utils.read_from_json(f"{self.GAR_ForwardWorker_meta_config.save_dir_path}/ProofVerifier_files/verified_proofs.json")
        utils.write_to_json(f"./{self.GAR_ForwardWorker_meta_config.save_dir_path}/verified_proofs.json", verified_proofs)

        print("GAR Forward Worker run completed.")