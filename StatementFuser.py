from NLStatementFuser import NLStatementFuser
from LeanChecker import LeanChecker
from Autoformalizer import Autoformalizer
from dataclasses import dataclass
from utils import utils
import subprocess
import signal
import sys
import os

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
class StatementFuserMetaConfig:
    base_dataset_path: str = "./Goedel_dataset_testing.json"
    save_dir_path: str = "./StatementFuser_files"
    cuda_device_id: int = 0
    generated_statement_num: int = 128
    num_base_statement_per_fuse: int = 2

@dataclass
class NLStatementFuserConfig:
    main_file_name: str = "./NLStatementFuser.py"
    model_id: str = "../data/models/Qwen3-8B"
    skip_thinking: bool = True
    max_tokens: int = 8192
    batch_size: int = 16
    micro_batch_size: int = 4
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    top_p: float = 0.95

@dataclass
class AutoformalizerConfig:
    main_file_name: str = "./Autoformalizer.py"
    model_id: str = "../data/models/Kimina-Autoformalizer-7B"
    max_tokens: int = 8192
    batch_size: int = 4
    temperature: float = 0.6
    repetition_penalty: float = 1.0
    top_p: float = 0.95

@dataclass
class StatementCheckerConfig:
    main_file_name: str = "LeanChecker.py"
    Lean4_server_url: str = "http://localhost:80"
    batch_size: int = 128
    timeout: int = 120



class StatementFuser:
    def __init__(self, 
                 meta_config: StatementFuserMetaConfig = StatementFuserMetaConfig(),
                 nl_fuser_config: NLStatementFuserConfig = NLStatementFuserConfig(),
                 autoformalizer_config: AutoformalizerConfig = AutoformalizerConfig(),
                 statement_checker_config: StatementCheckerConfig = StatementCheckerConfig()):
        """Initialize the StatementFuser with configurations."""

        self.meta_config = meta_config
        self.nl_fuser_config = nl_fuser_config
        self.autoformalizer_config = autoformalizer_config
        self.statement_checker_config = statement_checker_config

        # Check if all the directories exist, if not, create them
        utils.check_folder_exit(self.meta_config.save_dir_path)
        utils.check_folder_exit(f"{self.meta_config.save_dir_path}/NL_fused_statements_files/ckpt")
        utils.check_folder_exit(f"{self.meta_config.save_dir_path}/formalized_statements_files/ckpt")
        utils.check_folder_exit(f"{self.meta_config.save_dir_path}/statement_check_files/ckpt")
        utils.check_folder_exit(f"{self.meta_config.save_dir_path}/logs/")   # All the logs for intermediate runs will be saved in this folder

    def _formulate_bash_command_NL_fusion(self, log_file_name="NL_statement_fuse"):
        """
        Formulate the bash command for running the NL statement fusion step.
        """

        command = f"nohup python -u {self.nl_fuser_config.main_file_name} " \
                  f"--base_dataset_path {self.meta_config.base_dataset_path} " \
                  f"--save_path {self.meta_config.save_dir_path}/NL_fused_statements.json" \
                  f" --cuda_device_id {self.meta_config.cuda_device_id} " \
                  f"--model_id {self.nl_fuser_config.model_id} " \
                  f"--generated_statement_num {self.meta_config.generated_statement_num} " \
                  f"--num_base_statement_per_fuse {self.meta_config.num_base_statement_per_fuse} " \
                  f"--ckpt_path {self.meta_config.save_dir_path}/NL_fused_statements_files/ckpt " \
                  f"--skip_thinking {self.nl_fuser_config.skip_thinking} " \
                  f"--max_tokens {self.nl_fuser_config.max_tokens} " \
                  f"--batch_size {self.nl_fuser_config.batch_size} " \
                  f"--micro_batch_size {self.nl_fuser_config.micro_batch_size} " \
                  f"--temperature {self.nl_fuser_config.temperature} " \
                  f"--top_p {self.nl_fuser_config.top_p} " \
                  f"--repetition_penalty {self.nl_fuser_config.repetition_penalty} "\
                  f"> {self.meta_config.save_dir_path}/logs/{log_file_name}.log "\
                  f"2> {self.meta_config.save_dir_path}/logs/{log_file_name}.err"
        return command

    def _formulate_bash_command_autoformalization(self, log_file_name="autoformalization"):
        """
        Formulate the bash command for running the autoformalization step.
        """

        command = f"nohup python -u {self.autoformalizer_config.main_file_name} " \
                  f"--dataset_path {self.meta_config.save_dir_path}/NL_fused_statements.json " \
                  f"--save_path {self.meta_config.save_dir_path}/formalized_statements.json " \
                  f"--begin_idx 0 --end_idx -1 " \
                  f"--cuda_device_id {self.meta_config.cuda_device_id} " \
                  f"--model_id {self.autoformalizer_config.model_id} " \
                  f"--ckpt_path {self.meta_config.save_dir_path}/formalized_statements_files/ckpt " \
                  f"--max_tokens {self.autoformalizer_config.max_tokens} " \
                  f"--batch_size {self.autoformalizer_config.batch_size} " \
                  f"--temperature {self.autoformalizer_config.temperature} " \
                  f"--top_p {self.autoformalizer_config.top_p} " \
                  f"--repetition_penalty {self.autoformalizer_config.repetition_penalty} "\
                  f"> {self.meta_config.save_dir_path}/logs/{log_file_name}.log "\
                  f"2> {self.meta_config.save_dir_path}/logs/{log_file_name}.err"
        return command
    
    def _formulate_bash_command_statement_checker(self, log_file_name="statement_checker"):
        """
        Formulate the bash command for running the statement checker step.
        """

        command = f"nohup python -u {self.statement_checker_config.main_file_name} " \
                  f"--verify_type statement " \
                  f"--dataset_path {self.meta_config.save_dir_path}/formalized_statements.json " \
                  f"--Lean4_server_url {self.statement_checker_config.Lean4_server_url} " \
                  f"--save_path {self.meta_config.save_dir_path}/verified_results.json " \
                  f"--ckpt_path {self.meta_config.save_dir_path}/verified_results_files/ckpt " \
                  f"--batch_size {self.statement_checker_config.batch_size} " \
                  f"--timeout {self.statement_checker_config.timeout} "\
                  f"> {self.meta_config.save_dir_path}/logs/{log_file_name}.log "\
                  f"2> {self.meta_config.save_dir_path}/logs/{log_file_name}.err"
        return command
    
    def run(self):
        """
        Run the entire statement fusion process.
        """

        # Step 1: Run the NL statement fusion
        print("Starting NL statement fusion...")
        command_NL_fusion = self._formulate_bash_command_NL_fusion()
        print(f"Running NL statement fusion with command: {command_NL_fusion}")
        run_command(command_NL_fusion)

        # Step 2: Run the autoformalization
        print("Starting autoformalization...")
        command_autoformalization = self._formulate_bash_command_autoformalization()
        print(f"Running autoformalization with command: {command_autoformalization}")
        run_command(command_autoformalization)

        # Step 3: Run the statement checker
        print("Starting statement checker...")
        command_statement_checker = self._formulate_bash_command_statement_checker()
        print(f"Running statement checker with command: {command_statement_checker}")
        run_command(command_statement_checker)

        print("All steps completed successfully. Results saved in:", self.meta_config.save_dir_path)

# Testing main
if __name__ == "__main__":
    # Initialize configs
    meta_config = StatementFuserMetaConfig(
        base_dataset_path="./Goedel_dataset_testing.json",
        save_dir_path="./StatementFuser_files/Experiment_0",
        cuda_device_id=0,
        generated_statement_num=128,
        num_base_statement_per_fuse=2
    )
    nl_fuser_config = NLStatementFuserConfig(
        main_file_name="./NLStatementFuser.py",
        model_id="../data/models/Qwen3-8B",
        skip_thinking=True,
        max_tokens=8192,
        batch_size=16,
        micro_batch_size=4,
        temperature=0.8,
        repetition_penalty=1.0,
        top_p=0.95
    )
    autoformalizer_config = AutoformalizerConfig(
        main_file_name="./Autoformalizer.py",
        model_id="../data/models/Kimina-Autoformalizer-7B",
        max_tokens=8192,
        batch_size=4,
        temperature=0.6,
        repetition_penalty=1.0,
        top_p=0.95
    )
    statement_checker_config = StatementCheckerConfig(
        main_file_name="LeanChecker.py",
        Lean4_server_url="http://localhost:80",
        batch_size=128,
        timeout=120
    )

    # Create the StatementFuser instance
    statement_fuser = StatementFuser(
        meta_config=meta_config,
        nl_fuser_config=nl_fuser_config,
        autoformalizer_config=autoformalizer_config,
        statement_checker_config=statement_checker_config
    )

    # Run the entire process
    statement_fuser.run()
    pass_verified_results = utils.read_json_in_folder(f"{statement_fuser.meta_config.save_dir_path}/verified_results_files/ckpt/pass_statements/")
    utils.write_to_json(f"{statement_fuser.meta_config.save_dir_path}/Mass_fusion_Statements.json", pass_verified_results)
    # run_command(f"cp {statement_fuser.meta_config.save_dir_path}/verified_results.json {statement_fuser.meta_config.save_dir_path}/")