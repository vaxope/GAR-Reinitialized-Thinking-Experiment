from transformers import PreTrainedModel
from dataclasses import dataclass

@dataclass
class MetaConfig:
    """
    The meta configration for the model

    Attributes:
        training_idx: The name for the training
    """
    training_idx: str

@dataclass
class DatasetConfig:
    """
    Configurations for the dataset for training
    """
    training_data_path: str
    eval_data_path: str

class ModelConfig:
    """
    Configurations for model setting

    Attributes:
        statement_writer: The statement writer model, it can be an autoformalizer
            (from NL statement to Lean4 statement) or a statement composer (combines two Lean4 statements to a harder one)
        prover: The based model for prover to train
    """
    statement_writer: PreTrainedModel
    prover: PreTrainedModel

@dataclass
class TrainingConfig:
    """
    Configurations for training

    Attributes:
        global_batch_size: The batch size for the entire machine.
        micro_batch_size: the batch size for every machine. If trained on an 8B model, recommend to be 1
    """
    global_batch_size: int = 4
    micro_batch_size: int = 1

@dataclass
class GrpoConfig:
    """
    Basic configuration for GRPO training

    Attributes:
        group_size: G in grpo, indicating how many roll-outs we need to do for a single question
        eps_clip: epsilon clip for GRPO
        beta_KL: The beta factor for KL divergence in GRPO
    """
    group_size: int = 8
    eps_clip: float = 0.2
    beta_KL: float = 0.01