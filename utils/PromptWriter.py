"""
Set of helper functions for writing prompts
"""
from utils import utils
from transformers import PreTrainedTokenizer
from typing import List, Dict

Lean4_HEADER = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
"""

KIMINA_PROVER_SYS_PROMPT = """You are an expert in mathematics and Lean 4."""

OpenR1_SYS_PROMPT = """You are an expert in Lean4 theorem proving with exceptional strategic reasoning abilities. When solving problems, strictly follow this process:
1. First create a natural language proof draft explaining key insights and logical steps
2. Then analyze required Lean4 tactics, specifying exact syntax and their logical purpose"""


def formulate_prompt_autoformalization_Goedel(
        name: str,
        problem_to_autoformalize: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None, 
):
    """
    This is the helper function for writing the prompt to autoformalize the problem into Lean4 code using Goedel-Autoformalizer model.
    """

    user_prompt = f"""Please autoformalize the following natural language problem statement in Lean 4. Use the following theorem name: {name}
The natural language statement is: {problem_to_autoformalize}"""

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def formulate_prompt_proof_writing_Goedel_V2_verification_mode(
        NL_statement: str,
        Lean4_statement: str,
        tokenizer: PreTrainedTokenizer, 
        Lean4_header = None
):
    """
    This function formulates the prompt for proof writing using Goedel V2 in verification mode. Different from the 
    normal mode, in verification mode, we will add an extra instruction to prevent the model from modifying the statement.
    :param NL_statement: The natural language statement of the theorem.
    :param Lean4_statement: The Lean4 statement of the theorem.
    :param tokenizer: The tokenizer to be used for formatting the prompt.
    """
    if Lean4_header is None or Lean4_HEADER == "":
        Lean4_header = Lean4_HEADER

    prompt = f"""Complete the following Lean 4 code:

```lean4
{Lean4_header}

{utils.preprocess_theorem_statement(Lean4_statement)}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def formulate_prompt_proof_writing_Goedel_V2(
        NL_statement: str, 
        Lean4_statement: str, 
        tokenizer: PreTrainedTokenizer, 
        Lean4_header = None
):
    """
    This function formulates the prompt for proof writing using Goedel V2.
    :param NL_statement: The natural language statement of the theorem.
    :param Lean4_statement: The Lean4 statement of the theorem.
    :param tokenizer: The tokenizer to be used for formatting the prompt.
    :return: The formatted prompt string.
    """
    if Lean4_header is None or Lean4_HEADER == "":
        Lean4_header = Lean4_HEADER
    

    prompt = f"""Complete the following Lean 4 code:

```lean4
{Lean4_header}


/-- {NL_statement} -/
{utils.preprocess_theorem_statement(Lean4_statement)}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def formulate_prompt_fuse_statements(
        base_statement_ls: List[str],
        tokenizer: PreTrainedTokenizer, 
        skip_thinking: bool = False
    ) -> str:
    """
    The function to formulate the prompt for fusing two statements.
    :param base_statement_ls: A list of two base statements to be fused, it should contain at least two strings
    :param tokenizer: The tokenizer to be used for formatting the prompt
    :param skip_thinking: Whether to skip the thinking process in the prompt
    :return: the prompt for fusing two statements
    """
    assert len(base_statement_ls) >= 2, "At least two base statements are required to fuse."

    problem = f"""You are an expert mathematics educator skilled in problem design. Your task is to combine multiple given problems into a SINGLE, more challenging problem by combining their key elements. Follow these steps: 
    
Please first do the following steps as your analysis process enclosed within <analysis></analysis>.
1. Analyze the points of knowledge that need to be used when solving the proof problem and identify overlapping or complementary aspects (e.g., shared topic areas or contrasting difficulty levels)
2. Draft the new problem that integrates at least 2 key components from each original problem and make sure the new problem requires multi-step reasoning (e.g., combining algebraic manipulation with probabilistic analysis). Also, your combined problem should have no non-trivial extension.
3. Additionally, you should make sure that the new problem is solvable.

After your analysis, you should put the new problem into an MD code block. The new problem should be a SINGLE proof problem.

Here are the statements you need to fuse:"""
    for i, base_statement in enumerate(base_statement_ls):
        problem += f"\n\nProblem {i+1}:\n```md\n{base_statement}\n```"
    messages = [
        {"role": "system", "content": "You are an expert in mathematics and are good at problem design and reformulation."},
        {"role": "user", "content": problem}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    if skip_thinking:
        text += "\n<think>\n\n</think>\n"
    return text

def formulate_postLongCoT_prompt(
        post_LongCoT_content: str,
        NL_statement: str,
        FL_statement: str,
        output_begin_sign: str = "",
        pre_Lean4_codeblock_content: str = "",
        pre_statement_content: str = Lean4_HEADER,
) -> str:
    """
    The function to formulate post Long CoT prompt to perform Long CoT control generation. It's major usage is to boost the performance of Lean4 inference
    :param post_LongCoT_content: the previous sequence before the output of Long CoT content, it should contain the Long CoT stop sign
    :param NL_statement: Natural language statement for the theorem
    :param FL_statement: Lean4 statement for the theorem
    :param output_begin_sign: begin sign for output, for DeepSeek-R1, it should be none
    :param pre_Lean4_codeblock_content: The content before lean4 statement code-block
    :param pre_statement_content: the content before the starting of Lean4 statement that we need to prove
    :return: formatted post Long CoT prompt to query the model
    """
    return f"""{post_LongCoT_content}
{output_begin_sign}
{pre_Lean4_codeblock_content}
```lean4
{pre_statement_content}

/--{NL_statement}-/
{utils.preprocess_theorem_statement(FL_statement)}
"""

def formulate_prompt_LoTSolver(
        NL: str,
        thm_name: str,
        FL_statement: str,
        system_prompt: str=OpenR1_SYS_PROMPT
) -> str:
    """
    This function formulate the prompt for the LoT-Solver
    :param NL: NL statement of theorem
    :param thm_name: Name of theorem
    :param FL_statement: FL statement
    :return: formulated prompt
    """
    prompt =  f"""<｜begin▁of▁sentence｜>{system_prompt}
### Instruction: Please solve this Lean4 problem by completing both sections below:

NL Proof Draft (Natural Language)
Explain the proof strategy using mathematical reasoning and high-level steps. Consider:
- Key lemmas/theorems to apply
- Structural decomposition approaches
- Critical logical dependencies
- Potential proof patterns/methods

Lean4 Tactics Analysis (Technical Specification)
Identify concrete tactics needed to implement the proof, including:
1. Tactic name and syntax template
2. Purpose within proof context 
3. Expected goal state before/after application
4. Alternative tactics considered
"""
    prompt += f"""@ Natural language theorem statement:
{thm_name}:
{NL}

@ Lean4 theorem statement:
```lean4
{utils.preprocess_theorem_statement(FL_statement)}
```&


@ Lean4 theorem statement and proof with explanatory comments preceding each line of code:
### Response:
<think>
Okay, I should do the following:

  1. Provide the natural language analysis for the theorem based on the Natural language theorem statement.

  2. Draft the Lean4 tactics I should use to solve the problem

The user also asks that I should avoid using the keyword `sorry` or `admit` to give up the proof, so I will not write it in my Lean4 code.

The `{thm_name}` can be proofed by"""
    return prompt