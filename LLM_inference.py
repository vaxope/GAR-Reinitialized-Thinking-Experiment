from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
import torch
import re
import random
from tqdm.auto import tqdm
from utils import utils, PromptWriter
from utils.PromptWriter import Lean4_HEADER
from typing import List, Dict, Optional, Union, Tuple
from vllm.inputs import TokensPrompt
from vllm import LLM, SamplingParams
from dataclasses import dataclass

class LLM_rollout:
    """
    The inference component for GAR training
    """

    def __init__(
            self,
            model: LLM,
            tokenizer: PreTrainedTokenizer
    ):
        """
        The constructor for the LLM_rollout class, parameters list are as follows
        :param model: LLM model in the vllm package, we assume everything is set for the initialization
        :param tokenizer: tokenizer for the corresponding LLM for input
        :param terminators: the terminators for the model
        """
        self.model = model
        self.tokenizer = tokenizer

    def batched_query_model(
            self,
            prompts: List[str],
            max_tokens=4096,
            temperature=1.0,
            top_p=0.95,
            repetition_penalty=1.0,
            sampling_num=2,
            stop_strs: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Function to query model with given prompts and return the result string, we assume all the instruction (or other) formats
        is provided in the prompt. The function will return the format for prompt + response
        :param prompts: List of prompts to query
        :param max_tokens: The max tokens for the whole sequence (prompt + generated text)
        :param temperature: temperature for sampling, larger means the result will be more random
        :param top_p: top_p for sampling. It means only sampling from the top p portion tokens.
        :param repetition_penalty: The penalty for repetition during sampling, larger means the model will prefer not to sampling tokens pre-existed in the sentence
        :param sampling_num: The number to sample for each prompt, when prompt is [a, b, c] and `sampling_num=2`, the sampled result will be [a_1, a_2, b_1, b_2, c_1, c_2]
        :param stop_strs: List of strings to indicate the LLM to stop sampling
        :return: List of strings of the completed sentence (prompt + response format)
        """
        tokenized_prompts = []
        for curr in prompts:
            curr_tokenized_prompt = self.tokenizer(
                [curr],
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"][0].tolist()

            tokenized_prompts += [curr_tokenized_prompt]

        if stop_strs == None:
            sampling_para = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                n=sampling_num,
                skip_special_tokens=False
            )
        else:
            sampling_para = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=stop_strs,
                max_tokens=max_tokens,
                n=sampling_num,
                skip_special_tokens=False
            )

        outputs = self.model.generate(
            prompts=None,
            prompt_token_ids=tokenized_prompts,
            sampling_params=sampling_para
        )
        output_strings = []
        for curr in outputs:
            for curr_response in curr.outputs:
                output_strings += [curr_response.text]

        return_strings = [x for x in prompts for _ in range(sampling_num)]
        generated_token_num = []
        assert len(output_strings) == len(return_strings), \
            (f"Output strings(len: {len(output_strings)}) and prompts (len: {len(return_strings)}) "
             f"duplicated number not identical, maybe some unknown bug")

        for i in range(len(return_strings)):
            return_strings[i] += output_strings[i]
            generated_token_num.append(len(self.tokenizer(output_strings[i], add_special_tokens=False)["input_ids"]))
        

        return return_strings, generated_token_num

    def _batched_query_model(
            self,
            prompts: List[str],
            max_tokens=4096,
            temperature=1.0,
            top_p=0.95,
            repetition_penalty=1.0,
            sampling_num=2,
            stop_strs: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Function to query model with given prompts and return the result string, we assume all the instruction (or other) formats
        is provided in the prompt. The function will return the format for prompt + response
        :param prompts: List of prompts to query
        :param max_tokens: The max tokens for the whole sequence (prompt + generated text)
        :param temperature: temperature for sampling, larger means the result will be more random
        :param top_p: top_p for sampling. It means only sampling from the top p portion tokens.
        :param repetition_penalty: The penalty for repetition during sampling, larger means the model will prefer not to sampling tokens pre-existed in the sentence
        :param sampling_num: The number to sample for each prompt, when prompt is [a, b, c] and `sampling_num=2`, the sampled result will be [a_1, a_2, b_1, b_2, c_1, c_2]
        :param stop_strs: List of strings to indicate the LLM to stop sampling
        :return: List of strings of the completed sentence (prompt + response format)
        """
        tokenized_prompts = []
        for curr in prompts:
            curr_tokenized_prompt = self.tokenizer(
                [curr],
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"][0].tolist()

            tokenized_prompts += [curr_tokenized_prompt]

        if stop_strs == None:
            sampling_para = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                n=sampling_num,
                skip_special_tokens=False
            )
        else:
            sampling_para = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=stop_strs,
                max_tokens=max_tokens,
                n=sampling_num,
                skip_special_tokens=False
            )

        outputs = self.model.generate(
            prompts=None,
            prompt_token_ids=tokenized_prompts,
            sampling_params=sampling_para
        )
        output_strings = []
        for curr in outputs:
            for curr_response in curr.outputs:
                output_strings += [curr_response.text]

        return_strings = [x for x in prompts for _ in range(sampling_num)]
        assert len(output_strings) == len(return_strings), \
            (f"Output strings(len: {len(output_strings)}) and prompts (len: {len(return_strings)}) "
             f"duplicated number not identical, maybe some unknown bug")

        for i in range(len(return_strings)):
            return_strings[i] += output_strings[i]

        return return_strings

    def _batched_query_model_LongCoT_control(
            self,
            prompts: List[str],
            NL_statement: str,
            FL_statement: str,
            max_tokens=4096,
            temperature=1.0,
            top_p=0.9,
            repetition_penalty=1.0,
            sampling_num=2,
            LongCoT_stop_sign="</think>",
            output_begin_sign=""
    ) -> List[str]:
        """
        This function is for query the model with given prompt and return the result string, we assume that
        the format is provided in the prompt. We will always return the format for prompt + response
        :param prompts: the prompts for the model
        :param NL_statement: Natural language statement of the theorem
        :param FL_statement: Formal language statement of the theorem
        :param max_tokens: The max token for the whole sequence (prompt + generated text)
        :param temperature: temperature for sampling, larger means the result will be more random
        :param top_p: top_p for sampling. It means only sampling from the top p portion tokens.
        :param repetition_penalty: The penalty for repetition during sampling, larger means the model will prefer not to sampling tokens pre-existed in the sentence
        :param sampling_num: The number to sample for each prompt, when prompt is [a, b, c] and `sampling_num=2`, the sampled result will be [a_1, a_2, b_1, b_2, c_1, c_2]
        :param LongCoT_stop_sign: The stop sign for the CoT portion
        :param output_begin_sign: The sign to mark the beginning of the output
        :return:
        """

        LongCoT_outputs = self._batched_query_model(
            prompts=prompts,
            max_tokens=int(max_tokens * 0.8),
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            sampling_num=sampling_num,
            stop_strs=[LongCoT_stop_sign]
        )

        postCoT_prompt_ls = []
        for curr in LongCoT_outputs:
            curr_input_str = PromptWriter.formulate_postLongCoT_prompt(
                curr + f"\n{LongCoT_stop_sign}",
                NL_statement=NL_statement,
                FL_statement=FL_statement,
                output_begin_sign=output_begin_sign,
                pre_Lean4_codeblock_content="",
                pre_statement_content=Lean4_HEADER
            )
            postCoT_prompt_ls += [curr_input_str]

        postCoT_outputs = self._batched_query_model(
            prompts=postCoT_prompt_ls,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            sampling_num=1
        )
        return postCoT_outputs

    def generate_proof_singleThm(
            self,
            thm_record: Dict,
            proof_num=8,
            temperature=1.0,
            top_p=0.9,
            max_tokens=4096,
            batch_size=4,
            repetition_penalty=1.0,
            print_result=True,
            LongCoT_control=False,
            return_LongCoT_content=True,
            LongCoT_begin_sign="<think>",
            LongCoT_stop_sign="</think>",
            output_begin_sign="",
            llm_type="LoT-Solver",
            system_prompt="You are an expert in mathematics and Lean 4.",

    ) -> Tuple[List[str], List[str], List[str]]:
        """
        The function for generating proof for a single theorem record and return the generated proof and Long CoT content, this is where model-specific settings take place
        :param thm_record: The theorem record in dict format to be proved. The theorem record should be the following format:
            {
                "NL": <Natural Language version of theorem statement and prove>,
                "Name": <Name of current theorem>,
                "FL_statement": <Formal Language version of theorem statement>, ...
            }
        :param proof_num: The number of proofs to be generated for each theorem record
        :param temperature: temperature for sampling, larger means the result will be more random
        :param top_p: top_p for sampling. It means only sampling from the top p portion tokens.
        :param max_tokens: The max tokens for the whole sequence (prompt + generated text)
        :param batch_size: The batch size for the model, we will query the model with `batch_size` number of prompts
        :param repetition_penalty: The penalty for repetition during sampling, larger means the model will prefer not to sampling tokens pre-existed in the sentence
        :param print_result: Whether to print the result
        :param LongCoT_control: Whether to use LongCoT control
        :param return_LongCoT_content: Whether to return the LongCoT content
        :param LongCoT_begin_sign: The sign to mark the beginning of the LongCoT content
        :param LongCoT_stop_sign: The sign to mark the end of the LongCoT content
        :param output_begin_sign: The sign mark the beginning of output after Long CoT
        :param llm_type: The type of LLM to be used, default is LoT-Solver
        :param system_prompt: The system prompt to be used for the model
        :return
            1. A list of completed Lean4 theorem proofs in the format:
                ['theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  -- aesop?\n  ring',
                'theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  rfl', ...]
            2. A list of LongCoT content in the format:
                [<long_cot_1>, <long_cot_2>, ...]
            3. A list of total generated content in the format:
                [<total_content_1>, <total_content_2>, ...]
        """
        proof_ls = []
        LongCoT_contents = []
        response_ls = []

        for i in range(int(proof_num/batch_size)):
            if "lot-solver" in llm_type.lower():
                prompt = PromptWriter.formulate_prompt_LoTSolver(
                    thm_record["NL"],
                    thm_record["Name"],
                    thm_record["FL_statement"],
                    system_prompt=system_prompt
                )
            elif "kimina" in llm_type.lower():
                prompt = PromptWriter.formulate_prompt_Kimina_Prover(
                    thm_record["NL"],
                    thm_record["FL_statement"],
                    system_prompt=system_prompt
                )
            else:
                raise NotImplementedError

            if print_result:
                print("current prompt is:")
                print(prompt)

            if not LongCoT_control:
                curr_responses = self._batched_query_model(
                    [prompt],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    sampling_num=batch_size
                )
            else:
                curr_responses = self._batched_query_model_LongCoT_control(
                    [prompt],
                    thm_record["NL"],
                    thm_record["FL_statement"],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    sampling_num=batch_size,
                    LongCoT_stop_sign=LongCoT_stop_sign,
                    output_begin_sign=output_begin_sign
                )

            response_ls += curr_responses

            if return_LongCoT_content:
                for curr_response in curr_responses:
                    matches = re.findall(rf"{re.escape(LongCoT_begin_sign)}(.*?){re.escape(LongCoT_stop_sign)}",
                                         curr_response, re.DOTALL)
                    if matches:
                        LongCoT_contents += [matches[-1]]
                    else:
                        LongCoT_contents += [""]
            else:
                LongCoT_contents = [""] * len(curr_responses)

            if print_result:
                for curr_response in curr_responses:
                    print(f"{'#' * 20}\ncurrent response is:\n{curr_response}")

            for curr_response in curr_responses:
                if utils.contains_code_block(curr_response, "lean4"):
                    curr_proof = utils.extract_code_blocks_as_list(curr_response, "lean4")[-1]
                    proof_ls += [curr_proof]
                else:
                    proof_ls += ["sorry"]
        return proof_ls, LongCoT_contents, response_ls


    def generate_proof_dataset(
            self,
            set_to_prove: List[Dict],
            proof_num=8,
            max_tokens=4096,
            temperature=0.9,
            top_p=0.9,
            batch_size=4,
            repetition_penalty=1.0,
            print_result=True,
            LongCoT_control=False,
            return_LongCoT_content=True,
            LongCoT_begin_sign="<think>",
            LongCoT_stop_sign="</think>",
            output_begin_sign="",
            ckpt_path="./Generated_proof_ckpt",
            llm_type="LoT-Solver",
            system_prompt="You are an expert in mathematics and Lean 4.",
    ) -> List[Dict]:
        """
        The function for generating proof for the whole dataset and return the updated dataset with the generated proof and Long CoT content
        :param set_to_prove: The list of theorem record in dict format to be proved. The theorem record should be the following format:
            [{
                    "NL": <Natural Language version of theorem statement and prove>,
                    "Name": <Name of current theorem>,
                    "FL_statement": <Formal Language version of theorem statement>, ...
            }, ...]
        :param proof_num: The number of proofs to be generated for each theorem record
        :param max_tokens: The max tokens for the whole sequence (prompt + generated text)
        :param temperature: temperature for sampling, larger means the result will be more random
        :param top_p: top_p for sampling. It means only sampling from the top p portion tokens.
        :param batch_size: The batch size for the model, we will query the model with `batch_size` number of prompts
        :param repetition_penalty: The penalty for repetition during sampling, larger means the model will prefer not to sampling tokens pre-existed in the sentence
        :param print_result: Whether to print the result
        :param LongCoT_control: Whether to use LongCoT control
        :param return_LongCoT_content: Whether to return the LongCoT content
        :param LongCoT_begin_sign: The sign to mark the beginning of the LongCoT content
        :param LongCoT_stop_sign: The sign to mark the end of the LongCoT content
        :param output_begin_sign: The sign mark the begin of output after Long CoT content
        :param ckpt_path: The path to save the generated proof, the filename of ckpt will be the hash of the total data record
        :param llm_type: The type of LLM to be used, default is LoT-Solver
        :param system_prompt: The system prompt to be used for the model
        :return: The updated list of dict dataset with the generated proof, each data record should have the following format:
                [{
                    "NL": <Natural Language version of theorem statement and prove>,
                    "Name": <Name of current theorem>,
                    "FL_statement": <Formal Language version of theorem statement>,
                    "Generation_results":
                        [{
                            "Generated_proof": <generated_proof_1>,
                            "Long_CoT_content": <long_cot_1>
                            "Generated_content": <total_content_1>
                        }, {
                            "Generated_proof": <generated_proof_2>,
                            "Long_CoT_content": <long_cot_2>
                            "Generated_content": <total_content_2>
                        }, ...]
                }, ...]
        """
        for i in tqdm(range(len(set_to_prove))):
            curr_thm_record = set_to_prove[i]
            curr_prove_ls, LongCoT_contents, response_ls = self.generate_proof_singleThm(
                curr_thm_record,
                proof_num=proof_num,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                batch_size=batch_size,
                repetition_penalty=repetition_penalty,
                print_result=print_result,
                LongCoT_control=LongCoT_control,
                return_LongCoT_content=return_LongCoT_content,
                LongCoT_begin_sign=LongCoT_begin_sign,
                LongCoT_stop_sign=LongCoT_stop_sign,
                output_begin_sign=output_begin_sign,
                llm_type=llm_type,
                system_prompt=system_prompt
            )

            assert len(curr_prove_ls) == len(LongCoT_contents) and len(curr_prove_ls) == len(response_ls), \
                (f"The length of curr_prove_ls (len: {len(curr_prove_ls)}), LongCoT_contents (len: {len(LongCoT_contents)}), and"
                 f"response_ls (len: {len(response_ls)}) should be the same")
            curr_thm_record["Generation_results"] = []
            for j in range(len(curr_prove_ls)):
                curr_thm_record["Generation_results"] += [{
                    "Generated_proof": curr_prove_ls[j],
                    "Long_CoT_content": LongCoT_contents[j],
                    "Generated_content": LongCoT_contents[j]
                }]
            set_to_prove[i] = curr_thm_record

            if ckpt_path != None:
                curr_ckpt_filename = utils.hash_dict(curr_thm_record) + ".json"
                utils.write_to_json(f"{ckpt_path}/{curr_ckpt_filename}", curr_thm_record)


