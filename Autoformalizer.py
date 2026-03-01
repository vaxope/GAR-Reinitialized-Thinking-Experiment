from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
import torch
import re
import random
from tqdm.auto import tqdm
from utils import utils, PromptWriter
from utils.PromptWriter import Lean4_HEADER
from typing import List, Dict, Optional, Union, Tuple
from vllm import LLM, SamplingParams
from LLM_inference import LLM_rollout
import os
import argparse


class Autoformalizer:
    """
    This is the helper class for writing autoformalization of existence problem
    """
    def __init__(
            self,
            model_id: str
    ):
        self.model_id = model_id
        self.model = LLM(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.generator = LLM_rollout(self.model, self.tokenizer)

    def _autoformalization(
            self, 
            data_record_ls_to_autoformalize: List[Dict], 
            max_tokens=8192, 
            temperature=0.6, 
            top_p=0.95,
            batch_size=4,
            repetition_penalty=1.0,
            print_result=True, 
            LLM_type="Kimina-autoformalizer"
    ):
        """
        This function autoformalizes the problems in natural language to Lean4 code.
        :param data_record_ls_to_autoformalize: List of dictionaries containing the problems to autoformalize. The dict should have the following keys:
            - "Name": The name of the theorem
            - "Natural_language_statement": The natural language statement that needs to be autoformalized
            - "Generation_log": A dict that contains the generation log, which is previously updated in statement fusion process. 
        :param max_tokens: The maximum number of tokens to be generated, it depends on the GPU memory and batch size.
        :param temperature:  The temperature for sampling. Since this module is only for autoformalization and is not the main point of our project. It should be set to a lower value.
        :param top_p: The top_p for sampling. It should be set to a lower value.
        :param batch_size: The batch size for sampling, it should be set according to the GPU memory.
        :param repetition_penalty: The repetition penalty for sampling, it should be 1.0 unless you want to penalize the repetition of tokens.
        :param print_result: Whether to print the result.
        :param LLM_type: The type of the LLM, it is not used in current version but can be used for future extensions.

        :return generated_autoformalization_content: List of strings containing the autoformalized Lean4 code.
        :return response_ls: List of strings containing the responses from the LLM.
        :return generated_token_num_ls: List of integers containing the number of tokens generated for each response.
        Note: If we do not detect the theorem code in the response, we will return -1 for that entry in generated_autoformalization_content and other as regular in the list.
        """

        generated_autoformalization_content = []
        response_ls = []
        generated_token_num_ls = []

        batched_data_record_to_autoformalize = utils.split_batch(data_record_ls_to_autoformalize, batch_size)

        for curr_batch in batched_data_record_to_autoformalize:
            prompt_ls = []
            for curr_data_record in curr_batch:
                curr_NL_statement = curr_data_record["Natural_language_statement"]
                curr_name = "thm_GAR"
                if "kimina" in LLM_type.lower():
                    curr_prompt = PromptWriter.formulate_prompt_autoformalization_Kimina(
                        name=curr_name, 
                        problem_to_autoformalize=curr_NL_statement, 
                        tokenizer=self.tokenizer
                    )

                elif "goedel" in self.model_id.lower():
                    curr_NL_statement = curr_data_record["Natural_language_statement"]
                    curr_name = "thm_GAR"
                    curr_prompt = PromptWriter.formulate_prompt_autoformalization_Goedel(
                        name=curr_name, 
                        problem_to_autoformalize=curr_NL_statement, 
                        tokenizer=self.tokenizer
                    )
                else:
                    raise NotImplementedError(f"LLM type {LLM_type} is not supported yet for autoformalization.")
                prompt_ls.append(curr_prompt)
                if print_result:
                    print("current prompt is:")
                    print(curr_prompt)
            curr_responses, generated_token_num = self.generator.batched_query_model(
                prompt_ls, 
                max_tokens=max_tokens, 
                temperature=temperature, 
                top_p=top_p, 
                repetition_penalty=repetition_penalty, 
                sampling_num=1
            )

            for i, curr_response in enumerate(curr_responses):
                curr_generated_content = curr_response[len(prompt_ls[i]):]

                if "kimina" in self.model_id.lower():
                    match = re.search(r'theorem\b[\s\S]*?:=\s*by\b[\s\S]*', curr_generated_content)
                    if match:
                        theorem_code = match.group(0)
                        generated_autoformalization_content.append(theorem_code)
                        response_ls.append(curr_response)
                        generated_token_num_ls.append(generated_token_num[i])
                    else:
                        generated_autoformalization_content.append(-1)
                        response_ls.append(curr_response)
                        generated_token_num_ls.append(generated_token_num[i])
                else:
                    curr_generated_code_block = utils.extract_code_blocks_as_list(curr_generated_content, code_type="lean4")
                    if len(curr_generated_code_block) == 0:
                        generated_autoformalization_content.append(-1)
                        response_ls.append(curr_response)
                        generated_token_num_ls.append(generated_token_num[i])
                        if print_result:
                            print(f"{'#' * 20}\ncurrent response is:\n{curr_response}")
                        continue
                    curr_generated_code_block = curr_generated_code_block[-1]
                    match = re.search(r'theorem\b[\s\S]*?:=\s*by\b[\s\S]*', curr_generated_code_block)
                    if match:
                        theorem_code = match.group(0)
                        generated_autoformalization_content.append(theorem_code)
                        response_ls.append(curr_response)
                        generated_token_num_ls.append(generated_token_num[i])
                    else:
                        generated_autoformalization_content.append(-1)
                        response_ls.append(curr_response)
                        generated_token_num_ls.append(generated_token_num[i])
                if print_result:
                    print(f"{'#' * 20}\ncurrent response is:\n{curr_response}")
        return generated_autoformalization_content, response_ls, generated_token_num_ls

    def autoformalize_dataset(
            self, 
            dataset_to_autoformalize: List[Dict], 
            max_tokens: int = 8192,
            temperature: float = 0.6,
            top_p: float = 0.95,
            batch_size: int = 4, 
            repetition_penalty: float = 1.0,
            print_result: bool = True,
            ckpt_path: str = "./autoformalization_ckpt", 
            LLM_type="Kimina-autoformalizer"
    ):
        """
        This function autoformalizes the dataset in natural language to Lean4 code.
        :param dataset_to_autoformalize: List of dictionaries containing the problems to autoformalize. The dict should have the following keys:
            - "Name": The name of the theorem
            - "Natural_language_statement": The natural language statement that needs to be autoformalized
            - "Generation_log": A dict that contains the generation log, which is previously updated in statement fusion process.
        :param max_tokens: The maximum number of tokens to generate for each response.
        :param temperature: The temperature to use for sampling.
        :param top_p: The top-p sampling parameter.
        :param batch_size: The batch size to use for processing.
        :param repetition_penalty: The repetition penalty to use for sampling.
        :param print_result: Whether to print the results.
        :param ckpt_path: The path to the checkpoint directory.
        :param LLM_type: The type of LLM to use for generation.
        :return: A list of dict, and each dict contains the following keys:
            - "Name": The name of the fused statement.
            - "Statement": The fused problem in Lean4 format.
            - "Natural_language_statement": The natural language version of the fused statement.
            - "Generation_log": The generated contents, list of dict contains the following:
                - "Statement_fusion_log": The log of the statement fusion process, it is a dict contains the following:
                    - "Base_statements": The base statements used for fusion.
                    - "Generated_fused_statement": The generated fused statement.
                    - "Response": The response from the LLM.
                    - "Generated_token_num": The number of tokens generated for this fused statement.
                - "Autoformalization_log": The log of the autoformalization process, it is a dict contains the following:
                    - "Response": The response from the LLM.
                    - "Generated_token_num": The number of tokens generated for this autoformalized statement.
        """
        batched_dataset_to_autoformalize = utils.split_batch(dataset_to_autoformalize, batch_size)
        all_generated_data_record_ls = []

        for i in tqdm(range(len(batched_dataset_to_autoformalize))):
            curr_batch = batched_dataset_to_autoformalize[i]
            curr_autoformalized_statements, curr_response_ls, curr_generated_token_num_ls = self._autoformalization(
                data_record_ls_to_autoformalize=curr_batch, 
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size,
                repetition_penalty=repetition_penalty,
                print_result=print_result,
                LLM_type=LLM_type
            )

            assert len(curr_autoformalized_statements) == len(curr_batch) and len(curr_autoformalized_statements) == len(curr_response_ls) and len(curr_autoformalized_statements) == len(curr_generated_token_num_ls), \
                "The length of autoformalized statements, responses and generated token numbers should be the same."

            for j, curr_data_record in enumerate(curr_batch):
                if curr_autoformalized_statements[j] == -1:
                    continue
                curr_data_record = {
                    "Name": curr_data_record["Name"], 
                    "Statement": curr_autoformalized_statements[j],
                    "Natural_language_statement": curr_data_record["Natural_language_statement"],
                    "Generation_log": {
                        "Statement_fusion_log": curr_data_record["Generation_log"]["Statement_fusion_log"],
                        "Autoformalization_log": {
                            "Response": curr_response_ls[j],
                            "Generated_token_num": curr_generated_token_num_ls[j]
                        }
                    }
                }
                all_generated_data_record_ls.append(curr_data_record)
                utils.write_to_json(f"./{ckpt_path}/{curr_data_record['Name']}_autoformalized.json", curr_data_record)
        print(f"Total {len(all_generated_data_record_ls)} autoformalized statements generated and ckpt saved.")
        return all_generated_data_record_ls
        
def main():
    parser = argparse.ArgumentParser(description="Main program for autoformalization the Lean4 statements")
    parser.add_argument("--dataset_path", type=str, help="The path to json file which store the NL dataset that has finished fusion")
    parser.add_argument("--save_path", type=str, default="./autoformalized_results.json", help="The json file's path to save the autoformalized results.")
    parser.add_argument("--begin_idx", type=int, default=0, help="The begin index of the dataset to be autoformalized in the provided dataset, default is 0")
    parser.add_argument("--end_idx", type=int, default=-1, help="The end index of the dataset to be autoformalized in the provided dataset, default is -1, which means the end of the dataset")
    parser.add_argument("--cuda_device_id", type=int, default=0, help="The cuda device id used for autoformalization")
    parser.add_argument("--model_id", type=str, default="AI-MO/Kimina-Autoformalizer-7B", help="The model id for the autoformalization, it can be a local path or a HuggingFace model id")
    parser.add_argument("--ckpt_path", type=str, default="./Autoformalization_ckpt", help="Path to the checkpoint folder")
    parser.add_argument("--max_tokens", type=int, default=8192, help="The maximum number of tokens for the autoformalization, default is 8192")
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size for the autoformalization, it depend on the power of GPU.")
    parser.add_argument("--temperature", type=float, default=0.6, help="The temperature for LLM sampling, default is 0.6")
    parser.add_argument("--top_p", type=float, default=0.95, help="The top_p for LLM sampling, default is 0.95")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="The repetition penalty for LLM sampling, default is 1.0")

    args = parser.parse_args()

    utils.check_folder_exit(args.ckpt_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device_id)

    dataset_to_autoformalize = utils.read_from_json(args.dataset_path)
    if args.end_idx == -1:
        args.end_idx = len(dataset_to_autoformalize)
    dataset_to_autoformalize = dataset_to_autoformalize[args.begin_idx:args.end_idx]
    autoformalizer = Autoformalizer(
        model_id=args.model_id
    )
    formalized_statements = autoformalizer.autoformalize_dataset(
        dataset_to_autoformalize=dataset_to_autoformalize, 
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        ckpt_path=args.ckpt_path
    )
    utils.write_to_json(args.save_path, formalized_statements)

if __name__ == "__main__":
    main()


# # testing main
# if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     model_id = "../data/models/Kimina-Autoformalizer-7B"
#     utils.check_folder_exit("./test_autoformalization_ckpt")
#     dataset_to_autoformalize = utils.read_json_in_folder("./test_fused_ckpt")
#     autoformalizer = Autoformalizer(model_id=model_id)
#     formalized_statements = autoformalizer.autoformalize_dataset(
#         dataset_to_autoformalize=dataset_to_autoformalize, 
#         ckpt_path="./test_autoformalization_ckpt",
#         max_tokens=8192,
#         batch_size=4
#     )
#     utils.write_to_json("./test_autoformalization.json", formalized_statements)

