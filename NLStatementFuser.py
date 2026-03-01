from transformers import AutoTokenizer
from utils import utils
from utils import PromptWriter
from vllm import LLM, SamplingParams
from LLM_inference import LLM_rollout
from typing import List, Dict
import random
from tqdm.auto import tqdm
import os
import argparse

class NLStatementFuser:
    def __init__(
        self, 
        model_id: str,
        base_statements_dataset: List[Dict]
    ):
        """
        Initializes the StatementFuser with a language model and a dataset of base statements.
        :param model_id: The name/path of the LLM to rollout.
        :param base_statements_dataset: A list of dict containing base statements to be fused, it should contain at least the following keys:
            - "Name": The name of the base statement.
            - "Natural_language_statement": The natural language version of the base statement.
        """
        self.model = LLM(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.generator = LLM_rollout(self.model, self.tokenizer)
        self.base_statements_dataset = base_statements_dataset
        self.have_problem_type = False
        if "Problem_type" in base_statements_dataset[0]:
            print("The dataset has problem type information.")
            self.have_problem_type = True
            self.problems_by_type = {}
            for item in base_statements_dataset:
                if "Problem_type" in item:
                    p_type = item["Problem_type"]
                    if p_type not in self.problems_by_type:
                        self.problems_by_type[p_type] = []
                    self.problems_by_type[p_type].append(item)

    def _fuse_statement(
        self, 
        base_statement_ls: List[List[str]],
        skip_thinking: bool = False,
        max_tokens: int = 8192,
        temperature: float = 0.8,
        top_p: float = 0.95,
        batch_size: int = 4, 
        repetition_penalty=1.0,
        print_result: bool = True, 
        LLM_type: str = "Qwen3"
    ):
        """
        The function that fuses multiple base statements into a single and harder problem using LLM.
        :param base_statement_ls: A batch contains of lists of base statements to be fused, each base list should contain at least two statements.
        :param skip_thinking: Whether to skip the thinking process in the prompt.
        :param max_tokens: The maximum number of tokens to be generated, it depends on the GPU memory and batch size.
        :param temperature: The temperature for sampling, in the training case, it should be set higher.
        :param top_p: The top-p value for sampling.
        :param batch_size: The batch size for sampling, it should be set according to the GPU memory.
        :param repetition_penalty: The repetition penalty for sampling, it should be 1.0 unless you want to penalize the repetition of tokens.
        :param print_result: Whether to print the result.
        :param LLM_type: The type of the LLM, it is not used in current version but can be used for future extensions.
        """

        generated_fused_statements = []
        response_ls = []
        generated_token_num_ls = []
        base_statement_used = []

        batched_base_statements = utils.split_batch(base_statement_ls, batch_size)
        
        # TEST Code
        print(f"curr experiment type is: {os.environ.get('EXP_TYPE')}")
        exp_type = os.environ.get('EXP_TYPE', None)

        for curr_batch in batched_base_statements:
            prompt_ls = []
            for curr_base_statement_ls in curr_batch:
                if exp_type == 'MagicCoder_test':
                    curr_prompt = PromptWriter.formulate_prompt_statement_gen_MagicCoder(
                        base_statement_ls=curr_base_statement_ls, 
                        tokenizer=self.tokenizer,
                        skip_thinking=skip_thinking
                    )
                else:
                    curr_prompt = PromptWriter.formulate_prompt_fuse_statements(
                        base_statement_ls=curr_base_statement_ls, 
                        tokenizer=self.tokenizer,
                        skip_thinking=skip_thinking
                    )
                prompt_ls.append(curr_prompt)
                if print_result:
                    print("#" * 20)
                    print("current prompt is:")
                    print(curr_prompt)
            curr_responses, generated_token_num = self.generator.batched_query_model(
                prompts=prompt_ls, 
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                sampling_num=1
            )

            for i, curr_response in enumerate(prompt_ls):
                curr_generated_content = curr_responses[i][len(prompt_ls[i]):]
                print(f"current generated content is:\n{curr_generated_content}")
                if utils.contains_code_block(curr_generated_content, code_type="md"):
                    problem_fused = utils.extract_code_blocks_as_list(curr_generated_content, code_type="md")[-1]
                elif utils.contains_code_block(curr_generated_content, code_type="markdown"):
                    problem_fused = utils.extract_code_blocks_as_list(curr_generated_content, code_type="markdown")[-1]
                elif utils.contains_code_block(curr_generated_content, code_type="Markdown"):
                    problem_fused = utils.extract_code_blocks_as_list(curr_generated_content, code_type="Markdown")[-1]
                elif  "<problem>" in curr_generated_content and "</problem>" in curr_generated_content:
                    problem_fused = curr_generated_content.split("<problem>")[-1].split("</problem>")[0].strip()
                else:
                    # If no code block is found, we assume the response is not valid
                    print("No code block found in the response, please check the response.")
                    continue
                generated_fused_statements.append(problem_fused)
                response_ls.append(curr_generated_content)
                generated_token_num_ls.append(generated_token_num[i])
                base_statement_used.append(curr_batch[i])
                if print_result:
                    print(f"{'#' * 20}\ncurrent response is:\n{curr_response}")
        return generated_fused_statements, response_ls, generated_token_num_ls, base_statement_used
    
    def _sampling_base_statement_ls(
        self, 
        generated_statement_num: int = 1000, 
        num_base_statements_per_fuse: int = 2
    ):
        """
        This function samples base statements from the dataset to be fused
        :return: A list of lists, each inner list contains the base statements to be fused.
        """
        base_statement_ls = []
        if self.have_problem_type:
            for _ in range(generated_statement_num):
                import random, time, os
                random.seed(time.time_ns() + os.getpid())
                base_sampled_statement = random.sample(self.base_statements_dataset, 1)[0]
                p_type = base_sampled_statement["Problem_type"]
                candidate_statements = self.problems_by_type[p_type]
                sampled_base_statements = random.sample(candidate_statements, num_base_statements_per_fuse - 1)
                sampled_base_statements.append(base_sampled_statement)
                base_statement_ls.append([statement["Natural_language_statement"] for statement in sampled_base_statements])
        else:
            for _ in range(generated_statement_num):
                import random, time, os
                random.seed(time.time_ns() + os.getpid())
                sampled_base_statements = random.sample(self.base_statements_dataset, num_base_statements_per_fuse)
                base_statement_ls.append([statement["Natural_language_statement"] for statement in sampled_base_statements])
        return base_statement_ls


    def fuse_statements_dataset(
        self, 
        generated_statement_num: int = 1000,
        num_base_statements_per_fuse: int = 2,
        skip_thinking: bool = False,
        max_tokens: int = 8192,
        temperature: float = 0.8,
        top_p: float = 0.95,
        batch_size: int = 16,
        micro_batch_size: int = 4, 
        repetition_penalty=1.0,
        print_result: bool = True, 
        ckpt_path: str = "./fused_statements_ckpt", 
        LLM_type: str = "Qwen3"
    ):
        """
        This function fuses multiple base statements from the dataset into a single and harder problem using LLM.
        :param generated_statement_num: The number of statements to be generated, it should depends on how large is each step for RL training.
        :param num_base_statements_per_fuse: The number of base statements to be fused, it should be at least 2.
        :param skip_thinking: Whether to skip the thinking process in the prompt.
        :param max_tokens: The maximum number of tokens to be generated, it depends on the GPU memory and batch size.
        :param temperature: The temperature for sampling, in the training case, it should be set higher.
        :param top_p: The top-p value for sampling.
        :param batch_size: The batch size for sampling, it should be set according to the GPU memory.
        :param repetition_penalty: The repetition penalty for sampling, it should be 1.0 unless you want to penalize the repetition of tokens.
        :param print_result: Whether to print the result.
        :param ckpt_path: The path to save the checkpoint of the fused statements.
        :param LLM_type: The type of the LLM, it is not used in current version but can be used for future extensions.
        :return: A list of dict, each dict contains the following keys:
            - "Name": The name of the fused statement.
            - "Natural_language_statement": The natural language version of the fused statement.
            - "Generation_log": The generated contents, list of dict contains the following:
                - "Statement_fusion_log": The log of the statement fusion process, it is a dict contains the following:
                    - "Base_statements": The base statements used for fusion.
                    - "Generated_fused_statement": The generated fused statement.
                    - "Response": The response from the LLM.
                    - "Generated_token_num": The number of tokens generated for this fused statement.
        """
        base_statement_ls = self._sampling_base_statement_ls(
            generated_statement_num=generated_statement_num, 
            num_base_statements_per_fuse=num_base_statements_per_fuse
        )

        batched_base_statement_ls = utils.split_batch(base_statement_ls, batch_size)
        all_generated_fused_statements = []

        for curr_batch in tqdm(batched_base_statement_ls):
            curr_fused_statements, curr_response_ls, curr_generated_token_num_ls, base_statement_used = self._fuse_statement(
                base_statement_ls=curr_batch,
                skip_thinking=skip_thinking,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                batch_size=micro_batch_size,  # Use micro_batch_size for LLM generation
                repetition_penalty=repetition_penalty,
                print_result=print_result,
                LLM_type=LLM_type
            )
            assert len(curr_fused_statements) == len(curr_response_ls) == len(curr_generated_token_num_ls), \
                "The length of generated statements, responses and generated token numbers should be the same."
            
            for i in range(len(curr_fused_statements)):
                curr_prompt = PromptWriter.formulate_prompt_fuse_statements(
                    base_statement_ls=curr_batch[i],
                    tokenizer=self.tokenizer,
                    skip_thinking=skip_thinking
                )
                curr_response = curr_prompt + curr_response_ls[i]
                curr_data_record = {
                    "Name": "NA", 
                    "Natural_language_statement": curr_fused_statements[i],
                    "Generation_log": {
                        "Statement_fusion_log": {
                            "Base_statements": base_statement_used[i],
                            "Generated_fused_statement": curr_fused_statements[i],
                            "Response": curr_response,
                            "Input": curr_prompt,
                            "Generated_token_num": curr_generated_token_num_ls[i]
                        }
                    }
                }
                curr_name = utils.hash_dict(curr_data_record)
                curr_data_record["Name"] = curr_name
                all_generated_fused_statements.append(curr_data_record)
                utils.write_to_json(f"./{ckpt_path}/{curr_name}.json", curr_data_record)
        print(f"Total {len(all_generated_fused_statements)} statements generated and saved to {ckpt_path}.")
        return all_generated_fused_statements


def main():
    parser = argparse.ArgumentParser(description="Main program for fusing NL statements")
    parser.add_argument("--base_dataset_path", type=str, help="The path to json file which store the base dataset")
    parser.add_argument("--save_path", type=str, default="./fused_results.json", help="The json file's path to save the fused results.")
    parser.add_argument("--cuda_device_id", type=int, default=0, help="The cuda device id used for fusing")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B", help="The model id for the fusing, it can be a local path or a HuggingFace model id")
    parser.add_argument("--generated_statement_num", type=int, default=20, help="The number of generated statements for each input statement, default is 20")
    parser.add_argument("--num_base_statement_per_fuse", type=int, default=2, help="The number of base statements to be fused for each generated statement, default is 2")
    parser.add_argument("--ckpt_path", type=str, default="./NLStatementFuser_ckpt", help="Path to the checkpoint folder")
    parser.add_argument("--skip_thinking", type=bool, default=False, help="Whether to use skip thinking for fusing, default is False")
    parser.add_argument("--max_tokens", type=int, default=8192, help="The maximum number of tokens for the fusing, default is 8192")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size for the fusing, it depend on the power of GPU.")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="The micro batch size for the fusing, it depend on the power of GPU.")
    parser.add_argument("--temperature", type=float, default=0.8, help="The temperature for the fusing, default is 0.8")
    parser.add_argument("--top_p", type=float, default=0.95, help="The top_p for fusing generation, default is 0.95")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="The repetition penalty for fusing generation, default is 1.0")
    args = parser.parse_args()

    utils.check_folder_exit(args.ckpt_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device_id)
    base_dataset = utils.read_from_json(args.base_dataset_path)
    fuser = NLStatementFuser(model_id=args.model_id, base_statements_dataset=base_dataset)
    fused_statements = fuser.fuse_statements_dataset(
        generated_statement_num=args.generated_statement_num, 
        num_base_statements_per_fuse=args.num_base_statement_per_fuse,
        skip_thinking=args.skip_thinking,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size, 
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        ckpt_path=args.ckpt_path,
        LLM_type=args.model_id
    )
    utils.write_to_json(args.save_path, fused_statements)

if __name__ == "__main__":
    main()