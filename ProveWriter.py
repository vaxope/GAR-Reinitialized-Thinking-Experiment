# ProveWriter.py — Proof generation orchestrator
from transformers import AutoTokenizer
import math, os, sys, json, argparse, hashlib
from typing import List, Dict, Optional
from pathlib import Path
from tqdm.auto import tqdm
from utils import utils, PromptWriter
from vllm import LLM
from LLM_inference import LLM_rollout
import re

# ---------------------------
# Helper utilities
# ---------------------------

def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def _load_json_silent(p: str) -> Optional[Dict]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _normalize_record(rec: Dict) -> Dict:
    rec.setdefault("Generated_proof", [])
    rec.setdefault("Proof_generation_log", [])
    rec.setdefault("Proof_attempts", rec.get("Proof_attempts", 0))
    return rec

def _merge_ckpts(dataset_path: str, gbegin: int, gend: int, ckpt_path: str) -> List[Dict]:
    raw = utils.read_from_json(dataset_path)[gbegin:gend]
    merged: List[Dict] = []
    for rec in raw:
        if "Natural_language_statement" in rec:
            name = rec.get("Name") or rec.get("name") or hashlib.sha256(
                (rec.get("Statement", "") + rec.get("Natural_language_statement", "")).encode("utf-8")
            ).hexdigest()
        elif "Informal_statement" in rec:
             name = rec.get("Name") or rec.get("name") or hashlib.sha256(
                (rec.get("Statement", "") + rec.get("Informal_statement", "")).encode("utf-8")
            ).hexdigest()
        else:
            raise NotImplementedError
        f = os.path.join(ckpt_path, f"{name}.json")
        dat = _load_json_silent(f)
        dat = _normalize_record(dat if dat is not None else dict(rec))
        merged.append(dat)
    return merged

# ---------------------------
# Core ProveWriter (unchanged API)
# ---------------------------
class ProveWriter:
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.95):
        self.model = LLM(model_name, gpu_memory_utilization=gpu_memory_utilization)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = LLM_rollout(self.model, self.tokenizer)
    
    def _write_proof_single_thm(
        self, 
        Lean4_statement: str, 
        NL_statement: str,
        proof_num: int = 16, 
        temperature: float = 0.8, 
        top_p: float = 0.95,
        repetition_penalty: float = 1.0, 
        max_tokens: int = 8192,
        batch_size: int = 4, 
        llm_type: str = "Qwen3",
        system_prompt: Optional[str] = None, 
        print_result: bool = True
    ):
        """
        Generate multiple proofs for a single theorem.
        """
        generated_proofs_content, response_ls, generated_token_num_ls = [], [], []
        if ("goedel" in llm_type.lower() and "v2" in llm_type.lower()) or \
           ("deepseek" in llm_type.lower() and "v2" in llm_type.lower()) or \
           ('gar' in llm_type.lower()):
            prompt = PromptWriter.formulate_prompt_proof_writing_Goedel_V2(
                NL_statement=NL_statement, Lean4_statement=Lean4_statement, tokenizer=self.tokenizer
            )
        else:
            raise NotImplementedError(f"LLM type {llm_type} is not implemented.")

        for _ in range(math.ceil(proof_num / batch_size)):
            curr_responses, gen_tok = self.generator.batched_query_model(
                [prompt], max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, repetition_penalty=repetition_penalty, sampling_num=batch_size
            )
            if print_result:
                print("current prompt is:\n", prompt)
            for i, resp in enumerate(curr_responses):
                gen = resp[len(prompt):]
                blocks = utils.extract_code_blocks_as_list(gen, code_type="lean4")
                if print_result:
                    print(f"{'#'*20}\ncurrent response is:\n{resp}")
                if blocks == -1 or len(blocks) == 0:
                    continue
                proof = blocks[-1]
                generated_proofs_content.append(proof)
                response_ls.append(resp)
                generated_token_num_ls.append(gen_tok[i])
        return generated_proofs_content, response_ls, generated_token_num_ls
    
    def write_proof_dataset(
        self, 
        dataset_to_prove: List[Dict], 
        proof_num: int = 16, 
        temperature: float = 0.8,
        top_p: float = 0.95, 
        repetition_penalty: float = 1.0, 
        max_tokens: int = 8192,
        batch_size: int = 4, 
        llm_type: str = "Qwen3", 
        system_prompt: Optional[str] = None,
        print_result: bool = True, 
        ckpt_path: str = "./Generated_proof_ckpt", 
        in_verification: bool = False
    ):
        """
        This function will write the proof for each theorem in the dataset and save the ckpt intermediate results in ckpt_path.

        :param dataset_to_prove: List of theorem records to prove. Each record should contain at least:
            - "Name": Unique identifier for the theorem.
            - "Statement": The Lean4 statement of the theorem.
            - "Natural_language_statement" or "Informal_statement": The NL statement of the theorem.
        :param proof_num: Number of proofs to generate per theorem.
        :param temperature: Sampling temperature for the LLM, larger means more diverse.
        :param in_verification: If true, run the proof in verification mode to prevent statement modification.
        :return The dataset with proof generated for each theorem, it includes the following keys:
            - "Name": The name of the theorem or unique identifier.
            - "Statement": The original Lean4 statement of the theorem.
            - "Natural_language_statement" or "Informal_statement": The original NL statement of the theorem.
            - "Generated_proof": List of generated proof texts.
            - "Proof_generation_log": List of logs for each proof generation attempt, each log contains:
                - "generation_idx": Unique identifier for the generation attempt.
                - "generated_content": The full content generated by the LLM.
                - "generated_token_num": Number of tokens generated.
                - "generated_proof": The extracted proof text from the generated content.
                - "input": The input prompt used for generation.
        """
        for rec in dataset_to_prove:
            rec.setdefault("Generated_proof", [])
            rec.setdefault("Proof_generation_log", [])
            rec.setdefault("Proof_attempts", 0)

        if in_verification:
            print("in the verification mode!")

        def build_prompt(nl, lean):
            if ("goedel" in llm_type.lower() and "v2" in llm_type.lower()) or \
               ("deepseek" in llm_type.lower() and "v2" in llm_type.lower()) or \
               ('gar' in llm_type.lower()):
                if in_verification:
                    return PromptWriter.formulate_prompt_proof_writing_Goedel_V2_verification_mode(
                        NL_statement=nl, Lean4_statement=lean, tokenizer=self.tokenizer
                    )
                else:
                    return PromptWriter.formulate_prompt_proof_writing_Goedel_V2(
                        NL_statement=nl, Lean4_statement=lean, tokenizer=self.tokenizer
                    )
            else:
                raise NotImplementedError(f"LLM type {llm_type} is not implemented.")

        work: List[Dict] = []
        for idx, rec in enumerate(dataset_to_prove):
            need = max(0, proof_num - rec["Proof_attempts"])
            if need > 0:
                if "Natural_language_statement" in rec:
                    prompt = build_prompt(rec["Natural_language_statement"], rec["Statement"])
                elif "Informal_statement" in rec:
                    prompt = build_prompt(rec["Informal_statement"], rec["Statement"])
                else:
                    raise NotImplementedError
                work.append({"thm_idx": idx, "prompt": prompt, "need": need})

        total = sum(w["need"] for w in work)
        pbar = tqdm(total=total)
        _ensure_dir(ckpt_path)

        while True:
            work = [w for w in work if w["need"] > 0]
            if not work:
                break
            batch_prompts, owners = [], []
            for w in work:
                while w["need"] > 0 and len(batch_prompts) < batch_size:
                    batch_prompts.append(w["prompt"]) ; owners.append(w["thm_idx"]) ; w["need"] -= 1
                if len(batch_prompts) >= batch_size:
                    break
            
            curr_responses, gen_tok = self.generator.batched_query_model(
                batch_prompts, 
                max_tokens=max_tokens, 
                temperature=temperature,
                top_p=top_p, 
                repetition_penalty=repetition_penalty, 
                sampling_num=1
            )

            # This is the list of generations that LLM modified the statement during generation, 
            # we will let them write the proof again using the original statement at the end of statement.
            if in_verification:
                records_need_to_regenerate = []

            for resp, gtok, owner_idx in zip(curr_responses, gen_tok, owners):
                rec = dataset_to_prove[owner_idx]
                if "Natural_language_statement" in rec:
                    prompt = build_prompt(rec["Natural_language_statement"], rec["Statement"])
                elif "Informal_statement" in rec:
                    prompt = build_prompt(rec["Informal_statement"], rec["Statement"])
                else:
                    raise NotImplementedError
                if print_result:
                    print("current prompt is:\n", prompt)
                    print(f"{'#'*20}\ncurrent response is:\n{resp}")

                rec["Proof_attempts"] += 1
                pbar.update(1)

                gen = resp[len(prompt):] if resp.startswith(prompt) else resp
                blocks = utils.extract_code_blocks_as_list(gen, code_type="lean4")
                if blocks != -1 and len(blocks) > 0:
                    proof_text = blocks[-1]

                    if in_verification:
                        # Check if the statement was modified

                        # if not in GAR training, extract proof as the name indicates
                        if 'thm_gar' not in proof_text.lower(): 
                            proof_text = utils.extract_theorem_proof(proof_text, rec["Name"])
                        else:
                            proof_text = utils.extract_theorem_proof(proof_text, "thm_GAR")
                        cleared_statement = utils.rstrip_space_newline_and_by(rec["Statement"])

                        if proof_text is not None and cleared_statement in proof_text:
                            # ✅ 首轮合格：记录下来
                            rec["Generated_proof"].append(proof_text)
                            rec["Proof_generation_log"].append({
                                "generation_idx": hashlib.sha256(proof_text.encode("utf-8")).hexdigest(),
                                "generated_content": resp,
                                "generated_token_num": gtok,
                                "generated_proof": proof_text,
                                "input": prompt
                           })
                        else:
                            # ❌ 修改了 statement，需要再生成
                            records_need_to_regenerate.append({
                                "owner_idx": owner_idx,
                                "original_statement": rec["Statement"],
                                "Natural_language_statement": rec.get("Natural_language_statement", rec.get("Informal_statement", "")),
                                "failed_response": resp,
                                "generated_token_num": gtok
                            })
                    else:
                        rec["Generated_proof"].append(proof_text)
                        rec["Proof_generation_log"].append({
                            "generation_idx": hashlib.sha256(proof_text.encode("utf-8")).hexdigest(),
                            "generated_content": resp,
                            "generated_token_num": gtok,
                            "generated_proof": proof_text,
                            "input": prompt
                        })
                utils.write_to_json(f"{ckpt_path}/{rec['Name']}.json", rec)
            # perform regenerate if in verification mode
            if in_verification:
                regeneration_prompts = []
                regeneration_owner_idx = []
                if len(records_need_to_regenerate) > 0:
                    print(f"Regeneration required for {len(records_need_to_regenerate)} records")
                    for rec in records_need_to_regenerate:
                        curr_prompt = PromptWriter.write_regenerate_prompt(
                            failed_response=rec["failed_response"],
                            original_statement=rec["original_statement"],
                            NL_statement=rec["Natural_language_statement"],
                        )
                        regeneration_prompts.append(curr_prompt)
                        regeneration_owner_idx.append(rec["owner_idx"])
                    # perform regeneration
                    regeneration_results, regen_gen_tok = self.generator.batched_query_model(
                        regeneration_prompts, 
                        max_tokens=int(max_tokens * 1.1), 
                        temperature=temperature,
                        top_p=top_p, 
                        repetition_penalty=repetition_penalty, 
                        sampling_num=1
                    )

                    for resp, gtok, regen_rec in zip(regeneration_results, regen_gen_tok, records_need_to_regenerate):
                        owner_idx = regen_rec["owner_idx"]
                        original_rec = dataset_to_prove[owner_idx]
                        blocks = utils.extract_code_blocks_as_list(resp, code_type="lean4")
                        if blocks != -1 and len(blocks) > 0:
                            proof_text = blocks[-1]
                            # if not in GAR training, extract proof as the name indicates
                            if 'thm_gar' not in proof_text.lower():
                                proof_text = utils.extract_theorem_proof(proof_text, original_rec["Name"])
                            else:
                                proof_text = utils.extract_theorem_proof(proof_text, "thm_GAR")

                            
                            if proof_text is not None:
                                original_rec["Generated_proof"].append(proof_text)
                                original_rec["Proof_generation_log"].append({
                                    "generation_idx": hashlib.sha256(proof_text.encode("utf-8")).hexdigest(),
                                    "generated_content": resp,
                                    "generated_token_num": gtok + regen_rec["generated_token_num"],
                                    "generated_proof": proof_text,
                                    "input": prompt
                                })
                        print(f"current regeneration result is:\n", resp)

                        utils.write_to_json(f"{ckpt_path}/{original_rec['Name']}.json", original_rec)
        pbar.close()
        print(f"Processed {len(dataset_to_prove)} theorems. Ckpts in {ckpt_path}")
        return dataset_to_prove


# ---------------------------
# Main entry point
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="ProveWriter - Proof generation for theorems")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument("--cuda_device_id", type=int, default=0, help="CUDA device ID to use")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="GPU memory utilization rate")
    parser.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-Prover-V2-7B", help="Model ID or path")
    parser.add_argument("--save_path", type=str, default="./Generated_proof", help="Path to save generated proofs")
    parser.add_argument("--ckpt_path", type=str, default="./Generated_proof_ckpt", help="Path to save checkpoints")
    parser.add_argument("--begin_idx", type=int, default=0, help="Begin index for dataset slice")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index for dataset slice (-1 for all)")
    parser.add_argument("--proof_num", type=int, default=16, help="Number of proofs to generate per theorem")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum tokens to generate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--in_verification", action="store_true", help="If set, run in verification mode to prevent statement modification")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device_id)
    _ensure_dir(args.ckpt_path)
    _ensure_dir(args.save_path)

    # Load dataset
    whole_dataset = utils.read_from_json(args.dataset_path)
    whole_dataset_size = len(whole_dataset)
    
    begin_idx = args.begin_idx
    end_idx = args.end_idx if args.end_idx != -1 else whole_dataset_size

    print(f"[ProveWriter] Running proof generation...")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Range: [{begin_idx}, {end_idx})")
    print(f"  Model: {args.model_id}")
    
    dataset_slice = whole_dataset[begin_idx:end_idx]
    prover = ProveWriter(args.model_id, args.gpu_memory_utilization)
    proof_data = prover.write_proof_dataset(
        dataset_to_prove=dataset_slice,
        proof_num=args.proof_num,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        llm_type=args.model_id,
        print_result=True,
        ckpt_path=args.ckpt_path,
        in_verification=args.in_verification
    )
    
    out_file = f"{args.save_path}/{begin_idx}-{end_idx}.json"
    utils.write_to_json(out_file, proof_data)
    print(f"[ProveWriter] Done. Saved: {out_file}")


if __name__ == "__main__":
    main()
