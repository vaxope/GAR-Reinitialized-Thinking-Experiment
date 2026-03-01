import json
import torch
import os
from datasets import Dataset
import re
from typing import List, Dict, Any
import hashlib

def extract_and_combine_proof(header, statement, generated_proof):
    
    statement = statement.strip()
    generated_proof = generated_proof.strip()
    
    if not re.search(r'theorem\s+\w+.*?:=\s*$', statement, re.DOTALL):
        raise RuntimeError(f"**Error**: statement statement form is incorrect\n{statement}")
    
    if generated_proof.strip() == "sorry":
        result = header + statement + " sorry"
        return result
    pattern_with_theorem = r'theorem\s+\w+.*?:=\s*(.+)'
    match_with_theorem = re.search(pattern_with_theorem, generated_proof, re.DOTALL)
    
    if match_with_theorem:
        proof_body = match_with_theorem.group(1).strip()

        if proof_body == "sorry":
            result = header + statement + " sorry"
        else:
            result = header + statement + " " + proof_body
        return result

    else:
        proof_body = generated_proof.strip()
        result = header + statement + " " + proof_body
        return result

def return_theorem_to_prove(text):
    pattern = r'((?:theorem).*?:=\s*by\s*sorry)'
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None

def remove_comments(text): # remove comments
    # First remove all /- ... -/ blocks
    text = re.sub(r'/-.*?-/', '', text, flags=re.DOTALL)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Split on -- and keep only the first part
        cleaned_line = line.split('--', 1)[0]
        cleaned_lines.append(cleaned_line)
    # Join back together and remove excessive empty lines
    cleaned_text = '\n'.join(cleaned_lines)
    return cleaned_text.strip()

def return_theorem_to_replace(text):
    # Pattern that matches from 'theorem' or 'lemma' to ':= by sorry' with any content in between
    # pattern = r'((?:theorem).*?:=\s*by)'
    pattern = r'((?:^|\s)theorem\s+.*?:=\s*by)'
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None

def replace_statement_in_proof(statement, proof):
    stats_re = remove_comments(statement)
    stats_span_= return_theorem_to_prove(stats_re)
    if stats_span_ is None:
        error_app = '\n'.join(["\n"] + ['-- ' + x for x in statement.split('\n')])
        return F"**Error**, can not find 'theorem' and ':= sorry' in {error_app}"
    proof_str = remove_comments(proof)
    span = return_theorem_to_replace(proof_str)
    if span is None:
        error_app = '\n'.join(["\n"] + ['-- ' + x for x in proof.split('\n')])
        return F"**Error**, can not find 'theorem' and ':=' in {error_app}"
    return stats_re[:stats_span_[1]].replace("sorry", "") + proof_str[span[1]:]


def judge_statement_modification(
    original_statement: str,
    generated_proof: str,
) -> bool:
    """
    This function judges whether the statement is modified in the generated proof. 
    """
    processed_original_statement = rstrip_space_newline_and_by(original_statement)
    processed_original_statement = processed_original_statement.replace(" ", "").replace("\n", "").replace("\t", "")
    processed_generated_proof = generated_proof.replace(" ", "").replace("\n", "").replace("\t", "")
    return processed_original_statement not in processed_generated_proof

def rstrip_space_newline_and_by(s: str) -> str:
    s = s.rstrip(' \n\t')
    while True:
        changed = False
        for word in ('sorry', 'by'):
            if s.endswith(word):
                before = len(s) - len(word)
                if before == 0 or not (s[before - 1].isalnum() or s[before - 1] == '_'):
                    s = s[:before].rstrip(' \n\t')
                    changed = True
                    break 
        if not changed:
            break
    return s





def split_batch(data_to_split: List[Any], batch_size: int) -> List[List[Any]]:
    """
    This function splits the data into batches of the given size.
    :param data_to_split: The data to be split
    :param batch_size: The size of each batch
    :return: A list of batches, each batch is a list of data
    """
    return [data_to_split[i:i + batch_size] for i in range(0, len(data_to_split), batch_size)]

def hash_dict(dict_to_hash: Dict) -> str:
    """
    This function generate the has for the dictionary provided.
    :param dict_to_hash: The dictionary needed to hash
    :return: sha256 result of the dict
    """
    dict_str = json.dumps(dict_to_hash, sort_keys=True)
    return (hashlib.sha256(dict_str.encode('utf-8'))).hexdigest()

def contains_code_block(text, code_type="md") -> bool:
    """
    This function test whether there is a code block inside the text
    :param text: text to check
    :param code_type: the code block type to check, default is markdown
    :return T/F of whether the code-block is found
    """
    pattern = rf'```{code_type}.*?```'
    match = re.search(pattern, text, re.DOTALL)
    return match is not None


def extract_code_blocks_as_list(text: str, code_type="md") -> List[str]:
    """
    The helper function to extract the code blocks from the text provided in LLMs and returns a list of content in code-block founded
    :param text: text to extract the code-block
    :param code_type: the code-block type to extract, default is md
    :return List of contents in the code-block founded, if not found, returns -1
    """
    lines = text.split('\n')
    inside_code_block = False
    code_blocks = []
    current_block = []

    for line in lines:
        if line.strip().startswith(f'```{code_type}'):
            inside_code_block = True
            continue 
        elif "```" in line.strip() and inside_code_block:
            code_blocks.append('\n'.join(current_block))
            current_block = []
            inside_code_block = False
            continue  

        if inside_code_block:
            current_block.append(line)

    try:
        if current_block != []:
            code_blocks.append('\n'.join(current_block))
        return code_blocks
    except Exception:
        RuntimeWarning(f"Code block of type {code_type} is not found in: {text}")
        return -1


def preprocess_theorem_statement(Lean4_statement: str):
    """
    The helper function to preprocess the theorem statement to let if follows the correct format that ends with := by
    :param Lean4_statement: the statement needed to be processed
    :return: The statement in correct format that ends with `:= by`
    """
    while not Lean4_statement.endswith(":="):
        Lean4_statement = Lean4_statement[:-1]

    if Lean4_statement.endswith(":="):
        Lean4_statement = Lean4_statement[:-len(":=")]
    Lean4_statement += ":= by"
    return Lean4_statement

def check_folder_exit(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"creating '{folder_path}'")
    else:
        print(f"folder '{folder_path}' exists")


def write_to_json(filePath, data):
    with open(filePath, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def write_to_file(filePath, txt_to_write):
    with open(filePath, 'w', encoding='utf-8') as file:
        file.write(txt_to_write)


# This function will read all json files in the folder and return a list of all json file
def read_json_in_folder(folder_path):
    json_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            curr_file = read_from_json(file_path)
            json_list += [curr_file]
    return json_list


def read_from_json(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def get_best_device(i=0):
    if torch.cuda.is_available():
        return torch.device("cuda:" + str(i))
    elif torch.backends.mps.is_built():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def load_dataset(dataset_path, begin_dix=0):
    data = read_from_json(dataset_path)
    dataset_keys = list(data[0].keys())
    dataset_dic = {key: [] for key in dataset_keys}
    for curr_record in data[begin_dix:]:
        for key in dataset_keys:
            dataset_dic[key] += [curr_record[key]]
    return Dataset.from_dict(dataset_dic)


def load_data_ls(data_path,
                 has_proof=False,
                 has_comment=False,
                 llm_type="Meta-Llama3-8B",
                 natural_language_statement_key="Informal_statement",
                 natural_language_proof_key="Informal_proof"):
    data = read_from_json(data_path)
    to_return = []
    for i in range(len(data)):
        curr_record = data[i]
        if has_proof:
            if "sorry" in data[i]["Proof"]:
                continue
            curr_record["FL"] = curr_record["Proof"]
            if has_comment:
                try:
                    curr_record["FL"] = curr_record["Commented_proof"]
                except Exception as e:
                    print(f"{i}-th data record have problem, {e}")
        if "deepseek" in llm_type.lower() and 'prover' in llm_type.lower():
            curr_record["NL"] = curr_record[natural_language_statement_key]
        else:
            curr_record["NL"] = f"{curr_record[natural_language_statement_key]}"
        curr_record["FL_statement"] = curr_record["Statement"]
        to_return += [curr_record]
    return to_return


def load_data_ls_pureLean(data_path, has_proof=False):
    data = read_from_json(data_path)
    to_return = []
    for i in range(len(data)):
        curr_record = data[i]
        if has_proof:
            if "sorry" in data[i]["Proof"]:
                continue
            curr_record["FL"] = curr_record["Proof"]
        curr_record["NL"] = ""
        curr_record["FL_statement"] = curr_record["Statement"]
        to_return += [curr_record]
    return to_return


# This function extracts the lean4 code that have the given theorem_name
def extract_theorem_proof(input_str, theorem_name):
    # Regular expression to match theorems and proofs
    pattern = re.compile(
        r'(?P<theorem>theorem\s+' + re.escape(theorem_name) + r'\s*.*?[^:])\s*:=\s*(?P<proof>by\s+.*?)(?=\n\n|\Z)',
        re.DOTALL
    )

    match = pattern.search(str(input_str))
    if match:
        return match.group('theorem') + ' := ' + match.group('proof')
    else:
        return None


# This function extracts the theorem name from the FL_statement
def find_theorem_name(input_str):
    # Regular expression to match the theorem name
    pattern = re.compile(r'theorem\s+(\w+)')

    match = pattern.search(input_str)
    if match:
        return match.group(1)
    else:
        return None
