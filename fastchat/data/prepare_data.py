import json
import argparse
import random
import os
import re
# ------------------------
# Reader functions for each of the supported datasets:
# Usage: python fastchat/data/prepare_data.py --data-path ../data/raw/databricks-dolly-15k/databricks-dolly-15k.jsonl --output-path ../data/processed/vicuna-fastchat/train/databricks-dolly-15k-en.json --lang en
# 

roles = ["human", "gpt"]

def check_data(data: list, log_path: str, mode: str) -> list:
    '''
    Iterate over prepared data and check that the following conditions are true:
        - used roles are "gpt" and "human".
        - roles are alternated.
        - conversations start with "human" role.
    If errors are found these are logged in an .err file.

    ### Arguments
    - data: list
        data to check
    - log_path: str
        path where errors will be logged (if found)
    - mode: str
        Mode to use when handling found errors. Options:
        - err: an error is raised.
        - drop: the erroneous examples are droped from the data.
        - warn: only a warning is printed and execution continues.
    '''
    modes = ["err", "drop", "warn"]
    assert mode in modes, f"mode must be one of {modes}"

    

    # lists to save erroneous examples indexes:
    err_other_role_idxs = []
    err_human_starts_idxs = []
    err_not_alternating_idxs = []
    err_empty_conversation = []

    for i, example in enumerate(data):
        if len(example["conversations"]) == 0:
            err_empty_conversation.append(i)
        else:
            for j, message in enumerate(example["conversations"]): # check alternating turns and that user starts conversation
                role = message["from"]
                if not role in roles:
                    err_other_role_idxs.append(i)
                    break
                elif roles[j % 2] != role:
                    if j == 0:
                        err_human_starts_idxs.append(i)
                    else:
                        err_not_alternating_idxs.append(i)
                    break
    total_errors = len(err_other_role_idxs) + len(err_human_starts_idxs) + len(err_not_alternating_idxs) + len(err_empty_conversation)
    if total_errors != 0:
        with open(log_path, 'w') as log:
            log.write(f"TOTAL ERRORS: {total_errors} (handling mode: {mode})\n")
            if len(err_other_role_idxs) > 0:
                log.write("==================\n")
                log.write(f"OTHER ROLE ERRORS: {len(err_other_role_idxs)}\n")
                for idx in err_other_role_idxs:
                    log.write("------------------\n")
                    log.write(f"Erroneous example (index: {idx}):\n")
                    log.write(str(data[idx]) + '\n')
            if len(err_human_starts_idxs) > 0:
                log.write("==================\n")
                log.write(f"HUMAN STARTS ERRORS: {len(err_human_starts_idxs)}\n")
                for idx in err_human_starts_idxs:
                    log.write("------------------\n")
                    log.write(f"Erroneous example (index: {idx}):\n")
                    log.write(str(data[idx]) + '\n')
            if len(err_not_alternating_idxs) > 0:
                log.write("==================\n")
                log.write(f"NOT ALTERNATING ERRORS: {len(err_not_alternating_idxs)}\n")
                for idx in err_not_alternating_idxs:
                    log.write("------------------\n")
                    log.write(f"Erroneous example (index: {idx}):\n")
                    log.write(str(data[idx]) + '\n')
            if len(err_empty_conversation) > 0:
                log.write("==================\n")
                log.write(f"EMPTY CONVERSATION ERRORS: {len(err_empty_conversation)}\n")
                for idx in err_empty_conversation:
                    log.write("------------------\n")
                    log.write(f"Erroneous example (index: {idx}):\n")
                    log.write(str(data[idx]) + '\n')
        if mode == "err":
            raise Exception(f"\n>> ERROR: Dataset NOT saved due to {total_errors} errors. Errors detailed in {log_path}\n>> ERROR: Modify source data or change check_mode to 'drop' or 'warn'")
        elif mode == "drop":
            print(f">> WARNING: Dataset contains {total_errors} errors. Errors detailed in {log_path}")
            print(f">> WARNING: Dropping {total_errors} erroneous samples...")
            err_idxs = err_other_role_idxs + err_human_starts_idxs + err_not_alternating_idxs + err_empty_conversation
            err_idxs = list(dict.fromkeys(err_idxs))
            for idx in sorted(err_idxs, reverse=True):
                del data[idx]
        elif mode == "warn":
            print(f">> WARNING: Dataset contains {total_errors} errors. Errors detailed in {log_path}")
            print(f">> WARNING: Continuing with normal execution")
    else:
        print("No errors found. No log file created.")

    return data



def read_json(data_path: str) -> tuple[list, dict]:
    print("Reading Dolly-type dataset...")
    try:
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
    except:
        with open(data_path, 'r') as f:
            data = json.loads(f.read())

    return data


def prepare_basic(args):
    template = args.humman.encode().decode('unicode_escape')
    data = read_json(args.data_path) 
    print("Preparing and adapting data fields...")
    prep_data = []
    for i, example in enumerate(data):
        prep_example = example.copy()
        placeholders = re.findall(r'{(.*?)}', template)

        # Format the string with values from the JSON dict
        humman_text = template.format(**{key: prep_example.pop(key) for key in placeholders})

        prep_example["conversations"] = [
        {
            "from": roles[0],
            # "value": (prep_example.pop(relevant_fields["instruction_field"]), prep_example.pop(relevant_fields["input_field"]))
            "value": humman_text
        },
        {
            "from": roles[1],
            "value": prep_example.pop(args.assistent)
        }]
        
        # setting language field
        if args.lang == "mm": # multilingual dataset
            if args.lang_field in prep_example:
                prep_example["lang"] = prep_example[args.lang_field]
        else: # monolingual dataset
            prep_example["lang"] = args.lang

        prep_data.append(prep_example)
            

    print("Checking dataset...")
    err_path = os.path.splitext(args.output_path)[0]+'.err'
    prep_data = check_data(data=prep_data, log_path=err_path, mode=args.check_mode)
    
    print("Saving prepared dataset...")
    with open(args.output_path, 'w') as out_file:
        out_file.write(json.dumps(prep_data, indent=2, ensure_ascii=False))
    print(f"Prepared dataset saved in {args.output_path}")

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, required=True, help="Source data path (can be a directory for Aya dataset type)")
    parser.add_argument("--output-path", type=str, required=True, help="Path where the output file will be saved, containing the desired file name.")
    parser.add_argument("--lang", type=str, required=True, help="ISO language code of the language of the dataset (set to 'mm' for multilingual datasets)")
    parser.add_argument("--lang-field", type=str, required=False, default="lang", help="Lang field from source data. Default: 'lang'")
    parser.add_argument("--humman", type=str, required=False, default="'{istruction}'",help="Humman field that can combine multiple fields. Default: '{istruction}'. E.g, '{prompt}\\n\\nContext:\\n{context}\\n\\nQuestion:\\n{instruction}'")
    parser.add_argument("--assistent", type=str, required=False, default="response",help="Lang field from source data. Default: 'response'")
    parser.add_argument("--check-mode", type=str, default="err", required=False, help="Mode used when checking prepared data. Options: 'err', 'drop', 'warn'")
    
    args = parser.parse_args()
    prepare_basic(args)