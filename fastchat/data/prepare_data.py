import json
import argparse
import random
import os
import re
import logging


# ------------------------
# Reader functions for each of the supported datasets:
# Usage: python fastchat/data/prepare_data.py --data-path ../data/raw/databricks-dolly-15k/databricks-dolly-15k.jsonl --output-path ../data/processed/vicuna-fastchat/train/databricks-dolly-15k-en.json --lang en
# 

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

roles = ["human", "gpt"]

def check_data(data: list, mode: str) -> list:
    '''
    Iterate over prepared data and check that the following conditions are true:
        - used roles are "gpt" and "human".
        - roles are alternated.
        - conversations start with "human" role.
    If errors are found these are logged in an .err file.

    ### Arguments
    - data: list
        data to check
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
        logger.error(f"TOTAL ERRORS: {total_errors} (handling mode: {mode})")
        if len(err_other_role_idxs) > 0:
            logger.error("==================")
            logger.error(f"OTHER ROLE ERRORS: {len(err_other_role_idxs)}")
            for idx in err_other_role_idxs:
                logger.error("------------------")
                logger.error(f"Erroneous example (index: {idx}):")
                logger.error(str(data[idx]))
            
        if len(err_human_starts_idxs) > 0:
            logger.error("==================")
            logger.error(f"HUMAN STARTS ERRORS: {len(err_human_starts_idxs)}")
            for idx in err_human_starts_idxs:
                logger.error("------------------")
                logger.error(f"Erroneous example (index: {idx}):")
                logger.error(str(data[idx]))


        if len(err_not_alternating_idxs) > 0:
            logger.error("==================")
            logger.error(f"NOT ALTERNATING ERRORS: {len(err_not_alternating_idxs)}")
            for idx in err_not_alternating_idxs:
                logger.error("------------------")
                logger.error(f"Erroneous example (index: {idx}):")
                logger.error(str(data[idx]))


        if len(err_empty_conversation) > 0:
            logger.error("==================")
            logger.error(f"EMPTY CONVERSATION ERRORS: {len(err_empty_conversation)}")
            for idx in err_empty_conversation:
                logger.error("------------------")
                logger.error(f"Erroneous example (index: {idx}):")
                logger.error(str(data[idx]))

        if mode == "err":
            logger.error(f"Dataset NOT saved due to {total_errors} errors. Modify source data or change check_mode to 'drop' or 'warn'.")
            raise Exception(f"Dataset NOT saved due to {total_errors} errors. Modify source data or change check_mode to 'drop' or 'warn'.")
    
        elif mode == "drop":
            logger.warning(f"Dataset contains {total_errors} errors. Dropping {total_errors} erroneous samples...")
            err_idxs = err_other_role_idxs + err_human_starts_idxs + err_not_alternating_idxs + err_empty_conversation
            err_idxs = list(dict.fromkeys(err_idxs))
            for idx in sorted(err_idxs, reverse=True):
                del data[idx]

        elif mode == "warn":
            logger.warning(f"Dataset contains {total_errors} errors. Continuing with normal execution.")

    else:
        logger.info("No errors found. No log file created.")

    return data



def read_json(data_path: str) -> tuple[list, dict]:
    logger.info("Reading dataset...")
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
    logger.info("Preparing and adapting data fields...")
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
            

    logger.info("Checking dataset...")
    prep_data = check_data(data=prep_data, mode=args.check_mode)
    return prep_data
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, required=True, help="Source data path (can be a directory for Aya dataset type)")
    parser.add_argument("--output-path", type=str, required=True, help="Path where the output file will be saved, containing the desired file name.")
    parser.add_argument("--lang", type=str, required=True, help="ISO language code of the language of the dataset (set to 'mm' for multilingual datasets)")
    parser.add_argument("--lang-field", type=str, required=False, default="lang", help="Lang field from source data. Default: 'lang'")
    parser.add_argument("--humman", type=str, required=False, default="'{istruction}'",help="Humman field that can combine multiple fields. Default: '{instruction}'. E.g, '{prompt}\\n\\nContext:\\n{context}\\n\\nQuestion:\\n{instruction}'")
    parser.add_argument("--assistent", type=str, required=False, default="response",help="Lang field from source data. Default: 'response'")
    parser.add_argument("--check-mode", type=str, default="err", required=False, help="Mode used when checking prepared data. Options: 'err', 'drop', 'warn'")
    
    args = parser.parse_args()
    prep_data = prepare_basic(args)

    logger.info("Saving prepared dataset...")
    with open(args.output_path, 'w') as out_file:
        out_file.write(json.dumps(prep_data, indent=2, ensure_ascii=False))
    logger.info(f"Prepared dataset saved in {args.output_path}")