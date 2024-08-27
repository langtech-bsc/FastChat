# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

import sys
#sys.path.append("/mnt/netapp1/Proxecto_NOS/bsc/sft/it-chat-v1")
# sys.path.append("/gpfs/projects/bsc88/instruction-tuning/it-chat-v1")
sys.path.append("/gpfs/projects/bsc88/text/models/instruction-tuning/it-chat-v1")

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template
import random; random.seed(88)

from tqdm import tqdm
import psutil

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

CHAT_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    add_chat_template: Optional[bool] = field(
        default=False, 
        metadata={
            "help": f"Whether or not to add and train the model with the chat template: {CHAT_TEMPLATE}"
        }
    )


@dataclass
class DataArguments:
    data_paths: list[str] = field(
        default=None, metadata={"help": "Paths to the training data."}
    )
    eval_data_paths: Optional[list[str]] = field(
        default=None, metadata={"help": "Paths to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()

def preprocess_BSC_chat_template(
    data,
    tokenizer: transformers.PreTrainedTokenizer,
    conv_template = "bsc_chat_template",
) -> Dict:
    # print("Iniciating Preprocessing")
    # print('RAM memory % used:', psutil.virtual_memory()[2], '%')
    # print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    conv = get_conv_template(conv_template) # BSC Comment: This might need to be changed depending on the conversation we are looking for, more info in conversation.py
    # if tokenizer.eos_token != conv.sep2:
    #         conv.sep2 = tokenizer.eos_token

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    

    sources = [source["conversations"] for source in data] # only conversations part from data
    metadata = deepcopy(data)
    for source in metadata: 
        source.pop("conversations")

    # print("Sources + Metadata from data (Done)")
    # print('RAM memory % used:', psutil.virtual_memory()[2], '%')
    # print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    del data

    # print("data deleted (Done)")
    # print('RAM memory % used:', psutil.virtual_memory()[2], '%')
    # print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    # Apply prompt templates
    # print("Applying prompt templates...")
    conversations = []
    # for i, source in enumerate(tqdm(sources)):
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}, \nErroneous source: {source}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt(tokenizer=tokenizer, metadata=metadata[i]))
    
    # print("Applied prompt templates (Done) - Tokenizing...")
    # print('RAM memory % used:', psutil.virtual_memory()[2], '%')
    # print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    # BSC: Filter out conversations with len > model_max_length
    # print("Filtering too long conversations...")
    # dropped_conv_counter = 0
    # convs_to_drop_idxs = []
    # for i, conversation in enumerate(tqdm(conversations)):
    #     conv_len = len(tokenizer(conversation).input_ids)
    #     if conv_len > tokenizer.model_max_length:
    #         convs_to_drop_idxs.append(i)
    #         dropped_conv_counter +=1
    # for idx in sorted(convs_to_drop_idxs, reverse=True):
    #     del conversations[idx]
    # print(f">>>TOTAL DISCARDED CONVERSATIONS (due to > model_max_length): {dropped_conv_counter}")
    # assert conversations != [], "No sequences to tokenize (all filtered due to > model_max_length) - Check dataset or consider setting lazy_preprocess = False, to avoid this error."

    # Tokenize conversations
    tok_output = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, # Ingnoring truncated sequences later when computing mask
    )
    input_ids = tok_output.input_ids
    att_masks = tok_output.attention_mask # Only used for ignoring truncated sequences.
    # print("Before cloning")
    # print('RAM memory % used:', psutil.virtual_memory()[2], '%')
    # print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    targets = input_ids.clone()

    # print("Tokenized (Done) - Masking...")
    # print('RAM memory % used:', psutil.virtual_memory()[2], '%')
    # print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    optimization_printed=False
    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + "\n"
    dropped_conv_counter = 0
    for i, (conversation, target) in enumerate(zip(conversations, targets)):

        if 0 in att_masks[i]: # padding was applied >> conversation was not truncated
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            cur_len = 0
            
            gemma_tok = type(tokenizer) == transformers.models.gemma.tokenization_gemma.GemmaTokenizer
            gemma_tok_fast = type(tokenizer) == transformers.models.gemma.tokenization_gemma_fast.GemmaTokenizerFast
            llama_tok_fast = type(tokenizer) == transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast # Mistral tokenizer
            if (gemma_tok or gemma_tok_fast or llama_tok_fast) and tokenizer.add_bos_token:
                cur_len += 1 # Gemma Tokenizer adds bos token
            target[:cur_len] = IGNORE_TOKEN_ID

            for i, turn in enumerate(turns):

                if turn == "":
                    break
                
                # We also add the separator that marks the end of turn (<|im_end|>) to teach the model how to stop.
                turn += conv.sep2
                turn_len = len(tokenizer.tokenize(turn))

                role = turn.split(conv.sep)[1].split("\n")[0]

                if role == conv.system_role or role == conv.roles[0]: # system or user turn
                    instruction_len = len(tokenizer.tokenize(turn))
                elif role == conv.roles[1]: # assistant turn
                    instruction_len = len(tokenizer.tokenize(conv.sep + role + "\n"))
                # Ignore the user instructions
                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += turn_len
                
                
            target[cur_len:] = IGNORE_TOKEN_ID

            if not optimization_printed:
                print("\n CONVERSATION: \n", conversation, "\n Optimization in ---------->",tokenizer.decode([el for i,el in enumerate(target) if el != -100]), "\n")
                # print("\n Optimization in ---------->",tokenizer.decode([el for i,el in enumerate(target) if el != -100]), "\n")
                optimization_printed=True
            
            if False:  # Inspect and check the correctness of masking
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                rank0_print(tokenizer.decode(z))
                exit()

            if cur_len < tokenizer.model_max_length:
                # if cur_len -i != total_len: # The -i is because we also added the endoftext to the i-1 answers
                if cur_len != total_len: # The -i is because we also added the endoftext to the i-1 answers
                    target[:] = IGNORE_TOKEN_ID
                    rank0_print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored)"
                    )
                    # z = target.clone()
                    # z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                    # rank0_print(tokenizer.decode(z))
        else: # conversation was truncated
            target[:] = IGNORE_TOKEN_ID # Ignoring it
            dropped_conv_counter +=1
            # print(f"Filtered conversation:\n{conversation}")

    # print(f">>>TOTAL IGNORED CONVERSATIONS (due to > model_max_length): {dropped_conv_counter} ({100*dropped_conv_counter/len(conversations)}% of total)")
    
    # print("Preprocessed (Done)")
    # print('RAM memory % used:', psutil.virtual_memory()[2], '%')
    # print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def preprocess_BSC(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    conv_template = "vicuna",
) -> Dict:
    print(">> WARNING: data processing with add_chat_templte=False is NOT up to date. For advanced features such as different system_prompt languages and formatting dependant on the example use add_chat_template=True")
    conv = get_conversation_template(conv_template) # BSC Comment: This might need to be changed depending on the conversation we are looking for, more info in conversation.py
    if conv_template == "vicuna":
        if tokenizer.eos_token != conv.sep2:
            conv.sep2 = tokenizer.eos_token

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID

        for i, turn in enumerate(turns):

            if turn == "":
                break
            
            # We also add the separator that marks the endofsentece to teach the model how to stop.
            turn += conv.sep2
            turn_len = len(tokenizer(turn).input_ids)
            parts = turn.split(sep)

            if len(parts) != 2:
                break
            parts[0] += sep

            if i==0:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            else:
                # instruction_len = len(tokenizer(parts[0]).input_ids) - 1 # For Flor
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2 # For Gemma

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

            if i == 0:
                cur_len += turn_len -1
            else:
                # cur_len += turn_len # For Flor
                cur_len += turn_len - 1 # For Gemma
            
        target[cur_len:] = IGNORE_TOKEN_ID

        print("\n CONVERSATION: ", conversation)
        print("\n Optimization in ---------->",tokenizer.decode([el for i,el in enumerate(target) if el != -100]), "\n")
        
        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            # if cur_len -i != total_len: # The -i is because we also added the endoftext to the i-1 answers
            if cur_len != total_len: # for Gemma model
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2


            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        if tokenizer.chat_template: # BSC
            data_dict = preprocess_BSC_chat_template(raw_data, tokenizer)
        else:
            sources = [example["conversations"] for example in raw_data]
            data_dict = preprocess_BSC(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        if self.tokenizer.chat_template: # BSC
            ret = preprocess_BSC_chat_template([self.raw_data[i]], self.tokenizer)
        else:
            ret = preprocess_BSC([self.raw_data[i]["conversations"]], self.tokenizer)
        
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    rank0_print(f"DATA:\n{data_args.data_paths}")
    rank0_print('RAM memory % used:', psutil.virtual_memory()[2], '%')
    rank0_print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    import timeit
    start_time = timeit.default_timer()
    train_json = []
    for data_path in data_args.data_paths: # BSC: To combine different data files
        train_json += json.load(open(data_path, "r"))
    random.shuffle(train_json)
    # train_json = train_json[:3000] # TODO: DELETE AFTER TESTS!!

    elapsed = timeit.default_timer() - start_time
    rank0_print(f">>>>>LOAD DATA TIME: {elapsed} sec")

    # train_json = train_json[:4]
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)
    rank0_print(f"Train Dataset: {len(train_dataset)}")

    elapsed = timeit.default_timer() - start_time
    rank0_print(f">>>>>PREPARED DATA TIME: {elapsed} sec")

    if data_args.eval_data_paths:
        eval_json = []
        for eval_data_path in data_args.eval_data_paths: # BSC: To combine different data files
            eval_json += json.load(open(eval_data_path, "r"))
        random.shuffle(eval_json)
        eval_size = int(len(train_dataset) / 10) # limiting size of eval dataset to 10% of train set
        eval_json = eval_json[:eval_size]
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
        rank0_print(f"Eval Dataset: {len(eval_dataset)}")
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)



def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = os.environ.get("SLURM_PROCID", None) if os.environ.get("SLURM_PROCID", None) else os.environ.get("RANK", None)
    if local_rank is None:
        local_rank = training_args.local_rank

    # BSC: get tokenizer path if different
    tokenizer_name_or_path = model_args.model_name_or_path
    if model_args.tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.tokenizer_name_or_path


    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False
    # Load model and tokenizer
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=training_args.cache_dir,
    #     trust_remote_code=model_args.trust_remote_code,
    # )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=True,
        trust_remote_code=model_args.trust_remote_code,
        add_prefix_space=None
    )

    special_tokens_to_add = ['<|im_start|>', '<|im_end|>']
    eos_token = '<|im_end|>'
    existing_special_tokens = set()

    # Iterate through dictionary values
    special_tokens_map = tokenizer.special_tokens_map
    for value in special_tokens_map.values():
        if isinstance(value, list):
            # Add tokens from list
            existing_special_tokens.update(value)
        elif isinstance(value, str):
            # Add single token
            existing_special_tokens.add(value)

    
    existing_additional_special_tokens = special_tokens_map.get('additional_special_tokens', [])
    tokens_to_add = list(set(special_tokens_to_add) - existing_special_tokens)

    # Add missing special tokens to the tokenizer, preserving the existing ones
    if tokens_to_add:
        tokens_to_add.sort()
        tokenizer.add_special_tokens({
            'additional_special_tokens': existing_additional_special_tokens + tokens_to_add
        })

        # model.resize_token_embeddings(len(tokenizer))
        
    
    if tokenizer.eos_token != eos_token:
        tokenizer.eos_token = eos_token
        # tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token) #It's done internaly automatically

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    
    conv = get_conv_template("chatml_template")
    if model_args.add_chat_template: # BSC: for using chat template
        tokenizer.chat_template = conv.chat_template
    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, conv=conv)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    # If deepspeed os from package or outside of package.
    # When you install it by pip install ., it will add deepspeed_configs/* to the package so you can read it.
    # If you need to pass config file from current directory you can do: --deepspeed ./config.json
    if '--deepspeed' in sys.argv:
        idx = sys.argv.index('--deepspeed') + 1
        if '/' not in sys.argv[idx] and '\\' not in sys.argv[idx] :
            with importlib.resources.path("fastchat.deepspeed_configs", sys.argv[idx]) as path:
                sys.argv[idx] = str(path)

    train()
