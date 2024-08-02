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
from enum import auto, IntEnum
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import re
import os
import psutil
import timeit
import sys 

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# current_directory = os.path.dirname(os.path.realpath(__file__))
# parent_directory = os.path.dirname(current_directory + '/../../..')
# sys.path.append(parent_directory)

from fastchat.conversation import SeparatorStyle, get_conv_template, Conversation

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
special_chars = r'.^$*+?{}[]\|()'
pattern = re.compile(f"([{re.escape(special_chars)}])")


def escape_special_characters_except_newline(text):
    return pattern.sub(r'\\\1', text)
    
class PreProcessStyle(IntEnum):
    """Separator styles."""

    BSC_CHAT = auto()
    DEFAULT = auto()

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_name_or_path: Optional[str] = field(
        default=None, 
        metadata={
            "help": f"Whether or not to add deferent tokenizer default: equal to model_name_or_path"
        }
    )
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
            "help": f"Whether or not to add and train the model with the chat template"
        }
    )


@dataclass
class DataArguments:
    data_paths: list[str] = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_paths: list[str] = field(
        default=None, metadata={"help": "Path to the evaluation data."}
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


def preprocess_bsc_chat(
    data,
    tokenizer: transformers.PreTrainedTokenizer,
    conv:Conversation
) -> Dict:
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    
    conversations = []
    
    conversations_key = "conversations"
    for i, raw in enumerate(data):
        source = raw[conversations_key]
        metadata = {k: v for k, v in raw.items() if k != conversations_key}
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}, \nErroneous source: {source}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt(tokenizer=tokenizer, metadata=metadata))
            

    # Tokenize conversations
    tok_output = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = tok_output.input_ids
    att_masks = tok_output.attention_mask # Only used for ignoring truncated sequences.
    targets = input_ids.clone()
    
    optimization_printed=False
    regex = conv.get_optimization_parts(tokenizer)
    for j, (conversation, target) in enumerate(zip(conversations, targets)):
        if 0 in att_masks[i]:
            total_len = int(target.ne(tokenizer.pad_token_id).sum())
            bos = 0
            gemma_tok = type(tokenizer) == transformers.models.gemma.tokenization_gemma.GemmaTokenizer
            gemma_tok_fast = type(tokenizer) == transformers.models.gemma.tokenization_gemma_fast.GemmaTokenizerFast
            llama_tok_fast = type(tokenizer) == transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast # Mistral tokenizer
            if gemma_tok or gemma_tok_fast or llama_tok_fast:
                bos = 1 # Gemma Tokenizer adds bos token
            target[:bos] = IGNORE_TOKEN_ID

            ignore_parts = []
            start = 0
            cur_len = bos
            matches = [(l.start(1), l.end(1), l.group(1)) for l in list(re.finditer(regex, conversation))]
            for match in matches:
                end = match[0]
                end_token = cur_len + len(tokenizer.tokenize(conversation[start: end]))
                ignore_parts.append((cur_len, end_token))
                start = match[1]
                cur_len = bos + len(tokenizer.tokenize(conversation[:start]))
                
            end = start + len(conversation[start:])
            end_token = cur_len + len(tokenizer.tokenize(conversation[start: end]))
            ignore_parts.append((cur_len, end_token))
            cur_len = bos + len(tokenizer.tokenize(conversation))
            ignore_parts.append((cur_len, None))
            for ignore in ignore_parts:
                target[ignore[0]: ignore[1]] = IGNORE_TOKEN_ID

            if not optimization_printed:
                print("\nCONVERSATION:\n" + conversation + "\nOptimization in ---------->\n" + tokenizer.decode([el for i,el in enumerate(target) if el != IGNORE_TOKEN_ID]) + "\n")
                optimization_printed = True

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    rank0_print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        # f" #turn = {len(turns) - 1}. (ignored)"
                    )

        else: # conversation was truncated
            target[:] = IGNORE_TOKEN_ID # Ignoring it
            rank0_print(f"WARNING: Filtered conversation due to truncated:\n{conversation}\n")


    
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conv_template("vicuna")
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

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, conv=None):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        self.conv=conv
        data_dict = preprocess_bsc_chat(raw_data, tokenizer, conv=self.conv)
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

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, conv=None):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.conv = conv

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess_bsc_chat([self.raw_data[i]], self.tokenizer, conv=self.conv)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, conv=None
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    rank0_print(f"DATA:\n{data_args.data_paths}")
    rank0_print('RAM memory % used:', psutil.virtual_memory()[2], '%')
    rank0_print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)


    start_time = timeit.default_timer()
    train_json = []
    for data_path in data_args.data_paths: # BSC: To combine different data files
        train_json += json.load(open(data_path, "r"))
    random.shuffle(train_json)
    # train_json = train_json[:1000] # TODO: DELETE AFTER TESTS!!

    elapsed = timeit.default_timer() - start_time
    rank0_print(f">>>>>LOAD DATA TIME: {elapsed} sec")

    # train_json = train_json[:4]
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, conv=conv)
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
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, conv=conv)
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
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=True,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    # BSC: Adding special tokens
    # tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>"]})
    # tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})
    # model.resize_token_embeddings(len(tokenizer))
    # model.config.eos_token_id = tokenizer.eos_token_id
    
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
    train()




# for j, (conversation, target) in enumerate(zip(conversations, targets)):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())
#         cur_len = 0
            
#         gemma_tok = type(tokenizer) == transformers.models.gemma.tokenization_gemma.GemmaTokenizer
#         gemma_tok_fast = type(tokenizer) == transformers.models.gemma.tokenization_gemma_fast.GemmaTokenizerFast
#         llama_tok_fast = type(tokenizer) == transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast # Mistral tokenizer
#         if gemma_tok or gemma_tok_fast or llama_tok_fast:
#             cur_len += 1 # Gemma Tokenizer adds bos token
#         target[:cur_len] = IGNORE_TOKEN_ID

#         turns = [text + conv.sep2 for text in conversation.split(conv.sep2) if text]
#         for turn in turns:
#             turn_len = len(tokenizer.tokenize(turn))
#             parts = turn.split(sep)
            
#             if len(parts) == 2:
#                 parts[0] += sep
#             elif len(parts) != 1:
#                 raise Exception(f"Error: found more the one or zero {sep} in Conversation:\n{conversation}\nTrun:\n{turn}")
            
#             instruction_len = len(tokenizer.tokenize(parts[0]))

#             # Ignore the user instructions
#             target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
#             print(cur_len, cur_len + instruction_len)
#             print()
#             cur_len += turn_len

#         target[cur_len:] = IGNORE_TOKEN_ID
#         print()

#         # print("\n CONVERSATION: \n", conversation, "\n Optimization in ---------->\n",tokenizer.decode([el for i,el in enumerate(target) if el != -100]), "\n")
#         # print("\n Optimization in ---------->",tokenizer.decode([el for i,el in enumerate(target) if el != -100]), "\n")
#         optimization_printed=True

#         if False:  # Inspect and check the correctness of masking
#             z = target.clone()
#             z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
#             rank0_print(tokenizer.decode(z))
#             exit()

#         if cur_len < tokenizer.model_max_length:
#             if cur_len != total_len:
#                 target[:] = IGNORE_TOKEN_ID
#                 rank0_print(
#                     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                     # f" #turn = {len(turns) - 1}. (ignored)"
#                 )
