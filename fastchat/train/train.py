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
import os, pathlib, gc, sys
from typing import Dict, Optional, Sequence
from enum import auto, IntEnum
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import transformers
import deepspeed
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother
import importlib.resources
import typing
import os
import psutil
import timeit
import sys 

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# current_directory = os.path.dirname(os.path.realpath(__file__))
# parent_directory = os.path.dirname(current_directory + '/../../..')
# sys.path.append(parent_directory)

from fastchat.conversation import SeparatorStyle, get_conv_template, Conversation

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
SEED = 42
    
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
        default=True, 
        metadata={
            "help": f"Whether or not to add and train the model with the chat template"
        }
    )

    update_model: Optional[bool] = field(
        default=False, 
        metadata={
            "help": f"This will save model with if vocab size is modified."
        }
    )

    function_calling: Optional[bool] = field(
        default=False, 
        metadata={
            "help": f"Whether or not to add and train the model with function calling template template"
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
    tools_paths: list[str] = field(
        default=None, metadata={"help": "List of tools to add to normal conversations."}
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


@dataclass
class LoraArguments:
    lora:Optional[bool] = field(
        default=False, 
        metadata={
            "help": f"Whether or not to lora"
        }
    )
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with deepspeed.zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

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
    conv:Conversation,
) -> Dict:
    if len(conv.roles) > 2:
        roles = {"human": conv.roles[0], "gpt": conv.roles[1], "tool": conv.roles[2]}
    else:
        roles = {"human": conv.roles[0], "gpt": conv.roles[1], "tool": None}

    conversations = []
    
    conversations_key = "conversations"
    for i, raw in enumerate(data):
        try:
            source = raw[conversations_key]
            metadata = {k: v for k, v in raw.items() if k != conversations_key}
            conv.messages = []

            if source[0]["from"] == conv.system_role:
                # If first is system role append it.
                conv.append_message(conv.system_role, source[0]["value"])
                source = source[1:]
            
            if roles[source[0]["from"]] != roles["human"]:
                # If first is not user
                source = source[1:]

            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                
                if j == 0:
                    assert role == roles["human"], f"{i}, \nErroneous source: {source}"
                else:
                    old_role = roles[source[j-1]["from"]]
                    if old_role == roles["human"]: # After user must be assistant
                        assert role == roles["gpt"], f"{i}, \nErroneous source: {json.dumps(source, indent=4)}"
                    elif old_role == roles["gpt"]: # After assistant, must be user or tool role
                        assert role in [roles["human"], roles["tool"]] and role != None, f"{i}, \nErroneous source: {source}"
                    else: # If previous role was tool, next must be assistant or tool
                        assert role in [roles["gpt"], roles["tool"]] and role != None, f"{i}, \nErroneous source: {source}"

                if sentence.get("tool_calls") is not None and len(sentence["tool_calls"]) > 0:
                    conv.append_message(role, sentence["tool_calls"])
                else:
                    conv.append_message(role, sentence["value"])
        except Exception as Err:
            print("ERROR ON CONV:", raw)
            raise Err
        
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
    # regex = conv.get_optimization_parts(tokenizer)
    assistant_start, end = conv.assistant_start, conv.sep2
    for j, (conversation, target) in enumerate(zip(conversations, targets)):
        if 0 in att_masks[i]:
            total_len = int(target.ne(tokenizer.pad_token_id).sum())
            bos = 0
            gemma_tok = type(tokenizer) == transformers.models.gemma.tokenization_gemma.GemmaTokenizer
            gemma_tok_fast = type(tokenizer) == transformers.models.gemma.tokenization_gemma_fast.GemmaTokenizerFast
            llama_tok_fast = type(tokenizer) == transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast # Mistral tokenizer
            has_bos = target[0] in list(tokenizer.all_special_ids) # To debug
            if gemma_tok or gemma_tok_fast or llama_tok_fast or has_bos:
                bos = 1 # Model's Tokenizer adds bos token

            cur_len = bos
               
            splits = conversation.split(assistant_start)
            assistant_starts = [part + assistant_start for part in splits[:-1]]  # Add delimiter to all but the last part
            assistant_starts.append(splits[-1])
            
            cur_len += len(tokenizer.tokenize(assistant_starts[0]))
            target[:cur_len] = IGNORE_TOKEN_ID
            for assistant_start_tmp in assistant_starts[1:]:
                end_index = assistant_start_tmp.find(end) + len(end)
                cur_len += len(tokenizer.tokenize(assistant_start_tmp[:end_index]))

                to_ignore = len(tokenizer.tokenize(assistant_start_tmp[end_index:]))
                target[cur_len: cur_len + to_ignore] = IGNORE_TOKEN_ID
                cur_len += to_ignore

            target[cur_len:] = IGNORE_TOKEN_ID
            
            # ignore_parts = []
            # start = 0
            # cur_len = bos
            # matches = [(l.start(1), l.end(1), l.group(1)) for l in list(re.finditer(regex, conversation))]
            # for match in matches:
            #     end = match[0]
            #     end_token = cur_len + len(tokenizer.tokenize(conversation[start: end]))
            #     ignore_parts.append((cur_len, end_token))
            #     start = match[1]
            #     cur_len = bos + len(tokenizer.tokenize(conversation[:start]))

            # end = start + len(conversation[start:])
            # end_token = cur_len + len(tokenizer.tokenize(conversation[start: end]))
            # ignore_parts.append((cur_len, end_token))
            # cur_len = bos + len(tokenizer.tokenize(conversation))
            # ignore_parts.append((cur_len, None))
            # for ignore in ignore_parts:
            #     target[ignore[0]: ignore[1]] = IGNORE_TOKEN_ID

            if not optimization_printed and local_rank == 0:
                optimization = tokenizer.decode([el for el in target if el != IGNORE_TOKEN_ID])
                print(f"\nCONVERSATION:\n{conversation}\nOptimization in ---------->\n'{optimization}'\n")
                optimization_printed = True

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}:\nCONVERSATION (mismatch):\n{conversation}\n====================\n"
                        # f" #turn = {len(turns) - 1}. (ignored)"
                    )

        else: # conversation was truncated
            target[:] = IGNORE_TOKEN_ID # Ignoring it
            print(f"WARNING: Filtered conversation due to truncated:\nCONVERSATION (truncated):\n{conversation}\n====================\n")


    
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
    rank0_print(f"EVAL:\n{data_args.eval_data_paths}")
    rank0_print('RAM memory % used:', psutil.virtual_memory()[2], '%')
    rank0_print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    tools = []
    for tools_path in data_args.tools_paths if data_args.tools_paths else []:
        data_loaded = json.load(open(tools_path, "r"))
        tools += data_loaded

    add_tools = False
    add_tools = len(tools) > 0
    narray = list(range(0, len(tools)))
    random.shuffle(narray)
    
    random.seed(SEED)
    start_time = timeit.default_timer()
    train_json = []
    eval_json = []
    for data_path in data_args.data_paths: # BSC: To combine different data files
        data_loaded = json.load(open(data_path, "r"))
        if not data_args.eval_data_paths:
            random.shuffle(data_loaded)
            eval_length = int(len(data_loaded) * 0.05)
            eval_json += data_loaded[:eval_length]
            data_loaded = data_loaded[eval_length:]
        
        train_json += data_loaded
    
    random.shuffle(train_json)
    # train_json = train_json[:500] # TODO: DELETE AFTER TESTS!!

    if data_args.eval_data_paths:
        eval_json = []
        for eval_data_path in data_args.eval_data_paths: # BSC: To combine different data files
            eval_json += json.load(open(eval_data_path, "r"))
        eval_size = max(500, int(len(train_json) / 10)) # limiting size of eval dataset to 10% of train set
        eval_json = eval_json[:eval_size]

    random.shuffle(eval_json)
    if add_tools:
        print("ADDING tools to train data")
        j = 0
        for i, row in enumerate(train_json):
            if i%10 == 0 and not row.get("tools"):
                row["tools"] = tools[narray[j]]
                j = (j + 1) % len(narray)

    if add_tools:
        print("ADDING tools to eval data")
        j = 0
        for i, row in enumerate(eval_json):
            if i%10 == 0 and not row.get("tools"):
                row["tools"] = tools[narray[j]]
                j = (j + 1) % len(narray)

    elapsed = timeit.default_timer() - start_time
    rank0_print(f">>>>>LOAD DATA TIME: {elapsed} sec")

    # train_json = train_json[:4]
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, conv=conv)
    rank0_print(f"Train Dataset: {len(train_dataset)}")

    elapsed = timeit.default_timer() - start_time
    rank0_print(f">>>>>PREPARED DATA TIME: {elapsed} sec")
    
    eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, conv=conv)
    rank0_print(f"Eval Dataset: {len(eval_dataset)}")

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def get_tokenizer(model_args):

    tokenizer_name_or_path = model_args.model_name_or_path
    if model_args.tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.tokenizer_name_or_path

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=True,
        trust_remote_code=model_args.trust_remote_code,
        add_prefix_space=False,
    )


    unk_token = "<unk>"
    eos_token = '<|im_end|>'
    start_token = '<|im_start|>'
    new_special_tokens = list(set([start_token, eos_token, unk_token]) - set(tokenizer.all_special_tokens))


    tokens_modified = False
    if new_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens}, replace_additional_special_tokens=False)
        tokens_modified = True
        
    if tokenizer.eos_token != eos_token:
        tokenizer.eos_token = eos_token # eos_token_id is set internaly.
        # tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token) #It's done internaly automatically

    if not tokenizer.unk_token:
        print("ADDING UNK TOKEN")
        tokenizer.unk_token = unk_token
        tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids(unk_token)
        tokens_modified = True
    
    # if not tokenizer.pad_token:
    #     pad_token = "<pad>"
    #     tokenizer.pad_token = pad_token
    #     tokenizer.add_tokens(list(set([pad_token]) - set(tokenizer.all_special_tokens)))
    #     tokens_modified = True

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.function_calling:
        conv = get_conv_template("chatml_func_template")
        tokenizer.add_tokens(
            list(
                set(
                    ["<tool_call>", "</tool_call>", "<tools>", "</tools>", "<tool_response>", "</tool_response>"]
                )
                - set(tokenizer.all_special_tokens)
            )
        )
        tokens_modified = True
    else:
        conv = get_conv_template("chatml_template")

    if model_args.add_chat_template: # BSC: for using chat template
        tokenizer.chat_template = conv.chat_template

    return tokenizer, tokens_modified, conv


def get_model_dtype(model):
    for p in model.parameters():
        return p.dtype
    return torch.float32  # fallba

def update_model(model_args, training_args):
    # 1) Kill any existing process group (if you launched inside a SLURM/MPI job).
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    # 2) Scrub env so Accelerate/DS doesn't engage in this single-process path.
    # for k in ("ACCELERATE_USE_DEEPSPEED", "DEEPSPEED_CONFIG_FILE", "DEEPSPEED_ZERO_STAGE"):
    #     os.environ.pop(k, None)
    # for k in list(os.environ):
    #     if k.startswith(("OMPI_", "PMI_", "PMIX_", "MPI_")):
    #         os.environ.pop(k, None)

    # os.environ["ACCELERATE_DISABLE_WEIGHTS_INIT"] = "1"  # avoid accelerate zero-init
    # os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    # If you want to be extra-safe, also ensure this temp path doesn't use DS config:
    # try:
    #     training_args.deepspeed = None
    # except Exception:
    #     pass

    # ---- now do your tokenizer/config/model load ----
    tokenizer, tokens_modified, conv = get_tokenizer(model_args=model_args)
    if not tokens_modified:
        return str(model_args.model_name_or_path)
    
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.use_cache = False

    print("Loading model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        low_cpu_mem_usage=True,   # force a plain (non-accelerate) load
        device_map="auto",           # don't shard
        offload_folder="offload"
    )
    # --- 4) Apply tokenizer changes to weights + config ---
    if tokens_modified:
        model.resize_token_embeddings(len(tokenizer))

    model.config.vocab_size = len(tokenizer)
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
        except Exception:
            pass

    # --- 5) Save to output_dir/base-model with strict dir policy ---
    out_dir = pathlib.Path(training_args.output_dir)
    base_dir = out_dir
    if base_dir.exists() and any(base_dir.iterdir()):
        raise FileExistsError(f"Base model directory already exists and is not empty: {base_dir}")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Consolidate to CPU to guarantee a full checkpoint
    # with torch.no_grad():
    #     state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if getattr(training_args, "bf16", False):
        save_dtype = torch.bfloat16
    elif getattr(training_args, "fp16", False):
        save_dtype = torch.float16
    else:
        save_dtype = torch.float32  # fallback

    print("Saving model...:", out_dir)
    model.save_pretrained(
        base_dir,
        safe_serialization=True,
        max_shard_size="5GB",
        torch_dtype=save_dtype
    )
    tokenizer.save_pretrained(base_dir)

    # --- 6) Cleanup ---
    try:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

    return str(base_dir)


def train(model_args, data_args, training_args, lora_args):
    # tokenizer path override (if provided)
    tokenizer, tokens_modified, conv = get_tokenizer(model_args=model_args)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    target_ctx = training_args.model_max_length
    
    # New add
    if getattr(config, "sliding_window", None):
        if config.sliding_window and config.sliding_window < target_ctx:
            config.sliding_window = target_ctx

    if orig_ctx_len and  target_ctx > orig_ctx_len:
        scaling_factor = float(target_ctx) / float(orig_ctx_len) #float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor} #{"type": "linear", "factor": scaling_factor}
        config.max_position_embeddings = target_ctx
    config.use_cache = False
    

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    try:
        model.generation_config.max_length = training_args.model_max_length
        # Alinear IDs por si cambiaste tokens especiales
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    except Exception:
        pass

    if tokens_modified:
        rank0_print("Tokens modified; resizing embeddings for base model")
        model.resize_token_embeddings(len(tokenizer))

    # if training_args.deepspeed is not None and training_args.local_rank == 0:
    #     model.print_trainable_parameters()
    
    if lora_args.lora and training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    
    # if tokens_modified:
    #     model.resize_token_embeddings(len(tokenizer))
    
    # Load data

    if lora_args.lora:
        rank0_print("Adding lora config")
        rank0_print("==================")
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, conv=conv)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    model.config.use_cache = False

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()

    if lora_args.lora:
        # check if zero3 mode enabled
        if is_deepspeed_zero3_enabled():
            # use deepspeed engine internal function to gather state dict
            # state_dict_zero3 contains whole parameters of base and lora adapters
            # we will not extract lora parameters since peft save_pretrained will do that
            # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
            # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
            state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
            if training_args.local_rank == 0:
                state_dict = state_dict_zero3
        else:
            # in other mode we use original code from fastchat team, to make sure our change is minimum
            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), lora_args.lora_bias
            )
        if training_args.local_rank == 0:
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            tokenizer.save_pretrained(training_args.output_dir)
    else:
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

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    local_rank = os.environ.get("SLURM_PROCID", os.environ.get("SLURM_PROCID", None))
    if local_rank is None:
        local_rank = training_args.local_rank
    
    local_rank = int(local_rank)

    if model_args.update_model:
        # Expect to run on a single process (rank 0 or -1). No DS/FSDP here.
        saved_path = update_model(model_args, training_args)
        print(saved_path)
        # If you want to stop here (recommended):
        sys.exit(0)
        
    rank0_print("MODEL ARGS: ", model_args)
    rank0_print("DATA ARGS: ", data_args)
    rank0_print("TRAINING ARGS: ", training_args)
    rank0_print("LORA ARGS: ", lora_args)

    train(model_args, data_args, training_args, lora_args)
