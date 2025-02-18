import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "BSC-LT/salamandra-7b",
    cache_dir="",
    model_max_length=4096,
    padding_side="right",
    use_fast=True,
    trust_remote_code=True,
    add_prefix_space=False
)

unk_token = "<unk>"
eos_token = '<|im_end|>'
start_token = '<|im_end|>'
new_special_tokens = list(set([start_token, eos_token, unk_token]) - set(tokenizer.all_special_tokens))
model = transformers.AutoModelForCausalLM.from_pretrained(
    "",
    config=config,
    cache_dir=training_args.cache_dir,
    trust_remote_code=model_args.trust_remote_code,
)
tokens_modified = False
if new_special_tokens:
    tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens}, replace_additional_special_tokens=False)
    tokens_modified = True
    
if tokenizer.eos_token != eos_token:
    tokenizer.eos_token = eos_token
    # tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token) #It's done internaly automatically

print(tokenizer.unk_token)
if not tokenizer.unk_token:
    print("SETTING UNK TOKEN")
    tokenizer.unk_token = unk_token
    print(tokenizer.unk_token_id)
    tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    tokens_modified = True

# if not tokenizer.pad_token:
#     pad_token = "<PAD>"
#     tokenizer.pad_token = pad_token
#     tokenizer.add_tokens(list(set([pad_token]) - set(tokenizer.all_special_tokens)))
#     tokens_modified = True

# if tokenizer.pad_token != tokenizer.unk_token:
#     tokenizer.pad_token = tokenizer.unk_token