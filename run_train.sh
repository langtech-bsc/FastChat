# !/bin/bash

# prefix="salamandra7b_baseline_ca_v0.1"
prefix="salamandra7b_rag_ca-en-es_v0.2"

export PATH_RESULTS="./"$prefix
mkdir -p $WANDB_DIR


export RANK=2
export WORLD_SIZE=30



torchrun $DIST_ARGS -m fastchat.train.train \
    --deepspeed ds_type3_config_autombs.json \
    --model_name_or_path $HOME/Documents/bsc_2b_hf \
    --data_paths \
        $HOME/Documents/langtech/demo_data.json \
    --eval_data_paths \
        $HOME/Documents/langtech/demo_data.json \
    --bf16 True \
    --output_dir $PATH_RESULTS \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "steps" \
    --eval_steps 0.25 \
    --save_strategy "steps" \
    --save_steps 0.25 \
    --save_total_limit 20 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --add_chat_template True \
    --lazy_preprocess False \
    --local_rank -1 